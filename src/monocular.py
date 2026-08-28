from dataclasses import dataclass

import cv2
import numpy as np
import scipy.ndimage as nd

from src.feature_tracker import FeatureTracker, FeatureTrackerConfig
from src.landmark_filter import LandmarkStatus, Sparse3DFilterBank, Sparse3DSettings
from src.mono_map import MonoMap, MonoMapConfig
from src.optimizer import PhotometricBA
from src.tracker import DirectTracker, PATTERN_DX, PATTERN_DY, ensure_pattern_intensities


@dataclass
class MonocularVOResult:
    T_W_C: np.ndarray
    observations: dict[int, np.ndarray]
    mature_tracks: int
    pending_tracks: int
    direct_landmarks: int
    direct_inliers: int
    direct_attempted: bool
    used_direct: bool
    essential_common_tracks: int
    essential_inliers: int
    essential_used: bool
    image_motion_fallback_used: bool
    bootstrap_active: bool
    motion_source: str
    motion_step_norm: float
    rotation_step_deg: float
    direct_hypotheses: int
    mono_keyframes: int
    mono_keyframe_inserted: bool
    mono_keyframe_reason: str
    mono_ba_ran: bool
    mono_ba_window: int
    mono_geo_ba_ran: bool
    mono_geo_ba_edges: int
    mono_depth_ba_ran: bool
    mono_depth_ba_landmarks: int
    mono_depth_ba_edges: int
    mono_depth_ba_cost_before: float
    mono_depth_ba_cost_after: float
    mono_depth_ba_updated: int
    mono_depth_ba_median_abs_log_update: float
    mono_depth_ba_max_abs_log_update: float
    mono_observations: int
    mono_map_landmarks: int
    mono_gftt_exclusions: int
    mono_keyframe_klt_tracks: int
    mono_keyframe_klt_inliers: int
    mono_keyframe_klt_used: bool
    mono_keyframe_klt_residual_median_px: float
    mono_keyframe_klt_residual_p90_px: float
    mono_keyframe_klt_flow_cos_median: float
    mono_keyframe_klt_flow_cos_p10: float
    mono_keyframe_pose_update_norm: float
    mono_keyframe_pose_update_rot_deg: float
    mono_direct_candidate_stats: list[dict]
    mono_keyframe_reference_counts: dict[int, int]
    mono_keyframe_visible_counts: dict[int, int]
    mono_keyframe_prefilter_kept_counts: dict[int, int]
    mono_keyframe_prefilter_rejected_counts: dict[int, int]
    mono_keyframe_low_usefulness_frames: dict[int, int]
    mono_keyframe_usefulness_ratio: dict[int, float]
    mono_direct_prefilter_kept: int
    mono_direct_prefilter_rejected: int
    mono_keyframe_discarded_id: int
    mono_keyframe_discard_reason: str


@dataclass
class MonoFrameObservation:
    landmark_id: int
    host_kf_id: int
    point_w: np.ndarray
    projected_pixel: np.ndarray
    initial_pixel: np.ndarray
    matched_pixel: np.ndarray
    photometric_error: float
    accepted: bool


class ExperimentalMonocularVO:
    """Left-image-only VO scaffold with arbitrary monocular scale."""

    def __init__(
        self,
        K: np.ndarray,
        feature_config: FeatureTrackerConfig | None = None,
        sparse_settings: Sparse3DSettings | None = None,
        mono_map_config: MonoMapConfig | None = None,
        min_essential_tracks: int = 20,
        min_direct_landmarks: int = 50,
        min_direct_inliers: int = 30,
        image_motion_fallback_depth: float = 4.0,
        max_bootstrap_step: float = 0.2,
        max_rotation_step_deg: float = 5.0,
        max_direct_step: float = 0.25,
        max_direct_rotation_step_deg: float = 10.0,
        recovery_rotation_step_deg: float = 1.0,
        ba_every_keyframes: int = 3,
    ):
        self.K = K
        self.feature_tracker = FeatureTracker(feature_config or FeatureTrackerConfig(max_features=1000))
        self.sparse3d = Sparse3DFilterBank(K, sparse_settings or Sparse3DSettings(
            min_track_length=3,
            birth_min_flow_px=0.5,
            conv_depth_variance=100.0,
        ))
        self.mono_map = MonoMap(K, mono_map_config or MonoMapConfig())
        self.direct_tracker = DirectTracker(K)
        self.ba = PhotometricBA(K)
        self.min_essential_tracks = min_essential_tracks
        self.min_direct_landmarks = min_direct_landmarks
        self.min_direct_inliers = min_direct_inliers
        self.image_motion_fallback_depth = image_motion_fallback_depth
        self.max_bootstrap_step = max_bootstrap_step
        self.max_rotation_step_rad = np.deg2rad(max_rotation_step_deg)
        self.max_direct_step = max_direct_step
        self.max_direct_rotation_step_rad = np.deg2rad(max_direct_rotation_step_deg)
        self.recovery_rotation_step_rad = np.deg2rad(recovery_rotation_step_deg)
        self.ba_every_keyframes = max(1, int(ba_every_keyframes))
        self.keyframes_since_ba = 0
        self.T_W_C = np.eye(4)
        self.last_T_W_C = self.T_W_C.copy()
        self.last_direct_delta = np.eye(4)
        self.have_direct_motion_model = False
        self.prev_observations: dict[int, np.ndarray] = {}
        self.current_a = 0.0
        self.current_b = 0.0
        self.last_essential_common_tracks = 0
        self.last_essential_inliers = 0
        self.last_essential_used = False
        self.bootstrap_complete = False
        self.last_keyframe_inserted = False
        self.last_keyframe_reason = ""
        self.last_ba_result = {"ran": False, "window": 0, "residuals": 0}
        self.last_geo_ba_result = {"ran": False, "window": 0, "edges": 0}
        self.last_depth_ba_result = {
            "ran": False,
            "window": 0,
            "landmarks": 0,
            "edges": 0,
            "cost_before": 0.0,
            "cost_after": 0.0,
            "updated": 0,
            "median_abs_log_depth_update": 0.0,
            "max_abs_log_depth_update": 0.0,
        }
        self.last_gftt_exclusions = 0
        self.last_keyframe_klt_tracks = 0
        self.last_keyframe_klt_inliers = 0
        self.last_keyframe_klt_used = False
        self.last_keyframe_klt_points_w = np.empty((0, 3), dtype=np.float64)
        self.last_keyframe_klt_prev_pixels = np.empty((0, 2), dtype=np.float64)
        self.last_keyframe_klt_pixels = np.empty((0, 2), dtype=np.float64)
        self.last_keyframe_klt_residual_median_px = 0.0
        self.last_keyframe_klt_residual_p90_px = 0.0
        self.last_keyframe_klt_flow_cos_median = 0.0
        self.last_keyframe_klt_flow_cos_p10 = 0.0
        self.last_keyframe_pose_update_norm = 0.0
        self.last_keyframe_pose_update_rot_deg = 0.0
        self.min_keyframe_klt_flow_cos = 0.2
        self.min_keyframe_klt_flow_gate_tracks = 50
        self.candidate_klt_residual_gate_tracks = 50
        self.candidate_klt_median_good_px = 1.0
        self.candidate_klt_p90_good_px = 3.0
        self.candidate_klt_median_penalty = 5.0
        self.candidate_klt_p90_penalty = 1.0
        self.direct_prefilter_abs_threshold = 45.0
        self.direct_prefilter_mad_factor = 3.0
        self.direct_prefilter_min_keep_ratio = 0.35
        self.last_direct_prefilter_kept = 0
        self.last_direct_prefilter_rejected = 0
        self.last_frame_observations: dict[int, MonoFrameObservation] = {}
        self.patch_match_search_radius = 2
        self.patch_match_max_error = 45.0

    def process(self, frame_id: int, image: np.ndarray) -> MonocularVOResult:
        exclusion_points = self._project_existing_map_landmarks(image.shape)
        self.last_gftt_exclusions = len(exclusion_points)
        observations = self.feature_tracker.update(image, exclusion_points=exclusion_points)

        image_motion_fallback_used = False
        motion_source = "hold"
        if self.bootstrap_complete:
            self._reset_essential_diagnostics(observations)
            T_W_C_guess = self.T_W_C.copy()
        else:
            T_W_C_guess = self._essential_pose_guess(observations)
            if T_W_C_guess is not None:
                motion_source = "essential_bootstrap"
            if T_W_C_guess is None:
                T_W_C_guess = self._image_motion_pose_guess(observations)
                image_motion_fallback_used = T_W_C_guess is not None
                if image_motion_fallback_used:
                    motion_source = "flow_bootstrap"
        if T_W_C_guess is None:
            T_W_C_guess = self.T_W_C.copy()

        used_direct = False
        direct_inliers = 0
        direct_landmarks = 0
        direct_attempted = False
        direct_hypotheses = 0
        direct_guesses = [("direct_bootstrap", T_W_C_guess)]
        if self.bootstrap_complete:
            direct_guesses = self._direct_pose_guesses()
            visible_refs = self.mono_map.visible_references(T_W_C_guess, image.shape)
            klt_pixel_guesses = self._keyframe_klt_pixel_guesses(image, visible_refs)
            matched_guess = self._matched_observation_pose_guess(image, T_W_C_guess, visible_refs, klt_pixel_guesses)
            if matched_guess is not None:
                direct_guesses = [("svo_match", matched_guess)] + direct_guesses

        old_T_W_C = self.T_W_C.copy()
        best = self._track_direct_candidates(image, direct_guesses)
        direct_hypotheses = best["hypotheses"]
        direct_landmarks = best["landmarks"]
        direct_attempted = best["attempted"]
        direct_inliers = best["inliers"]
        if best["accepted"]:
            self.T_W_C = best["T_W_C"]
            self.current_a = best["a"]
            self.current_b = best["b"]
            self.last_direct_delta = np.linalg.inv(old_T_W_C) @ self.T_W_C
            self.have_direct_motion_model = True
            used_direct = True
            motion_source = best["source"]
            self.last_keyframe_klt_used = best["source"] in {"keyframe_klt", "svo_match"}
        else:
            self.last_keyframe_klt_used = False
            if self.bootstrap_complete:
                self.T_W_C = old_T_W_C
            else:
                self.T_W_C = T_W_C_guess

        if used_direct and direct_inliers >= self.min_direct_inliers:
            self.bootstrap_complete = True
        elif not self.bootstrap_complete and direct_landmarks >= self.min_direct_landmarks * 2:
            self.bootstrap_complete = True

        if not self.bootstrap_complete or used_direct:
            self.sparse3d.update(
                frame_id,
                observations,
                self.T_W_C,
                remove_missing=not self.bootstrap_complete,
            )
        self._maybe_add_keyframe(frame_id, image, observations, used_direct, direct_inliers, direct_landmarks)
        motion_step_norm = float(np.linalg.norm(self.T_W_C[:3, 3] - self.last_T_W_C[:3, 3]))
        rotation_step_deg = float(np.rad2deg(_rotation_angle(self.last_T_W_C[:3, :3].T @ self.T_W_C[:3, :3])))
        self._update_keyframe_klt_residual_stats(image.shape)
        self.prev_observations = observations
        self.last_T_W_C = self.T_W_C.copy()
        return MonocularVOResult(
            T_W_C=self.T_W_C.copy(),
            observations=observations,
            mature_tracks=self.sparse3d.feature_count(),
            pending_tracks=len(self.sparse3d.pending),
            direct_landmarks=direct_landmarks,
            direct_inliers=direct_inliers,
            direct_attempted=direct_attempted,
            used_direct=used_direct,
            essential_common_tracks=self.last_essential_common_tracks,
            essential_inliers=self.last_essential_inliers,
            essential_used=self.last_essential_used,
            image_motion_fallback_used=image_motion_fallback_used,
            bootstrap_active=not self.bootstrap_complete,
            motion_source=motion_source,
            motion_step_norm=motion_step_norm,
            rotation_step_deg=rotation_step_deg,
            direct_hypotheses=direct_hypotheses,
            mono_keyframes=len(self.mono_map),
            mono_keyframe_inserted=self.last_keyframe_inserted,
            mono_keyframe_reason=self.last_keyframe_reason,
            mono_ba_ran=bool(self.last_ba_result["ran"]),
            mono_ba_window=int(self.last_ba_result["window"]),
            mono_geo_ba_ran=bool(self.last_geo_ba_result["ran"]),
            mono_geo_ba_edges=int(self.last_geo_ba_result["edges"]),
            mono_depth_ba_ran=bool(self.last_depth_ba_result["ran"]),
            mono_depth_ba_landmarks=int(self.last_depth_ba_result["landmarks"]),
            mono_depth_ba_edges=int(self.last_depth_ba_result["edges"]),
            mono_depth_ba_cost_before=float(self.last_depth_ba_result["cost_before"]),
            mono_depth_ba_cost_after=float(self.last_depth_ba_result["cost_after"]),
            mono_depth_ba_updated=int(self.last_depth_ba_result["updated"]),
            mono_depth_ba_median_abs_log_update=float(self.last_depth_ba_result["median_abs_log_depth_update"]),
            mono_depth_ba_max_abs_log_update=float(self.last_depth_ba_result["max_abs_log_depth_update"]),
            mono_observations=self.mono_map.observation_count(),
            mono_map_landmarks=self.mono_map.active_landmark_count(),
            mono_gftt_exclusions=self.last_gftt_exclusions,
            mono_keyframe_klt_tracks=self.last_keyframe_klt_tracks,
            mono_keyframe_klt_inliers=self.last_keyframe_klt_inliers,
            mono_keyframe_klt_used=self.last_keyframe_klt_used,
            mono_keyframe_klt_residual_median_px=self.last_keyframe_klt_residual_median_px,
            mono_keyframe_klt_residual_p90_px=self.last_keyframe_klt_residual_p90_px,
            mono_keyframe_klt_flow_cos_median=self.last_keyframe_klt_flow_cos_median,
            mono_keyframe_klt_flow_cos_p10=self.last_keyframe_klt_flow_cos_p10,
            mono_keyframe_pose_update_norm=self.last_keyframe_pose_update_norm,
            mono_keyframe_pose_update_rot_deg=self.last_keyframe_pose_update_rot_deg,
            mono_direct_candidate_stats=best.get("candidates", []),
            mono_keyframe_reference_counts=dict(self.mono_map.last_reference_counts),
            mono_keyframe_visible_counts=dict(self.mono_map.last_visible_counts),
            mono_keyframe_prefilter_kept_counts=dict(self.mono_map.last_prefilter_kept_counts),
            mono_keyframe_prefilter_rejected_counts=dict(self.mono_map.last_prefilter_rejected_counts),
            mono_keyframe_low_usefulness_frames=dict(self.mono_map.keyframe_low_usefulness_frames),
            mono_keyframe_usefulness_ratio=dict(self.mono_map.keyframe_usefulness_ratio),
            mono_direct_prefilter_kept=int(self.last_direct_prefilter_kept),
            mono_direct_prefilter_rejected=int(self.last_direct_prefilter_rejected),
            mono_keyframe_discarded_id=int(self.mono_map.last_discarded_kf_id),
            mono_keyframe_discard_reason=str(self.mono_map.last_discard_reason),
        )

    def _keyframe_klt_pose_guess(self, image: np.ndarray) -> np.ndarray | None:
        refs = self.mono_map.visible_references(self.T_W_C, image.shape)
        pixel_guesses = self._keyframe_klt_pixel_guesses(image, refs)
        return self._matched_observation_pose_guess(image, self.T_W_C, refs, pixel_guesses)

    def _keyframe_klt_pixel_guesses(self, image: np.ndarray, refs: list[dict]) -> dict[int, np.ndarray]:
        self.last_keyframe_klt_tracks = 0
        self.last_keyframe_klt_inliers = 0
        self.last_keyframe_klt_points_w = np.empty((0, 3), dtype=np.float64)
        self.last_keyframe_klt_prev_pixels = np.empty((0, 2), dtype=np.float64)
        self.last_keyframe_klt_pixels = np.empty((0, 2), dtype=np.float64)
        prev_img = self.feature_tracker.prev_img
        if prev_img is None or not refs:
            return {}

        pts_w_all = np.asarray([ref["point_w"] for ref in refs], dtype=np.float64)
        landmark_ids = np.asarray([ref["landmark_id"] for ref in refs], dtype=int)
        prev_uv, _, visible = self._project_points(self.T_W_C, pts_w_all, image.shape, margin=4)
        if np.sum(visible) < 8:
            return {}

        ids = landmark_ids[visible]
        pts_w = pts_w_all[visible].astype(np.float64)
        p0 = prev_uv[visible].astype(np.float32)
        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_img.astype(np.uint8),
            image.astype(np.uint8),
            p0.reshape(-1, 1, 2),
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if p1 is None or status is None:
            return {}

        p1 = p1.reshape(-1, 2)
        good = status.reshape(-1).astype(bool)
        h, w = image.shape
        good &= (p1[:, 0] >= 2) & (p1[:, 0] < w - 2)
        good &= (p1[:, 1] >= 2) & (p1[:, 1] < h - 2)
        self.last_keyframe_klt_tracks = int(np.sum(good))
        if np.sum(good) < 8:
            return {}

        tracked_points_w = pts_w[good]
        tracked_prev_pixels = p0[good].astype(np.float64)
        tracked_pixels = p1[good].astype(np.float64)
        tracked_ids = ids[good].astype(int)

        self.last_keyframe_klt_points_w = tracked_points_w
        self.last_keyframe_klt_prev_pixels = tracked_prev_pixels
        self.last_keyframe_klt_pixels = tracked_pixels
        return {int(fid): pixel.copy() for fid, pixel in zip(tracked_ids, tracked_pixels)}

    def _matched_observation_pose_guess(
        self,
        image: np.ndarray,
        T_W_C_guess: np.ndarray,
        refs: list[dict],
        pixel_guesses: dict[int, np.ndarray],
    ) -> np.ndarray | None:
        matched = self._match_visible_reference_patches(image, refs, pixel_guesses)
        accepted = [obs for obs in matched.values() if obs.accepted]
        self.last_frame_observations = matched
        self.last_keyframe_klt_inliers = len(accepted)
        if len(accepted) < max(12, self.min_direct_inliers // 2):
            return None

        points_w = np.asarray([obs.point_w for obs in accepted], dtype=np.float64)
        pixels = np.asarray([obs.matched_pixel for obs in accepted], dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_w,
            pixels,
            self.K,
            None,
            reprojectionError=3.0,
            confidence=0.99,
            iterationsCount=100,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None:
            return None

        self.last_keyframe_klt_inliers = int(len(inliers))
        if len(inliers) < max(12, self.min_direct_inliers // 2):
            return None
        inlier_idx = inliers.reshape(-1)
        self.last_keyframe_klt_points_w = points_w[inlier_idx].astype(np.float64)
        self.last_keyframe_klt_prev_pixels = np.asarray([obs.initial_pixel for obs in accepted], dtype=np.float64)[inlier_idx]
        self.last_keyframe_klt_pixels = pixels[inlier_idx].astype(np.float64)

        R_C_W, _ = cv2.Rodrigues(rvec)
        T_C_W = np.eye(4, dtype=np.float64)
        T_C_W[:3, :3] = R_C_W
        T_C_W[:3, 3] = tvec.reshape(3)
        T_W_C = np.linalg.inv(T_C_W)
        if not self._direct_step_is_plausible(T_W_C):
            return None
        return T_W_C

    def _match_visible_reference_patches(
        self,
        image: np.ndarray,
        refs: list[dict],
        pixel_guesses: dict[int, np.ndarray] | None = None,
    ) -> dict[int, MonoFrameObservation]:
        pixel_guesses = {} if pixel_guesses is None else pixel_guesses
        image_f = image.astype(np.float32)
        h, w = image.shape
        matched = {}
        radius = int(self.patch_match_search_radius)
        for ref in refs:
            fid = int(ref["landmark_id"])
            projected = np.asarray(ref["projected_pixel"], dtype=np.float64)
            initial = np.asarray(pixel_guesses.get(fid, projected), dtype=np.float64)
            best_pixel = initial.copy()
            best_error = float("inf")
            expected = (
                np.exp(self.current_a - float(ref["affine_a"]))
                * ensure_pattern_intensities(np.asarray(ref["intensity"], dtype=np.float32).reshape(1, -1))[0]
                + (self.current_b - float(ref["affine_b"]))
            )
            for du in range(-radius, radius + 1):
                for dv in range(-radius, radius + 1):
                    pixel = initial + np.array([du, dv], dtype=np.float64)
                    if pixel[0] < 2 or pixel[0] >= w - 2 or pixel[1] < 2 or pixel[1] >= h - 2:
                        continue
                    u_pat = pixel[0] + PATTERN_DX
                    v_pat = pixel[1] + PATTERN_DY
                    observed = nd.map_coordinates(image_f, [v_pat, u_pat], order=1)
                    error = float(np.mean(np.abs(observed - expected)))
                    if error < best_error:
                        best_error = error
                        best_pixel = pixel
            accepted = np.isfinite(best_error) and best_error <= self.patch_match_max_error
            matched[fid] = MonoFrameObservation(
                landmark_id=fid,
                host_kf_id=int(ref["host_kf_id"]),
                point_w=np.asarray(ref["point_w"], dtype=np.float64).copy(),
                projected_pixel=projected.copy(),
                initial_pixel=initial.copy(),
                matched_pixel=best_pixel.copy(),
                photometric_error=float(best_error),
                accepted=bool(accepted),
            )
        return matched

    def keyframe_klt_residual_segments(self, image_shape: tuple[int, int], max_segments: int = 400) -> dict[str, np.ndarray]:
        if len(self.last_keyframe_klt_points_w) == 0:
            return {
                "segments": [],
                "residuals": np.empty((0,), dtype=np.float64),
            }

        projected, _, visible = self._project_points(self.T_W_C, self.last_keyframe_klt_points_w, image_shape)
        if not np.any(visible):
            return {
                "segments": [],
                "residuals": np.empty((0,), dtype=np.float64),
            }

        observed = self.last_keyframe_klt_pixels[visible]
        projected = projected[visible]
        residuals = np.linalg.norm(observed - projected, axis=1)
        if len(residuals) > max_segments:
            idx = np.linspace(0, len(residuals) - 1, max_segments, dtype=int)
            observed = observed[idx]
            projected = projected[idx]
            residuals = residuals[idx]

        return {
            "segments": [np.array([obs, proj], dtype=np.float64) for obs, proj in zip(observed, projected)],
            "residuals": residuals.astype(np.float64),
        }

    def _update_keyframe_klt_residual_stats(self, image_shape: tuple[int, int]) -> None:
        residuals = self.keyframe_klt_residual_segments(image_shape)["residuals"]
        if len(residuals) == 0:
            self.last_keyframe_klt_residual_median_px = 0.0
            self.last_keyframe_klt_residual_p90_px = 0.0
            self.last_keyframe_klt_flow_cos_median = 0.0
            self.last_keyframe_klt_flow_cos_p10 = 0.0
            return
        self.last_keyframe_klt_residual_median_px = float(np.median(residuals))
        self.last_keyframe_klt_residual_p90_px = float(np.percentile(residuals, 90))

        flow_stats = self._keyframe_klt_flow_stats_for_pose(self.T_W_C, image_shape)
        if flow_stats["count"] == 0:
            self.last_keyframe_klt_flow_cos_median = 0.0
            self.last_keyframe_klt_flow_cos_p10 = 0.0
            return
        self.last_keyframe_klt_flow_cos_median = flow_stats["median"]
        self.last_keyframe_klt_flow_cos_p10 = flow_stats["p10"]

    def _keyframe_klt_flow_stats_for_pose(self, T_W_C: np.ndarray, image_shape: tuple[int, int]) -> dict[str, float]:
        if len(self.last_keyframe_klt_points_w) == 0:
            return {"count": 0, "median": 0.0, "p10": 0.0}

        projected, _, visible = self._project_points(T_W_C, self.last_keyframe_klt_points_w, image_shape)
        if not np.any(visible):
            return {"count": 0, "median": 0.0, "p10": 0.0}

        klt_flow = self.last_keyframe_klt_pixels[visible] - self.last_keyframe_klt_prev_pixels[visible]
        reproj_flow = projected[visible] - self.last_keyframe_klt_prev_pixels[visible]
        denom = np.linalg.norm(klt_flow, axis=1) * np.linalg.norm(reproj_flow, axis=1)
        valid = denom > 1e-6
        if not np.any(valid):
            return {"count": 0, "median": 0.0, "p10": 0.0}
        cos = np.sum(klt_flow[valid] * reproj_flow[valid], axis=1) / denom[valid]
        return {
            "count": int(len(cos)),
            "median": float(np.median(cos)),
            "p10": float(np.percentile(cos, 10)),
        }

    def _keyframe_klt_residual_stats_for_pose(self, T_W_C: np.ndarray, image_shape: tuple[int, int]) -> dict[str, float]:
        if len(self.last_keyframe_klt_points_w) == 0:
            return {"count": 0, "median": 0.0, "p90": 0.0}

        projected, _, visible = self._project_points(T_W_C, self.last_keyframe_klt_points_w, image_shape)
        if not np.any(visible):
            return {"count": 0, "median": 0.0, "p90": 0.0}

        residuals = np.linalg.norm(self.last_keyframe_klt_pixels[visible] - projected[visible], axis=1)
        if len(residuals) == 0:
            return {"count": 0, "median": 0.0, "p90": 0.0}
        return {
            "count": int(len(residuals)),
            "median": float(np.median(residuals)),
            "p90": float(np.percentile(residuals, 90)),
        }

    def _project_points(self, T_W_C: np.ndarray, points_w: np.ndarray, image_shape: tuple[int, int], margin: int = 0):
        if len(points_w) == 0:
            return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=bool)

        T_C_W = np.linalg.inv(T_W_C)
        points_c = points_w @ T_C_W[:3, :3].T + T_C_W[:3, 3]
        z = points_c[:, 2]
        valid_z = z > 0.01
        z_safe = np.clip(z, 0.01, None)
        u = self.K[0, 0] * points_c[:, 0] / z_safe + self.K[0, 2]
        v = self.K[1, 1] * points_c[:, 1] / z_safe + self.K[1, 2]
        h, w = image_shape
        valid = valid_z & (u >= margin) & (u < w - margin) & (v >= margin) & (v < h - margin)
        return np.stack([u, v], axis=1), z, valid

    def _project_existing_map_landmarks(self, image_shape: tuple[int, int]) -> np.ndarray:
        points_w, _, _, _ = self.mono_map.point_cloud()
        if len(points_w) == 0:
            return np.empty((0, 2), dtype=np.float64)

        uv, _, valid = self._project_points(self.T_W_C, points_w, image_shape)
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.float64)
        return uv[valid]

    def ekf_landmark_glyphs(self, image_shape: tuple[int, int], max_landmarks: int = 600) -> dict[str, np.ndarray]:
        mapped_ids = set()
        for kf in self.mono_map.keyframes:
            mapped_ids.update(int(fid) for fid in kf.landmark_ids)

        rows = []
        T_C_W = np.linalg.inv(self.T_W_C)
        h, w = image_shape
        for fid, lm in self.sparse3d.features.items():
            if int(fid) in mapped_ids:
                continue
            try:
                point_a = lm.anchor_point()
                point_w = lm.anchor_T_W_C[:3, :3] @ point_a + lm.anchor_T_W_C[:3, 3]
                point_c = T_C_W[:3, :3] @ point_w + T_C_W[:3, 3]
            except (ValueError, np.linalg.LinAlgError):
                continue
            if point_c[2] <= 0.01:
                continue

            u = self.K[0, 0] * point_c[0] / point_c[2] + self.K[0, 2]
            v = self.K[1, 1] * point_c[1] / point_c[2] + self.K[1, 2]
            if u < 0 or u >= w or v < 0 or v >= h:
                continue

            T_C_A = T_C_W @ lm.anchor_T_W_C
            depth_std = float(np.sqrt(max(lm.depth_variance(T_C_A), 0.0)))
            range_std = float(np.sqrt(max(lm.range_variance(T_C_A), 0.0)))
            rows.append((
                point_w,
                point_c,
                np.array([u, v], dtype=np.float64),
                depth_std,
                range_std,
                int(lm.status == LandmarkStatus.MATURE),
            ))
            if len(rows) >= max_landmarks:
                break

        if not rows:
            return {
                "points_w": np.empty((0, 3), dtype=np.float64),
                "points_c": np.empty((0, 3), dtype=np.float64),
                "pixels": np.empty((0, 2), dtype=np.float64),
                "depth_std": np.empty((0,), dtype=np.float64),
                "range_std": np.empty((0,), dtype=np.float64),
                "is_mature": np.empty((0,), dtype=bool),
            }

        points_w, points_c, pixels, depth_std, range_std, is_mature = zip(*rows)
        return {
            "points_w": np.asarray(points_w, dtype=np.float64),
            "points_c": np.asarray(points_c, dtype=np.float64),
            "pixels": np.asarray(pixels, dtype=np.float64),
            "depth_std": np.asarray(depth_std, dtype=np.float64),
            "range_std": np.asarray(range_std, dtype=np.float64),
            "is_mature": np.asarray(is_mature, dtype=bool),
        }

    def _maybe_add_keyframe(
        self,
        frame_id: int,
        image: np.ndarray,
        observations: dict[int, np.ndarray],
        used_direct: bool,
        direct_inliers: int,
        direct_landmarks: int,
    ) -> None:
        self.last_keyframe_inserted = False
        self.last_keyframe_reason = ""
        self.last_ba_result = {"ran": False, "window": len(self.mono_map), "residuals": 0}
        self.last_geo_ba_result = {"ran": False, "window": len(self.mono_map), "edges": 0}
        self.last_depth_ba_result = {
            "ran": False,
            "window": len(self.mono_map),
            "landmarks": 0,
            "edges": 0,
            "cost_before": 0.0,
            "cost_after": 0.0,
            "updated": 0,
            "median_abs_log_depth_update": 0.0,
            "max_abs_log_depth_update": 0.0,
        }
        self.last_keyframe_pose_update_norm = 0.0
        self.last_keyframe_pose_update_rot_deg = 0.0
        if not self.bootstrap_complete or not used_direct:
            return

        should_insert, reason = self.mono_map.should_insert(
            self.T_W_C,
            direct_inliers,
            direct_landmarks,
            self.min_direct_landmarks,
            force_first=len(self.mono_map) == 0,
        )
        if not should_insert:
            return

        pts_w, intensities, _, _, ids = self.sparse3d.mature_landmarks(
            self.T_W_C,
            image,
            affine_a=self.current_a,
            affine_b=self.current_b,
        )
        inserted = self.mono_map.add_keyframe(
            frame_id,
            image,
            self.T_W_C,
            self.current_a,
            self.current_b,
            ids,
            pts_w,
            intensities,
            reason,
            direct_inliers,
            direct_landmarks,
            observations=observations,
            observation_cov=np.eye(2) * self.sparse3d.settings.sigma_pixel * self.sparse3d.settings.sigma_pixel,
        )
        self.last_keyframe_inserted = inserted
        self.last_keyframe_reason = self.mono_map.last_insert_reason
        if inserted and len(self.mono_map) >= 2:
            T_before_opt = self.T_W_C.copy()
            self.sparse3d.retire(ids)
            self.feature_tracker.remove_ids(ids)
            self.keyframes_since_ba += 1
            should_run_ba = self.keyframes_since_ba >= self.ba_every_keyframes or reason == "low_inlier_ratio"
            if should_run_ba:
                self.last_geo_ba_result = self.ba.optimize_mono_geometric_pose_window(self.mono_map, max_iters=3)
                self.last_depth_ba_result = self.ba.optimize_mono_inverse_depth_window(self.mono_map, max_iters=3)
                self.last_ba_result = self.ba.optimize_mono_pose_window(self.mono_map, max_iters=1)
                self.keyframes_since_ba = 0
                self.T_W_C = self.mono_map.keyframes[-1].T_W_C.copy()
                self.last_keyframe_pose_update_norm = float(np.linalg.norm(self.T_W_C[:3, 3] - T_before_opt[:3, 3]))
                self.last_keyframe_pose_update_rot_deg = float(np.rad2deg(_rotation_angle(T_before_opt[:3, :3].T @ self.T_W_C[:3, :3])))
                self.last_direct_delta = np.linalg.inv(self.last_T_W_C) @ self.T_W_C
        elif inserted:
            self.sparse3d.retire(ids)
            self.feature_tracker.remove_ids(ids)

    def _track_direct_candidates(self, image: np.ndarray, guesses: list[tuple[str, np.ndarray]]) -> dict:
        best = {
            "accepted": False,
            "attempted": False,
            "hypotheses": 0,
            "landmarks": 0,
            "inliers": 0,
            "T_W_C": self.T_W_C.copy(),
            "a": self.current_a,
            "b": self.current_b,
            "source": "hold",
            "candidates": [],
            "score": float("-inf"),
            "_prefilter_ids": np.empty((0,), dtype=int),
            "_reference_ids": np.empty((0,), dtype=int),
            "_reference_kf_ids": np.empty((0,), dtype=int),
        }
        seen = set()
        for source, T_W_C_guess in guesses:
            key = tuple(np.round(T_W_C_guess.reshape(-1), 8))
            if key in seen:
                continue
            seen.add(key)
            best["hypotheses"] += 1
            pts_w, intensities, a_ref, b_ref, ids = self.sparse3d.mature_landmarks(
                T_W_C_guess,
                image,
                affine_a=self.current_a,
                affine_b=self.current_b,
            ) if len(self.mono_map) == 0 else self.mono_map.direct_references(T_W_C_guess)
            reference_ids = self.mono_map.last_direct_reference_landmark_ids.copy() if len(self.mono_map) > 0 else ids.copy()
            reference_kf_ids = self.mono_map.last_direct_reference_kf_ids.copy() if len(self.mono_map) > 0 else np.empty((0,), dtype=int)
            prefilter_total = len(pts_w)
            pts_w, intensities, a_ref, b_ref, ids, prefilter_stats = self._prefilter_direct_references(
                image,
                T_W_C_guess,
                pts_w,
                intensities,
                a_ref,
                b_ref,
                ids,
            )
            self.last_direct_prefilter_kept = int(prefilter_stats["kept"])
            self.last_direct_prefilter_rejected = int(prefilter_stats["rejected"])
            best["landmarks"] = max(best["landmarks"], len(pts_w))
            candidate = {
                "source": source,
                "landmarks": int(len(pts_w)),
                "prefilter_total": int(prefilter_total),
                "prefilter_kept": int(prefilter_stats["kept"]),
                "prefilter_rejected": int(prefilter_stats["rejected"]),
                "prefilter_threshold": float(prefilter_stats["threshold"]),
                "attempted": False,
                "inliers": 0,
                "accepted": False,
                "selected": False,
                "step": float(np.linalg.norm(T_W_C_guess[:3, 3] - self.T_W_C[:3, 3])),
                "rotation_deg": float(np.rad2deg(_rotation_angle(self.T_W_C[:3, :3].T @ T_W_C_guess[:3, :3]))),
                "klt_residual_median_px": 0.0,
                "klt_residual_p90_px": 0.0,
                "klt_flow_cos_median": 0.0,
                "klt_flow_cos_p10": 0.0,
                "reject_reason": "",
            }
            if len(pts_w) < self.min_direct_landmarks:
                candidate["reject_reason"] = "insufficient_landmarks"
                best["candidates"].append(candidate)
                continue

            best["attempted"] = True
            candidate["attempted"] = True
            T_C_W, inliers, a_opt, b_opt = self.direct_tracker.track_map(
                image,
                pts_w,
                intensities,
                a_ref,
                b_ref,
                np.linalg.inv(T_W_C_guess),
                self.current_a,
                self.current_b,
                max_iters=10,
            )
            inlier_count = int(np.sum(inliers))
            candidate["inliers"] = inlier_count

            T_W_C_opt = np.linalg.inv(T_C_W)
            candidate["step"] = float(np.linalg.norm(T_W_C_opt[:3, 3] - self.T_W_C[:3, 3]))
            candidate["rotation_deg"] = float(np.rad2deg(_rotation_angle(self.T_W_C[:3, :3].T @ T_W_C_opt[:3, :3])))
            residual_stats = self._keyframe_klt_residual_stats_for_pose(T_W_C_opt, image.shape)
            flow_stats = self._keyframe_klt_flow_stats_for_pose(T_W_C_opt, image.shape)
            candidate["klt_residual_median_px"] = residual_stats["median"]
            candidate["klt_residual_p90_px"] = residual_stats["p90"]
            candidate["klt_flow_cos_median"] = flow_stats["median"]
            candidate["klt_flow_cos_p10"] = flow_stats["p10"]
            candidate["score"] = self._direct_candidate_score(inlier_count, residual_stats)
            if candidate["score"] <= best["score"]:
                candidate["reject_reason"] = "not_best_score"
                best["candidates"].append(candidate)
                continue
            if not self._direct_step_is_plausible(T_W_C_opt):
                candidate["reject_reason"] = "implausible_step"
                best["candidates"].append(candidate)
                continue
            if (
                flow_stats["count"] >= self.min_keyframe_klt_flow_gate_tracks
                and flow_stats["median"] < self.min_keyframe_klt_flow_cos
            ):
                candidate["reject_reason"] = "opposite_klt_flow"
                best["candidates"].append(candidate)
                continue
            best.update({
                "accepted": inlier_count >= self.min_direct_inliers,
                "inliers": inlier_count,
                "T_W_C": T_W_C_opt,
                "a": a_opt,
                "b": b_opt,
                "source": source,
                "score": candidate["score"],
                "_prefilter_ids": ids.copy(),
                "_reference_ids": reference_ids,
                "_reference_kf_ids": reference_kf_ids,
            })
            candidate["accepted"] = inlier_count >= self.min_direct_inliers
            candidate["selected"] = True
            best["candidates"].append(candidate)
            if (
                best["accepted"]
                and source in {"keyframe_klt", "svo_match"}
                and residual_stats["count"] >= self.candidate_klt_residual_gate_tracks
                and residual_stats["median"] <= self.candidate_klt_median_good_px
                and residual_stats["p90"] <= self.candidate_klt_p90_good_px
            ):
                break
            if best["accepted"] and source in {"direct_motion", "direct_hold"}:
                break
        if len(self.mono_map) > 0 and best["accepted"]:
            self.mono_map.record_direct_reference_prefilter(
                best["_prefilter_ids"],
                best["_reference_ids"],
                best["_reference_kf_ids"],
            )
        best.pop("_prefilter_ids", None)
        best.pop("_reference_ids", None)
        best.pop("_reference_kf_ids", None)
        return best

    def _prefilter_direct_references(
        self,
        image: np.ndarray,
        T_W_C_guess: np.ndarray,
        pts_w: np.ndarray,
        intensities: np.ndarray,
        a_ref: np.ndarray,
        b_ref: np.ndarray,
        ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        if len(pts_w) == 0:
            stats = {"kept": 0, "rejected": 0, "threshold": 0.0}
            return pts_w, intensities, a_ref, b_ref, ids, stats

        residuals, visible = self._direct_reference_photometric_errors(
            image,
            T_W_C_guess,
            pts_w,
            intensities,
            a_ref,
            b_ref,
        )
        if len(residuals) == 0:
            stats = {"kept": 0, "rejected": float(len(pts_w)), "threshold": 0.0}
            empty = np.empty((0,), dtype=int)
            return pts_w[empty], intensities[empty], a_ref[empty], b_ref[empty], ids[empty], stats

        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        robust_sigma = 1.4826 * mad
        robust_threshold = median + self.direct_prefilter_mad_factor * robust_sigma
        threshold = min(self.direct_prefilter_abs_threshold, max(8.0, robust_threshold))
        keep_visible = residuals <= threshold

        min_keep = int(np.ceil(self.direct_prefilter_min_keep_ratio * len(residuals)))
        if np.sum(keep_visible) < min_keep and len(residuals) > 0:
            keep_count = max(min_keep, min(len(residuals), self.min_direct_landmarks))
            keep_order = np.argsort(residuals)[:keep_count]
            keep_visible = np.zeros(len(residuals), dtype=bool)
            keep_visible[keep_order] = True

        keep = np.zeros(len(pts_w), dtype=bool)
        keep[np.flatnonzero(visible)] = keep_visible
        kept = int(np.sum(keep))
        stats = {
            "kept": kept,
            "rejected": int(len(pts_w) - kept),
            "threshold": float(threshold),
        }
        return pts_w[keep], intensities[keep], a_ref[keep], b_ref[keep], ids[keep], stats

    def _direct_reference_photometric_errors(
        self,
        image: np.ndarray,
        T_W_C_guess: np.ndarray,
        pts_w: np.ndarray,
        intensities: np.ndarray,
        a_ref: np.ndarray,
        b_ref: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_f = image.astype(np.float32)
        intensities = ensure_pattern_intensities(intensities)
        uv, _depth, visible = self._project_points(T_W_C_guess, pts_w, image.shape, margin=2)
        if not np.any(visible):
            return np.empty((0,), dtype=np.float32), visible

        uv_visible = uv[visible]
        u_pat = (uv_visible[:, None, 0] + PATTERN_DX).reshape(-1)
        v_pat = (uv_visible[:, None, 1] + PATTERN_DY).reshape(-1)
        cur_intensities = nd.map_coordinates(image_f, [v_pat, u_pat], order=1).reshape(-1, len(PATTERN_DX))

        idx = np.flatnonzero(visible)
        exp_diff = np.exp(self.current_a - a_ref[idx])[:, None]
        bias_diff = (self.current_b - b_ref[idx])[:, None]
        predicted = exp_diff * intensities[idx] + bias_diff
        errors = np.mean(np.abs(cur_intensities - predicted), axis=1)
        return errors.astype(np.float32), visible

    def _direct_candidate_score(self, inlier_count: int, klt_residual_stats: dict[str, float]) -> float:
        score = float(inlier_count)
        if klt_residual_stats["count"] < self.candidate_klt_residual_gate_tracks:
            return score
        median_penalty = max(0.0, klt_residual_stats["median"] - self.candidate_klt_median_good_px)
        p90_penalty = max(0.0, klt_residual_stats["p90"] - self.candidate_klt_p90_good_px)
        return (
            score
            - self.candidate_klt_median_penalty * median_penalty
            - self.candidate_klt_p90_penalty * p90_penalty
        )

    def _direct_step_is_plausible(self, T_W_C: np.ndarray) -> bool:
        step = np.linalg.norm(T_W_C[:3, 3] - self.T_W_C[:3, 3])
        if step > self.max_direct_step:
            return False
        angle = _rotation_angle(self.T_W_C[:3, :3].T @ T_W_C[:3, :3])
        return angle <= self.max_direct_rotation_step_rad

    def _direct_pose_guesses(self) -> list[tuple[str, np.ndarray]]:
        guesses = []
        if self.have_direct_motion_model:
            guesses.append(("direct_motion", self.T_W_C @ self.last_direct_delta))
        guesses.append(("direct_hold", self.T_W_C.copy()))

        for axis in np.eye(3):
            for sign in (-1.0, 1.0):
                delta = np.eye(4)
                delta[:3, :3] = _rotation_matrix(axis * sign * self.recovery_rotation_step_rad)
                guesses.append(("direct_recovery", self.T_W_C @ delta))
        return guesses

    def _reset_essential_diagnostics(self, observations: dict[int, np.ndarray]) -> None:
        self.last_essential_common_tracks = len(set(self.prev_observations) & set(observations))
        self.last_essential_inliers = 0
        self.last_essential_used = False

    def _essential_pose_guess(self, observations: dict[int, np.ndarray]) -> np.ndarray | None:
        common = sorted(set(self.prev_observations) & set(observations))
        self.last_essential_common_tracks = len(common)
        self.last_essential_inliers = 0
        self.last_essential_used = False
        if len(common) < self.min_essential_tracks:
            return None

        pts_prev = np.asarray([self.prev_observations[fid] for fid in common], dtype=np.float64)
        pts_cur = np.asarray([observations[fid] for fid in common], dtype=np.float64)
        E, mask = cv2.findEssentialMat(
            pts_prev,
            pts_cur,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if E is None:
            return None
        if E.shape[0] > 3:
            E = E[:3, :3]

        inliers, R_cur_prev, t_cur_prev, _ = cv2.recoverPose(E, pts_prev, pts_cur, self.K, mask=mask)
        self.last_essential_inliers = int(inliers)
        if inliers < self.min_essential_tracks:
            return None
        if _rotation_angle(R_cur_prev) > self.max_rotation_step_rad:
            return None

        T_cur_prev = np.eye(4)
        T_cur_prev[:3, :3] = R_cur_prev
        direction = t_cur_prev.reshape(3)
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-9:
            return None
        T_cur_prev[:3, 3] = direction / direction_norm * self._bootstrap_step_scale(observations)
        self.last_essential_used = True
        return self.T_W_C @ np.linalg.inv(T_cur_prev)

    def _image_motion_pose_guess(self, observations: dict[int, np.ndarray]) -> np.ndarray | None:
        common = sorted(set(self.prev_observations) & set(observations))
        if len(common) < 8:
            return None
        pts_prev = np.asarray([self.prev_observations[fid] for fid in common], dtype=np.float64)
        pts_cur = np.asarray([observations[fid] for fid in common], dtype=np.float64)
        flow = np.median(pts_cur - pts_prev, axis=0)
        if not np.all(np.isfinite(flow)) or np.linalg.norm(flow) < 0.05:
            return None

        R_cur_prev = self._flow_rotation_guess(pts_prev, pts_cur)
        delta = np.eye(4)
        if R_cur_prev is not None:
            delta[:3, :3] = R_cur_prev.T
        delta[0, 3] = -flow[0] * self.image_motion_fallback_depth / self.K[0, 0]
        delta[1, 3] = -flow[1] * self.image_motion_fallback_depth / self.K[1, 1]
        step = np.linalg.norm(delta[:3, 3])
        if step > self.max_bootstrap_step:
            delta[:3, 3] *= self.max_bootstrap_step / step
        return self.T_W_C @ delta

    def _flow_rotation_guess(self, pts_prev: np.ndarray, pts_cur: np.ndarray) -> np.ndarray | None:
        prev_n = cv2.undistortPoints(pts_prev.reshape(-1, 1, 2), self.K, None).reshape(-1, 2)
        cur_n = cv2.undistortPoints(pts_cur.reshape(-1, 1, 2), self.K, None).reshape(-1, 2)
        d = cur_n - prev_n

        x = prev_n[:, 0]
        y = prev_n[:, 1]
        A = np.zeros((len(prev_n) * 2, 3), dtype=np.float64)
        rhs = d.reshape(-1)
        A[0::2, 0] = x * y
        A[0::2, 1] = -(1.0 + x * x)
        A[0::2, 2] = y
        A[1::2, 0] = 1.0 + y * y
        A[1::2, 1] = -x * y
        A[1::2, 2] = -x

        try:
            omega, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return None
        angle = np.linalg.norm(omega)
        if not np.isfinite(angle) or angle < 1e-5:
            return None
        if angle > self.max_rotation_step_rad:
            omega *= self.max_rotation_step_rad / angle

        R_cur_prev, _ = cv2.Rodrigues(omega.reshape(3, 1))
        return R_cur_prev

    def _bootstrap_step_scale(self, observations: dict[int, np.ndarray]) -> float:
        common = sorted(set(self.prev_observations) & set(observations))
        if len(common) < 8:
            return min(0.05, self.max_bootstrap_step)
        pts_prev = np.asarray([self.prev_observations[fid] for fid in common], dtype=np.float64)
        pts_cur = np.asarray([observations[fid] for fid in common], dtype=np.float64)
        flow = np.median(pts_cur - pts_prev, axis=0)
        if not np.all(np.isfinite(flow)):
            return min(0.05, self.max_bootstrap_step)
        step = np.linalg.norm([
            flow[0] * self.image_motion_fallback_depth / self.K[0, 0],
            flow[1] * self.image_motion_fallback_depth / self.K[1, 1],
        ])
        return float(np.clip(step, 1e-3, self.max_bootstrap_step))


def _rotation_angle(R: np.ndarray) -> float:
    cos_angle = (np.trace(R) - 1.0) * 0.5
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _rotation_matrix(omega: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(omega, dtype=np.float64).reshape(3, 1))
    return R
