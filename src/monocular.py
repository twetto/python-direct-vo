from dataclasses import dataclass

import cv2
import numpy as np
import scipy.ndimage as nd

from src.feature_tracker import FeatureTracker, FeatureTrackerConfig
from src.landmark_filter import LandmarkStatus, Sparse3DFilterBank, Sparse3DSettings, bearing_from_pixel, two_ray_ranges
from src.mono_map import MonoMap, MonoMapConfig
from src.optimizer import PhotometricBA
from src.tracker import DirectTracker, PATTERN_DX, PATTERN_DY, PATTERN_SIZE, ensure_pattern_intensities


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
    mono_svo_match_stats: dict
    mono_keyframe_reproj_rejected: int
    mono_keyframe_reproj_median_px: float
    mono_landmark_updates: int
    mono_landmark_update_median_m: float
    mono_landmark_reproj_before_px: float
    mono_landmark_reproj_after_px: float
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
        self.initial_keyframe_min_landmarks = max(self.min_direct_landmarks * 3, self.mono_map.config.min_keyframe_landmarks)
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
        self._current_observations: dict[int, np.ndarray] = {}
        self.bootstrap_anchor_observations: dict[int, np.ndarray] = {}
        self.bootstrap_anchor_kf_id: int | None = None
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
        self.last_keyframe_reproj_rejected = 0
        self.last_keyframe_reproj_median_px = 0.0
        self.last_landmark_updates = 0
        self.last_landmark_update_median_m = 0.0
        self.last_landmark_reproj_before_px = 0.0
        self.last_landmark_reproj_after_px = 0.0
        self.min_keyframe_klt_flow_cos = 0.2
        self.min_keyframe_klt_flow_gate_tracks = 50
        self.candidate_klt_residual_gate_tracks = 50
        self.candidate_klt_median_good_px = 1.0
        self.candidate_klt_p90_good_px = 3.0
        self.candidate_klt_median_reject_px = 5.0
        self.candidate_klt_p90_reject_px = 12.0
        self.candidate_klt_median_penalty = 5.0
        self.candidate_klt_p90_penalty = 1.0
        self.direct_prefilter_abs_threshold = 45.0
        self.direct_prefilter_mad_factor = 3.0
        self.direct_prefilter_min_keep_ratio = 0.35
        self.last_direct_prefilter_kept = 0
        self.last_direct_prefilter_rejected = 0
        self.last_frame_observations: dict[int, MonoFrameObservation] = {}
        self.last_svo_match_stats = self._empty_svo_match_stats("not_run")
        self.patch_match_search_radius = 2
        self.patch_match_max_error = 35.0
        self.matched_pose_min_inlier_ratio = 0.55
        self.matched_pose_min_inliers = max(24, self.min_direct_inliers)
        self.matched_pose_max_median_residual_px = 1.2
        self.matched_pose_max_p90_residual_px = 3.0
        self.matched_pose_max_median_patch_error = 18.0
        self.matched_pose_max_p90_patch_error = 35.0
        self.matched_pose_min_flow_cos = 0.45
        self.matched_pose_min_flow_gate_tracks = 50
        self.first_keyframe_max_reprojection_error_px = 1.5
        self.max_bootstrap_anchor_rotation_rad = np.deg2rad(45.0)
        # Two-view bootstrap gates (Phase 1: scale fixed once by triangulation).
        # Aligned with SVO Pro's initialization contract (svo/src/initialization.cpp):
        # gate on MEDIAN feature disparity (their init_min_disparity, = 30 px in the
        # EuRoC configs), then require an init_min_inliers-style triangulated floor.
        # At marginal parallax the correspondences simply do not triangulate, so
        # both SVO and DSO refuse to initialize there rather than seeking more points.
        self.bootstrap_min_flow_px = 30.0
        self.bootstrap_min_parallax_deg = 2.0
        self.bootstrap_min_triangulated = max(int(self.min_essential_tracks), 40)
        # Stage 1 (DSO-style pure-Gaussian depth): promote a filter landmark into the
        # VO map once it has real anchor->current parallax, using the filter's own
        # (birth-triangulated, recursively refined) depth -- one estimator, not two.
        # KLT outlier rejection is already handled upstream by FeatureTracker's RANSAC.
        self.map_promotion_min_parallax_deg = 5.0
        # Scale-invariant keyframing (SVO needNewKf, frame_handler_base.cpp). Metric
        # motion is unusable in an under-scaled mono frame; pixel disparity vs the last
        # keyframe and tracked-landmark count are scale-invariant. EuRoC values from
        # svo exp_euroc_nolc.yaml (kfselect_min_disparity 40, lower_thresh 100).
        self.kf_min_disparity_px = 40.0
        self.kf_min_tracked_landmarks = 60
        self.kf_min_frames_between = 2
        # Pose = PnP-RANSAC from the reliable KLT map-point correspondences (an
        # absolute, drift-free anchor), then direct photometric refinement seeded from
        # it. Replaces the drifting motion-model / candidate-selection soup.
        self.pnp_min_correspondences = 12
        self.pnp_reproj_thresh = 3.0
        self.pnp_min_inliers = 12
        self.landmark_update_max_reprojection_px = 8.0
        self.landmark_update_min_parallax_sin = 0.003
        self.landmark_update_max_abs_log_range = 0.06

    def process(self, frame_id: int, image: np.ndarray) -> MonocularVOResult:
        self.last_keyframe_inserted = False
        self.last_keyframe_reason = ""
        self.last_keyframe_reproj_rejected = 0
        self.last_keyframe_reproj_median_px = 0.0
        self.last_landmark_updates = 0
        self.last_landmark_update_median_m = 0.0
        self.last_landmark_reproj_before_px = 0.0
        self.last_landmark_reproj_after_px = 0.0
        exclusion_points = self._project_existing_map_landmarks(image.shape)
        self.last_gftt_exclusions = len(exclusion_points)
        observations = self.feature_tracker.update(image, exclusion_points=exclusion_points)
        self._current_observations = observations
        seeded_sparse_depth = self._ensure_first_keyframe(frame_id, image, observations)

        image_motion_fallback_used = False
        motion_source = "hold"

        # --- Bootstrap: a single two-view triangulation that fixes scale once ---
        # (Phase 1) The map is initialized in one step from anchor<->current
        # essential + triangulation, with median scene depth normalized to 1.0.
        # No assumed-depth / flow-scaled magnitude is ever injected.
        bootstrapped = False
        if not self.bootstrap_complete:
            self._reset_essential_diagnostics(observations)
            if not seeded_sparse_depth:
                bootstrapped = self._try_two_view_bootstrap(frame_id, image, observations)
            if bootstrapped:
                motion_source = "two_view_bootstrap"

        used_direct = False
        direct_inliers = 0
        direct_landmarks = 0
        direct_attempted = False
        direct_hypotheses = 0
        visible_refs = []
        klt_pixel_guesses = {}
        best = {"candidates": []}
        if self.bootstrap_complete and not bootstrapped:
            self._reset_essential_diagnostics(observations)
            self.last_svo_match_stats = self._empty_svo_match_stats("klt_pnp")
            old_T_W_C = self.T_W_C.copy()
            visible_refs = self.mono_map.visible_references(self.T_W_C, image.shape)
            klt_pixel_guesses = self._keyframe_klt_pixel_guesses(image, visible_refs)
            # 1) Absolute pose from the reliable KLT 2D-3D correspondences (drift-free).
            T_C_W_pnp, pnp_inliers = self._pnp_from_klt(visible_refs, klt_pixel_guesses)
            if T_C_W_pnp is not None:
                direct_attempted = True
                # 2) Direct photometric refinement seeded from the PnP pose.
                pts_w, ints, a_ref, b_ref, ids = self.mono_map.direct_references(np.linalg.inv(T_C_W_pnp))
                direct_landmarks = int(len(pts_w))
                if len(pts_w) >= self.min_direct_landmarks:
                    T_C_W_out, inl, a_opt, b_opt = self.direct_tracker.track_map(
                        image, pts_w, ints, a_ref, b_ref, T_C_W_pnp,
                        self.current_a, self.current_b, max_iters=10,
                    )
                    self.T_W_C = np.linalg.inv(T_C_W_out)
                    self.current_a = a_opt
                    self.current_b = b_opt
                    direct_inliers = int(np.sum(inl))
                else:
                    self.T_W_C = np.linalg.inv(T_C_W_pnp)  # too few map refs; trust PnP
                    direct_inliers = int(pnp_inliers)
                self.last_direct_delta = np.linalg.inv(old_T_W_C) @ self.T_W_C
                self.have_direct_motion_model = True
                self.last_keyframe_klt_used = True
                used_direct = True
                motion_source = "klt_pnp_direct"
            else:
                self.last_keyframe_klt_used = False
                self.T_W_C = old_T_W_C
                motion_source = "hold"
        elif not bootstrapped:
            self.last_svo_match_stats = self._empty_svo_match_stats("bootstrap")

        # Post-bootstrap: keep Sparse3D refining depths for future keyframes.
        # (KLT tracks are already fundamental-RANSAC filtered inside FeatureTracker.)
        if used_direct and self.bootstrap_complete:
            self.sparse3d.update(frame_id, observations, self.T_W_C, remove_missing=False)
        if used_direct and len(self.mono_map) > 0 and visible_refs and klt_pixel_guesses:
            self._update_map_landmarks_from_klt(image.shape, visible_refs, klt_pixel_guesses, self.T_W_C)
        if not self.last_keyframe_inserted:
            self._maybe_add_keyframe(frame_id, image, observations, used_direct, direct_inliers, direct_landmarks, klt_pixel_guesses)
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
            mono_svo_match_stats=dict(self.last_svo_match_stats),
            mono_keyframe_reproj_rejected=int(self.last_keyframe_reproj_rejected),
            mono_keyframe_reproj_median_px=float(self.last_keyframe_reproj_median_px),
            mono_landmark_updates=int(self.last_landmark_updates),
            mono_landmark_update_median_m=float(self.last_landmark_update_median_m),
            mono_landmark_reproj_before_px=float(self.last_landmark_reproj_before_px),
            mono_landmark_reproj_after_px=float(self.last_landmark_reproj_after_px),
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

    def _ensure_first_keyframe(
        self,
        frame_id: int,
        image: np.ndarray,
        observations: dict[int, np.ndarray],
    ) -> bool:
        if len(self.mono_map) > 0:
            return False
        inserted = self.mono_map.add_empty_keyframe(
            frame_id,
            image,
            self.T_W_C,
            self.current_a,
            self.current_b,
            reason="first",
        )
        if not inserted:
            return False
        self.bootstrap_anchor_kf_id = int(self.mono_map.keyframes[0].kf_id)
        self.bootstrap_anchor_observations = {
            int(fid): np.asarray(pixel, dtype=np.float64).copy()
            for fid, pixel in observations.items()
        }
        self.sparse3d.update(frame_id, observations, self.T_W_C, remove_missing=True)
        self.last_keyframe_inserted = True
        self.last_keyframe_reason = self.mono_map.last_insert_reason
        return True

    def _keyframe_klt_pose_guess(self, image: np.ndarray) -> np.ndarray | None:
        refs = self.mono_map.visible_references(self.T_W_C, image.shape)
        pixel_guesses = self._keyframe_klt_pixel_guesses(image, refs)
        return self._matched_observation_pose_guess(image, self.T_W_C, refs, pixel_guesses)

    def _keyframe_klt_pixel_guesses(self, image: np.ndarray, refs: list[dict]) -> dict[int, np.ndarray]:
        """Associate visible map points with their persistent KLT observation.

        Map points keep their FeatureTracker track after promotion (not retired), so
        each one's current pixel is read directly from the tracker's observations --
        a real frame-to-frame track, not a per-frame re-projection through the pose.
        """
        self.last_keyframe_klt_tracks = 0
        self.last_keyframe_klt_inliers = 0
        self.last_keyframe_klt_points_w = np.empty((0, 3), dtype=np.float64)
        self.last_keyframe_klt_prev_pixels = np.empty((0, 2), dtype=np.float64)
        self.last_keyframe_klt_pixels = np.empty((0, 2), dtype=np.float64)
        if not refs:
            return {}

        obs = self._current_observations or {}
        prev = self.prev_observations or {}
        h, w = image.shape
        guesses = {}
        points_w, prev_px, cur_px, ids = [], [], [], []
        for ref in refs:
            fid = int(ref["landmark_id"])
            cur = obs.get(fid)
            if cur is None:
                continue  # map point is not currently tracked by the KLT front-end
            cur = np.asarray(cur, dtype=np.float64)
            if cur[0] < 2 or cur[0] >= w - 2 or cur[1] < 2 or cur[1] >= h - 2:
                continue
            guesses[fid] = cur.copy()
            points_w.append(ref["point_w"])
            cur_px.append(cur)
            prev_px.append(np.asarray(prev.get(fid, cur), dtype=np.float64))
            ids.append(fid)

        self.last_keyframe_klt_tracks = len(ids)
        if ids:
            self.last_keyframe_klt_points_w = np.asarray(points_w, dtype=np.float64)
            self.last_keyframe_klt_prev_pixels = np.asarray(prev_px, dtype=np.float64)
            self.last_keyframe_klt_pixels = np.asarray(cur_px, dtype=np.float64)
        return guesses

    def _matched_observation_pose_guess(
        self,
        image: np.ndarray,
        T_W_C_guess: np.ndarray,
        refs: list[dict],
        pixel_guesses: dict[int, np.ndarray],
    ) -> np.ndarray | None:
        matched = self._match_visible_reference_patches(image, refs, pixel_guesses)
        accepted = [obs for obs in matched.values() if obs.accepted]
        self.last_svo_match_stats = {
            "reason": "not_evaluated",
            "visible_refs": int(len(refs)),
            "klt_pixel_guesses": int(len(pixel_guesses)),
            "patch_matches": int(len(accepted)),
            "pnp_inliers": 0,
            "pnp_inlier_ratio": 0.0,
            "patch_median_error": float(np.median([obs.photometric_error for obs in accepted])) if accepted else 0.0,
            "patch_p90_error": float(np.percentile([obs.photometric_error for obs in accepted], 90)) if accepted else 0.0,
            "residual_median_px": 0.0,
            "residual_p90_px": 0.0,
            "flow_cos_median": 0.0,
        }
        self.last_frame_observations = matched
        self.last_keyframe_klt_inliers = len(accepted)
        if len(accepted) < self.matched_pose_min_inliers:
            self.last_svo_match_stats["reason"] = "too_few_patch_matches"
            return None

        points_w = np.asarray([obs.point_w for obs in accepted], dtype=np.float64)
        pixels = np.asarray([obs.matched_pixel for obs in accepted], dtype=np.float64)
        patch_errors = np.asarray([obs.photometric_error for obs in accepted], dtype=np.float64)

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
            self.last_svo_match_stats["reason"] = "pnp_failed"
            return None

        self.last_keyframe_klt_inliers = int(len(inliers))
        self.last_svo_match_stats["pnp_inliers"] = int(len(inliers))
        self.last_svo_match_stats["pnp_inlier_ratio"] = float(len(inliers)) / float(len(accepted))
        if len(inliers) < self.matched_pose_min_inliers:
            self.last_svo_match_stats["reason"] = "too_few_pnp_inliers"
            return None
        if float(len(inliers)) / float(len(accepted)) < self.matched_pose_min_inlier_ratio:
            self.last_svo_match_stats["reason"] = "low_pnp_inlier_ratio"
            return None
        inlier_idx = inliers.reshape(-1)
        self.last_keyframe_klt_points_w = points_w[inlier_idx].astype(np.float64)
        self.last_keyframe_klt_prev_pixels = np.asarray([obs.initial_pixel for obs in accepted], dtype=np.float64)[inlier_idx]
        self.last_keyframe_klt_pixels = pixels[inlier_idx].astype(np.float64)
        inlier_patch_errors = patch_errors[inlier_idx]
        if (
            float(np.median(inlier_patch_errors)) > self.matched_pose_max_median_patch_error
            or float(np.percentile(inlier_patch_errors, 90)) > self.matched_pose_max_p90_patch_error
        ):
            self.last_svo_match_stats["reason"] = "high_patch_error"
            return None

        R_C_W, _ = cv2.Rodrigues(rvec)
        T_C_W = np.eye(4, dtype=np.float64)
        T_C_W[:3, :3] = R_C_W
        T_C_W[:3, 3] = tvec.reshape(3)
        T_W_C = np.linalg.inv(T_C_W)
        if not self._direct_step_is_plausible(T_W_C):
            self.last_svo_match_stats["reason"] = "implausible_step"
            return None
        residual_stats = self._keyframe_klt_residual_stats_for_pose(T_W_C, image.shape)
        self.last_svo_match_stats["residual_median_px"] = float(residual_stats["median"])
        self.last_svo_match_stats["residual_p90_px"] = float(residual_stats["p90"])
        if (
            residual_stats["count"] >= self.candidate_klt_residual_gate_tracks
            and (
                residual_stats["median"] > self.matched_pose_max_median_residual_px
                or residual_stats["p90"] > self.matched_pose_max_p90_residual_px
            )
        ):
            self.last_svo_match_stats["reason"] = "high_reprojection_residual"
            return None
        flow_stats = self._keyframe_klt_flow_stats_for_pose(T_W_C, image.shape)
        self.last_svo_match_stats["flow_cos_median"] = float(flow_stats["median"])
        if (
            flow_stats["count"] >= self.matched_pose_min_flow_gate_tracks
            and flow_stats["median"] < self.matched_pose_min_flow_cos
        ):
            self.last_svo_match_stats["reason"] = "inconsistent_flow_direction"
            return None
        self.last_svo_match_stats["reason"] = "accepted"
        return T_W_C

    def _empty_svo_match_stats(self, reason: str) -> dict:
        return {
            "reason": reason,
            "visible_refs": 0,
            "klt_pixel_guesses": 0,
            "patch_matches": 0,
            "pnp_inliers": 0,
            "pnp_inlier_ratio": 0.0,
            "patch_median_error": 0.0,
            "patch_p90_error": 0.0,
            "residual_median_px": 0.0,
            "residual_p90_px": 0.0,
            "flow_cos_median": 0.0,
        }

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
        klt_pixel_guesses: dict[int, np.ndarray] | None = None,
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

        should_insert, reason = self._needs_keyframe(frame_id, klt_pixel_guesses, direct_landmarks)
        if not should_insert:
            return

        # One estimator: use the filter's own (birth-triangulated, refined) depth, and
        # gate promotion on geometric parallax rather than EKF convergence -- the
        # convergence gate lagged camera motion and froze the map (plan Phase 3B).
        pts_w, intensities, _, _, ids = self.sparse3d.mature_landmarks(
            self.T_W_C,
            image,
            affine_a=self.current_a,
            affine_b=self.current_b,
            require_converged=False,
        )
        keep = self._map_promotion_parallax_mask(pts_w, ids)
        pts_w, intensities, ids = pts_w[keep], intensities[keep], ids[keep]
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
        if inserted:
            # Carry the filter's uncertainty into the map BEFORE retiring the filters.
            self._store_promotion_covariances(ids)
            # Record covisibility: existing map points tracked into this frame gain a
            # new observation here, giving BA real multi-view depth constraints.
            if klt_pixel_guesses:
                self.mono_map.add_covisibility_observations(
                    self.mono_map.keyframes[-1].kf_id,
                    klt_pixel_guesses,
                    covariance=np.eye(2) * self.sparse3d.settings.sigma_pixel * self.sparse3d.settings.sigma_pixel,
                )
        if inserted and len(self.mono_map) >= 2:
            T_before_opt = self.T_W_C.copy()
            # Retire the depth filter (BA owns depth now) but KEEP the KLT track: the
            # map point must stay tracked by the front-end for covisibility/tracking.
            self.sparse3d.retire(ids)
            # One joint local BA (poses + covisible points, covariance priors),
            # every keyframe -- replaces the three alternating pose/depth-only BAs.
            self.last_ba_result = self.ba.optimize_mono_joint_window(self.mono_map, max_nfev=40)
            if self.last_ba_result.get("ran"):
                self.T_W_C = self.mono_map.keyframes[-1].T_W_C.copy()
                self.last_keyframe_pose_update_norm = float(np.linalg.norm(self.T_W_C[:3, 3] - T_before_opt[:3, 3]))
                self.last_keyframe_pose_update_rot_deg = float(np.rad2deg(_rotation_angle(T_before_opt[:3, :3].T @ self.T_W_C[:3, :3])))
                self.last_direct_delta = np.linalg.inv(self.last_T_W_C) @ self.T_W_C
        elif inserted:
            self.sparse3d.retire(ids)

    def _maybe_add_initial_keyframe(
        self,
        frame_id: int,
        image: np.ndarray,
        observations: dict[int, np.ndarray],
    ) -> None:
        if self.bootstrap_complete or len(self.mono_map) == 0:
            return
        host_kf = self.mono_map.keyframes[0]
        initial_landmark_target = int(self.initial_keyframe_min_landmarks)
        if len(host_kf.landmark_ids) >= initial_landmark_target:
            self.bootstrap_complete = True
            return
        self.last_keyframe_inserted = False
        self.last_keyframe_reason = ""
        pts_w, intensities, ids, host_observations = self._initial_keyframe_landmarks(host_kf, observations)
        if len(ids) == 0:
            return
        added = self.mono_map.add_landmarks_to_keyframe(
            host_kf.kf_id,
            ids,
            pts_w,
            intensities,
            observations=host_observations,
            observation_cov=np.eye(2) * self.sparse3d.settings.sigma_pixel * self.sparse3d.settings.sigma_pixel,
        )
        self.last_keyframe_inserted = added > 0
        self.last_keyframe_reason = "bootstrap_landmarks" if added > 0 else ""
        if added > 0 and len(host_kf.landmark_ids) >= initial_landmark_target:
            self.bootstrap_complete = True
            self.sparse3d.retire(ids)
            self.feature_tracker.remove_ids(ids)
            self.have_direct_motion_model = False
            self.last_direct_delta = np.eye(4)

    def _initial_keyframe_landmarks(
        self,
        host_kf,
        observations: dict[int, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
        points_w = []
        intensities = []
        ids = []
        host_observations = {}
        residuals = []
        rejected = 0
        image_f = host_kf.image.astype(np.float32)
        h, w = host_kf.image.shape
        T_C_W = np.linalg.inv(self.T_W_C)
        max_error = float(self.first_keyframe_max_reprojection_error_px)
        for fid, lm in self.sparse3d.features.items():
            fid_i = int(fid)
            if int(lm.anchor_frame_id) != int(host_kf.frame_id):
                rejected += 1
                continue
            observed = observations.get(fid_i)
            if observed is None:
                rejected += 1
                continue
            try:
                point_w = lm.anchor_T_W_C[:3, :3] @ lm.anchor_point() + lm.anchor_T_W_C[:3, 3]
            except ValueError:
                rejected += 1
                continue
            point_c = T_C_W[:3, :3] @ point_w + T_C_W[:3, 3]
            if point_c[2] <= 0.01:
                rejected += 1
                continue
            projected = np.array([
                self.K[0, 0] * point_c[0] / point_c[2] + self.K[0, 2],
                self.K[1, 1] * point_c[1] / point_c[2] + self.K[1, 2],
            ])
            observed = np.asarray(observed, dtype=np.float64)
            error = float(np.linalg.norm(projected - observed))
            if not np.isfinite(error) or error > max_error:
                rejected += 1
                continue
            host_pixel = self._pixel_from_bearing(lm.anchor_bearing)
            if host_pixel[0] < 2 or host_pixel[0] >= w - 2 or host_pixel[1] < 2 or host_pixel[1] >= h - 2:
                rejected += 1
                continue
            patch = nd.map_coordinates(
                image_f,
                [host_pixel[1] + PATTERN_DY, host_pixel[0] + PATTERN_DX],
                order=1,
            ).astype(np.float32)
            points_w.append(point_w)
            intensities.append(patch)
            ids.append(fid_i)
            host_observations[fid_i] = host_pixel
            residuals.append(error)

        self.last_keyframe_reproj_rejected = int(rejected)
        self.last_keyframe_reproj_median_px = float(np.median(residuals)) if residuals else 0.0
        if not ids:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, PATTERN_SIZE), dtype=np.float32),
                np.empty((0,), dtype=int),
                {},
            )
        return (
            np.asarray(points_w, dtype=np.float64),
            np.asarray(intensities, dtype=np.float32),
            np.asarray(ids, dtype=int),
            host_observations,
        )

    def _update_map_landmarks_from_klt(
        self,
        image_shape: tuple[int, int],
        refs: list[dict],
        pixel_guesses: dict[int, np.ndarray],
        T_W_C_observe: np.ndarray | None = None,
    ) -> None:
        if not refs or not pixel_guesses or len(self.mono_map) == 0:
            return

        T_W_C_obs = self.T_W_C if T_W_C_observe is None else np.asarray(T_W_C_observe, dtype=np.float64)
        keyframes_by_id = {int(kf.kf_id): kf for kf in self.mono_map.keyframes}
        before_errors = []
        after_errors = []
        update_norms = []
        h, w = image_shape
        max_reproj = float(self.landmark_update_max_reprojection_px)
        max_log = float(self.landmark_update_max_abs_log_range)
        for ref in refs:
            fid = int(ref["landmark_id"])
            observed = pixel_guesses.get(fid)
            if observed is None:
                continue
            observed = np.asarray(observed, dtype=np.float64)
            if observed[0] < 2 or observed[0] >= w - 2 or observed[1] < 2 or observed[1] >= h - 2:
                continue

            matched_obs = self.last_frame_observations.get(fid)
            if matched_obs is not None and not matched_obs.accepted:
                continue

            host_kf = keyframes_by_id.get(int(ref["host_kf_id"]))
            if host_kf is None:
                continue
            host_pixel = np.asarray(ref["host_pixel"], dtype=np.float64)
            old_point = np.asarray(ref["point_w"], dtype=np.float64)
            projected, _depths, visible = self._project_points(T_W_C_obs, old_point.reshape(1, 3), image_shape)
            if not bool(visible[0]):
                continue
            before_error = float(np.linalg.norm(projected[0] - observed))
            if not np.isfinite(before_error) or before_error > max_reproj:
                continue

            b_host = bearing_from_pixel(host_pixel, self.K)
            b_current = bearing_from_pixel(observed, self.K)
            T_H_C = np.linalg.inv(host_kf.T_W_C) @ T_W_C_obs
            R_H_C = T_H_C[:3, :3]
            t_H_C = T_H_C[:3, 3]
            if np.linalg.norm(np.cross(b_host, R_H_C @ b_current)) < self.landmark_update_min_parallax_sin:
                continue
            ranges = two_ray_ranges(b_host, b_current, R_H_C, t_H_C)
            if ranges is None:
                continue
            host_range, current_range = ranges
            if (
                not np.isfinite(host_range)
                or not np.isfinite(current_range)
                or host_range < self.sparse3d.settings.min_depth
                or current_range < self.sparse3d.settings.min_depth
                or host_range > self.sparse3d.settings.max_depth
                or current_range > self.sparse3d.settings.max_depth
            ):
                continue

            old_host = np.linalg.inv(host_kf.T_W_C)[:3, :3] @ old_point + np.linalg.inv(host_kf.T_W_C)[:3, 3]
            old_range = float(np.linalg.norm(old_host))
            if old_range <= 1e-9:
                continue
            log_update = float(np.clip(np.log(host_range / old_range), -max_log, max_log))
            refined_range = old_range * np.exp(log_update)
            refined_host = b_host * refined_range
            refined_point = host_kf.T_W_C[:3, :3] @ refined_host + host_kf.T_W_C[:3, 3]

            refined_projected, _depths, refined_visible = self._project_points(
                T_W_C_obs,
                refined_point.reshape(1, 3),
                image_shape,
            )
            if not bool(refined_visible[0]):
                continue
            after_error = float(np.linalg.norm(refined_projected[0] - observed))
            if not np.isfinite(after_error) or after_error > before_error:
                continue
            if self.mono_map.update_landmark_point(fid, refined_point):
                before_errors.append(before_error)
                after_errors.append(after_error)
                update_norms.append(float(np.linalg.norm(refined_point - old_point)))

        self.last_landmark_updates = int(len(update_norms))
        self.last_landmark_update_median_m = float(np.median(update_norms)) if update_norms else 0.0
        self.last_landmark_reproj_before_px = float(np.median(before_errors)) if before_errors else 0.0
        self.last_landmark_reproj_after_px = float(np.median(after_errors)) if after_errors else 0.0

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
                require_converged=self.bootstrap_complete,
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
            required_direct_landmarks = self.min_direct_landmarks
            if len(self.mono_map) > 0:
                required_direct_landmarks = min(self.min_direct_landmarks, self.mono_map.config.min_keyframe_landmarks)
            if len(pts_w) < required_direct_landmarks:
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
            if (
                residual_stats["count"] >= self.candidate_klt_residual_gate_tracks
                and (
                    residual_stats["median"] > self.candidate_klt_median_reject_px
                    or residual_stats["p90"] > self.candidate_klt_p90_reject_px
                )
            ):
                candidate["reject_reason"] = "high_klt_reprojection_residual"
                best["candidates"].append(candidate)
                continue
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

    def _sample_patches(self, image: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bilinearly sample the tracker photometric pattern at each pixel.

        Returns (patches, keep) where keep is False for pixels too close to the
        image border to hold a full pattern.
        """
        image_f = image.astype(np.float32)
        h, w = image.shape
        patches = []
        keep = []
        for px in np.asarray(pixels, dtype=np.float64):
            u, v = float(px[0]), float(px[1])
            if u < 2 or u >= w - 2 or v < 2 or v >= h - 2:
                patches.append(np.zeros(PATTERN_SIZE, dtype=np.float32))
                keep.append(False)
                continue
            patch = nd.map_coordinates(image_f, [v + PATTERN_DY, u + PATTERN_DX], order=1).astype(np.float32)
            patches.append(patch)
            keep.append(True)
        return (
            np.asarray(patches, dtype=np.float32).reshape(-1, PATTERN_SIZE),
            np.asarray(keep, dtype=bool),
        )

    def _filter_point_covariance(self, lm) -> np.ndarray | None:
        """World-frame 3x3 point covariance of a Sparse3D landmark from its EKF state."""
        try:
            _point, jac = lm.anchor_point_and_jacobian()
        except (ValueError, np.linalg.LinAlgError):
            return None
        p3_anchor = jac @ lm.covariance[:3, :3] @ jac.T
        R_w_a = lm.anchor_T_W_C[:3, :3]
        p3_world = R_w_a @ p3_anchor @ R_w_a.T
        if not np.all(np.isfinite(p3_world)):
            return None
        return 0.5 * (p3_world + p3_world.T)

    def _needs_keyframe(
        self,
        frame_id: int,
        klt_pixel_guesses: dict[int, np.ndarray] | None,
        direct_landmarks: int,
    ) -> tuple[bool, str]:
        """Scale-invariant keyframe decision (SVO needNewKf).

        Metric motion is unusable when the estimate is under-scaled; instead use
        median pixel disparity of tracked map points vs the last keyframe, and the
        tracked-landmark count. Both are scale-invariant.
        """
        if len(self.mono_map) == 0:
            return True, "first"
        latest = self.mono_map.keyframes[-1]
        if int(frame_id) - int(latest.frame_id) < self.kf_min_frames_between:
            return False, ""
        tracked = klt_pixel_guesses or {}
        if len(tracked) < self.kf_min_tracked_landmarks:
            return True, "low_tracked"
        disparities = []
        for fid, pixel in tracked.items():
            ref = latest.observations.get(int(fid)) or latest.features.get(int(fid))
            if ref is not None:
                disparities.append(float(np.linalg.norm(np.asarray(pixel, dtype=np.float64) - np.asarray(ref.pixel, dtype=np.float64))))
        if disparities and float(np.median(disparities)) > self.kf_min_disparity_px:
            return True, "disparity"
        return False, ""

    def _store_promotion_covariances(self, ids: np.ndarray) -> None:
        """Transfer each promoted filter's 3-D covariance onto its map landmark (a prior)."""
        for fid in ids:
            lm = self.sparse3d.features.get(int(fid))
            if lm is None:
                continue
            cov = self._filter_point_covariance(lm)
            if cov is not None:
                self.mono_map.set_landmark_covariance(int(fid), cov)

    def _pnp_from_klt(
        self,
        refs: list[dict],
        klt_pixel_guesses: dict[int, np.ndarray],
    ) -> tuple[np.ndarray | None, int]:
        """Absolute pose (T_C_W) via PnP-RANSAC from map-point 3D <-> KLT 2D matches.

        KLT is pose-drift-free (pure 2D optical flow), so this is a per-frame absolute
        anchor -- unlike a motion-model seed, it does not inherit prior drift.
        """
        if not refs or not klt_pixel_guesses:
            return None, 0
        ref_by_id = {int(r["landmark_id"]): r for r in refs}
        points_w, pixels = [], []
        for fid, pixel in klt_pixel_guesses.items():
            ref = ref_by_id.get(int(fid))
            if ref is None:
                continue
            points_w.append(ref["point_w"])
            pixels.append(pixel)
        if len(points_w) < self.pnp_min_correspondences:
            return None, 0
        points_w = np.asarray(points_w, dtype=np.float64)
        pixels = np.asarray(pixels, dtype=np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_w, pixels, self.K, None,
            reprojectionError=self.pnp_reproj_thresh, confidence=0.99,
            iterationsCount=100, flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < self.pnp_min_inliers:
            return None, 0
        R, _ = cv2.Rodrigues(rvec)
        T_C_W = np.eye(4, dtype=np.float64)
        T_C_W[:3, :3] = R
        T_C_W[:3, 3] = tvec.reshape(3)
        return T_C_W, int(len(inliers))

    def _map_promotion_parallax_mask(self, points_w: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Keep only landmarks with real anchor->current parallax for map promotion.

        Promotion quality is geometric (parallax), not the filter's radial-blind
        depth variance (see plan Phase 3B).
        """
        keep = np.zeros(len(ids), dtype=bool)
        if len(ids) == 0:
            return keep
        cur_center = self.T_W_C[:3, 3]
        min_cos = np.cos(np.deg2rad(self.map_promotion_min_parallax_deg))
        for i, fid in enumerate(ids):
            lm = self.sparse3d.features.get(int(fid))
            if lm is None:
                continue
            anchor_center = lm.anchor_T_W_C[:3, 3]
            v0 = points_w[i] - anchor_center
            v1 = points_w[i] - cur_center
            n0 = float(np.linalg.norm(v0))
            n1 = float(np.linalg.norm(v1))
            if n0 < 1e-9 or n1 < 1e-9:
                continue
            cos = float(np.dot(v0, v1) / (n0 * n1))
            if cos <= min_cos:  # angle >= threshold
                keep[i] = True
        return keep

    def _try_two_view_bootstrap(
        self,
        frame_id: int,
        image: np.ndarray,
        observations: dict[int, np.ndarray],
    ) -> bool:
        """Initialize the map from anchor(KF0)<->current via essential+triangulation.

        Scale is fixed exactly once here by normalizing the median triangulated
        depth to 1.0. Returns True only when a full initial map was seeded.
        """
        if len(self.mono_map) == 0 or not self.bootstrap_anchor_observations:
            return False
        anchor_kf = self.mono_map.keyframes[0]
        common = sorted(set(self.bootstrap_anchor_observations) & set(observations))
        self.last_essential_common_tracks = len(common)
        self.last_essential_inliers = 0
        self.last_essential_used = False
        if len(common) < self.min_essential_tracks:
            return False

        pts_anchor = np.asarray([self.bootstrap_anchor_observations[fid] for fid in common], dtype=np.float64)
        pts_cur = np.asarray([observations[fid] for fid in common], dtype=np.float64)
        # Wait for real parallax before triangulating (avoids degenerate scale).
        if float(np.median(np.linalg.norm(pts_cur - pts_anchor, axis=1))) < self.bootstrap_min_flow_px:
            return False

        E, mask = cv2.findEssentialMat(
            pts_anchor, pts_cur, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0,
        )
        if E is None:
            return False
        if E.shape[0] > 3:
            E = E[:3, :3]

        n_inliers, R_cur_anchor, t_cur_anchor, pose_mask = cv2.recoverPose(
            E, pts_anchor, pts_cur, self.K, mask=mask,
        )
        self.last_essential_inliers = int(n_inliers)
        if n_inliers < self.min_essential_tracks:
            return False
        if _rotation_angle(R_cur_anchor) > self.max_bootstrap_anchor_rotation_rad:
            return False

        t_dir = t_cur_anchor.reshape(3)
        t_norm = float(np.linalg.norm(t_dir))
        if t_norm < 1e-9:
            return False
        t_dir = t_dir / t_norm

        inlier_mask = pose_mask.reshape(-1).astype(bool)
        if int(np.sum(inlier_mask)) < self.min_essential_tracks:
            return False
        ids = np.asarray(common, dtype=int)[inlier_mask]
        p_anchor = pts_anchor[inlier_mask]
        p_cur = pts_cur[inlier_mask]

        # Triangulate with anchor at [I|0] and current at [R|t] (unit baseline).
        P0 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = self.K @ np.hstack([R_cur_anchor, t_dir.reshape(3, 1)])
        X4 = cv2.triangulatePoints(P0, P1, p_anchor.T, p_cur.T)
        w = X4[3]
        finite = np.abs(w) > 1e-9
        pts_a = np.full((X4.shape[1], 3), np.nan, dtype=np.float64)
        pts_a[finite] = (X4[:3, finite] / w[finite]).T

        z_anchor = pts_a[:, 2]
        pts_c = pts_a @ R_cur_anchor.T + t_dir
        z_cur = pts_c[:, 2]
        cheirality = finite & np.isfinite(z_anchor) & (z_anchor > 0) & (z_cur > 0)
        if int(np.sum(cheirality)) < self.bootstrap_min_triangulated:
            return False

        # Median triangulation angle (parallax) between the two viewing rays.
        cam_cur_center = -R_cur_anchor.T @ t_dir
        ray0 = pts_a[cheirality]
        ray1 = pts_a[cheirality] - cam_cur_center
        cos_par = np.sum(ray0 * ray1, axis=1) / (
            np.linalg.norm(ray0, axis=1) * np.linalg.norm(ray1, axis=1) + 1e-12
        )
        parallax_deg = np.degrees(np.arccos(np.clip(cos_par, -1.0, 1.0)))
        if float(np.median(parallax_deg)) < self.bootstrap_min_parallax_deg:
            return False

        ids = ids[cheirality]
        p_anchor = p_anchor[cheirality]
        X = pts_a[cheirality]

        # Fix scale exactly once: normalize median anchor-frame depth to 1.0.
        median_depth = float(np.median(X[:, 2]))
        if not np.isfinite(median_depth) or median_depth <= 1e-6:
            return False
        scale = 1.0 / median_depth
        X = X * scale
        t_scaled = t_dir * scale

        # Anchor pose is the gauge; recover the current pose from the scaled baseline.
        T_cur_anchor = np.eye(4)
        T_cur_anchor[:3, :3] = R_cur_anchor
        T_cur_anchor[:3, 3] = t_scaled
        T_W_cur = anchor_kf.T_W_C @ np.linalg.inv(T_cur_anchor)
        points_w = (anchor_kf.T_W_C[:3, :3] @ X.T).T + anchor_kf.T_W_C[:3, 3]

        intensities, patch_ok = self._sample_patches(anchor_kf.image, p_anchor)
        if int(np.sum(patch_ok)) < self.bootstrap_min_triangulated:
            return False
        ids = ids[patch_ok]
        points_w = points_w[patch_ok]
        intensities = intensities[patch_ok]
        p_anchor = p_anchor[patch_ok]

        host_observations = {int(fid): p_anchor[i].copy() for i, fid in enumerate(ids)}
        sigma = self.sparse3d.settings.sigma_pixel
        added = self.mono_map.add_landmarks_to_keyframe(
            anchor_kf.kf_id,
            ids,
            points_w,
            intensities,
            observations=host_observations,
            observation_cov=np.eye(2) * sigma * sigma,
        )
        if added <= 0:
            return False

        self.T_W_C = T_W_cur
        self.current_a = anchor_kf.affine_a
        self.current_b = anchor_kf.affine_b
        self.last_essential_used = True
        self.bootstrap_complete = True
        self.have_direct_motion_model = False
        self.last_direct_delta = np.eye(4)
        # Retire the depth filters but KEEP the KLT tracks so the initial map points
        # stay tracked by the front-end (covisibility/tracking).
        self.sparse3d.retire(ids)
        self.last_keyframe_inserted = True
        self.last_keyframe_reason = "two_view_bootstrap"
        return True

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

    def _pixel_from_bearing(self, bearing: np.ndarray) -> np.ndarray:
        b = np.asarray(bearing, dtype=np.float64)
        if abs(float(b[2])) < 1e-12:
            return np.array([np.nan, np.nan], dtype=np.float64)
        return np.array([
            self.K[0, 0] * b[0] / b[2] + self.K[0, 2],
            self.K[1, 1] * b[1] / b[2] + self.K[1, 2],
        ], dtype=np.float64)

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

    def _point_flow_step_scale(self, pts_prev: np.ndarray, pts_cur: np.ndarray) -> float:
        if len(pts_prev) < 8 or len(pts_cur) < 8:
            return min(0.05, self.max_bootstrap_step)
        flow = np.median(np.asarray(pts_cur, dtype=np.float64) - np.asarray(pts_prev, dtype=np.float64), axis=0)
        if not np.all(np.isfinite(flow)):
            return min(0.05, self.max_bootstrap_step)
        step = np.linalg.norm([
            flow[0] * self.image_motion_fallback_depth / self.K[0, 0],
            flow[1] * self.image_motion_fallback_depth / self.K[1, 1],
        ])
        return float(np.clip(step, 1e-3, self.max_bootstrap_step))

    def _bootstrap_step_scale(self, observations: dict[int, np.ndarray]) -> float:
        common = sorted(set(self.prev_observations) & set(observations))
        if len(common) < 8:
            return min(0.05, self.max_bootstrap_step)
        pts_prev = np.asarray([self.prev_observations[fid] for fid in common], dtype=np.float64)
        pts_cur = np.asarray([observations[fid] for fid in common], dtype=np.float64)
        return self._point_flow_step_scale(pts_prev, pts_cur)


def _rotation_angle(R: np.ndarray) -> float:
    cos_angle = (np.trace(R) - 1.0) * 0.5
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _rotation_matrix(omega: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(omega, dtype=np.float64).reshape(3, 1))
    return R
