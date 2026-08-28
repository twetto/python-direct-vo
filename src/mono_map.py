from dataclasses import dataclass, field

import numpy as np

from src.landmark_filter import bearing_from_pixel
from src.tracker import PATTERN_SIZE


@dataclass
class MonoObservation:
    landmark_id: int
    pixel: np.ndarray
    point_w: np.ndarray
    covariance: np.ndarray


@dataclass
class MonoFeature:
    landmark_id: int
    pixel: np.ndarray
    bearing: np.ndarray
    level: int = 0
    kind: str = "corner"
    intensity: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    point_w: np.ndarray | None = None
    covariance: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=np.float64))
    host_kf_id: int = -1


@dataclass
class MonoLandmarkTrack:
    landmark_id: int
    point_w: np.ndarray
    host_kf_id: int
    observations: dict[int, MonoFeature] = field(default_factory=dict)
    n_succeeded_reproj: int = 0
    n_failed_reproj: int = 0
    last_projected_kf_id: int = -1


@dataclass
class MonoKeyframe:
    kf_id: int
    frame_id: int
    image: np.ndarray
    T_W_C: np.ndarray
    affine_a: float
    affine_b: float
    landmark_ids: np.ndarray
    points_w: np.ndarray
    intensities: np.ndarray
    insertion_reason: str
    direct_inliers: int = 0
    direct_landmarks: int = 0
    observations: dict[int, MonoObservation] = field(default_factory=dict)
    features: dict[int, MonoFeature] = field(default_factory=dict)


@dataclass
class MonoMapConfig:
    window_size: int = 5
    protected_latest_keyframes: int = 2
    max_reference_keyframes: int = 3
    min_covisible_landmarks: int = 20
    min_insert_inlier_ratio: float = 0.45
    min_motion_from_latest: float = 0.12
    min_direct_candidate_factor: float = 1.5
    min_keyframe_landmarks: int = 30
    max_direct_refs: int = 800
    min_useful_reference_count: int = 30
    min_useful_reference_ratio: float = 0.2
    max_low_usefulness_frames: int = 5


@dataclass
class MonoMap:
    K: np.ndarray
    config: MonoMapConfig = field(default_factory=MonoMapConfig)
    keyframes: list[MonoKeyframe] = field(default_factory=list)
    next_kf_id: int = 0
    last_insert_reason: str = ""
    last_inserted: bool = False
    last_reference_counts: dict[int, int] = field(default_factory=dict)
    last_visible_counts: dict[int, int] = field(default_factory=dict)
    last_prefilter_kept_counts: dict[int, int] = field(default_factory=dict)
    last_prefilter_rejected_counts: dict[int, int] = field(default_factory=dict)
    keyframe_low_usefulness_frames: dict[int, int] = field(default_factory=dict)
    keyframe_usefulness_ratio: dict[int, float] = field(default_factory=dict)
    last_direct_reference_landmark_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    last_direct_reference_kf_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    last_discarded_kf_id: int = -1
    last_discard_reason: str = ""
    landmarks: dict[int, MonoLandmarkTrack] = field(default_factory=dict)

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.keyframes)

    def latest(self) -> MonoKeyframe | None:
        return self.keyframes[-1] if self.keyframes else None

    def should_insert(
        self,
        T_W_C: np.ndarray,
        direct_inliers: int,
        direct_landmarks: int,
        min_direct_landmarks: int,
        force_first: bool = False,
    ) -> tuple[bool, str]:
        if force_first or not self.keyframes:
            return True, "first"
        if direct_landmarks < self.config.min_keyframe_landmarks:
            return False, "insufficient_landmarks"

        inlier_ratio = direct_inliers / max(direct_landmarks, 1)
        if inlier_ratio < self.config.min_insert_inlier_ratio:
            return True, "low_inlier_ratio"

        latest = self.keyframes[-1]
        motion = np.linalg.norm(T_W_C[:3, 3] - latest.T_W_C[:3, 3])
        if motion > self.config.min_motion_from_latest:
            return True, "motion"

        if direct_landmarks < int(min_direct_landmarks * self.config.min_direct_candidate_factor):
            return True, "low_coverage"
        return False, ""

    def add_keyframe(
        self,
        frame_id: int,
        image: np.ndarray,
        T_W_C: np.ndarray,
        affine_a: float,
        affine_b: float,
        landmark_ids: np.ndarray,
        points_w: np.ndarray,
        intensities: np.ndarray,
        reason: str,
        direct_inliers: int,
        direct_landmarks: int,
        observations: dict[int, np.ndarray] | None = None,
        observation_cov: np.ndarray | None = None,
    ) -> bool:
        if len(points_w) < self.config.min_keyframe_landmarks:
            self.last_inserted = False
            self.last_insert_reason = "insufficient_landmarks"
            return False

        landmark_ids_arr = np.asarray(landmark_ids, dtype=int).copy()
        points_w_arr = np.asarray(points_w, dtype=np.float64).copy()
        intensities_arr = np.asarray(intensities, dtype=np.float32).copy()
        if len(landmark_ids_arr) != len(points_w_arr) or len(landmark_ids_arr) != len(intensities_arr):
            raise ValueError("landmark ids, points, and intensities must have matching lengths")

        kf_id = self.next_kf_id
        T_W_C_arr = np.asarray(T_W_C, dtype=np.float64).copy()
        cov = np.eye(2, dtype=np.float64) if observation_cov is None else np.asarray(observation_cov, dtype=np.float64)
        features, obs_graph = self._make_keyframe_features(
            kf_id,
            image.shape,
            T_W_C_arr,
            landmark_ids_arr,
            points_w_arr,
            intensities_arr,
            observations,
            cov,
        )
        kf = MonoKeyframe(
            kf_id=kf_id,
            frame_id=frame_id,
            image=image.copy(),
            T_W_C=T_W_C_arr,
            affine_a=float(affine_a),
            affine_b=float(affine_b),
            landmark_ids=landmark_ids_arr,
            points_w=points_w_arr,
            intensities=intensities_arr,
            insertion_reason=reason,
            direct_inliers=int(direct_inliers),
            direct_landmarks=int(direct_landmarks),
            observations=obs_graph,
            features=features,
        )
        self.keyframes.append(kf)
        self._add_landmark_observations(kf)
        self.next_kf_id += 1
        if len(self.keyframes) > self.config.window_size:
            self._discard_redundant_keyframe()
        self.last_inserted = True
        self.last_insert_reason = reason
        return True

    def _discard_redundant_keyframe(self) -> None:
        self.last_discarded_kf_id = -1
        self.last_discard_reason = ""
        protected = max(1, int(self.config.protected_latest_keyframes))
        candidates = self.keyframes[:-protected]
        if not candidates:
            dropped = self.keyframes.pop(0)
            self._remove_keyframe_observations(dropped.kf_id)
            self.last_discarded_kf_id = int(dropped.kf_id)
            self.last_discard_reason = "oldest"
            return

        latest_kf = self.keyframes[-1]
        best_drop_idx = 0
        max_redundancy_score = -float("inf")
        for i, c_kf in enumerate(candidates):
            if self.keyframe_low_usefulness_frames.get(c_kf.kf_id, 0) >= self.config.max_low_usefulness_frames:
                best_drop_idx = i
                max_redundancy_score = float("inf")
                self.last_discard_reason = "low_usefulness"
                break

            reference_count = self.last_reference_counts.get(c_kf.kf_id, 0)
            visible_count = self.last_visible_counts.get(c_kf.kf_id, 0)
            if reference_count < self.config.min_covisible_landmarks and visible_count < self.config.min_covisible_landmarks:
                best_drop_idx = i
                max_redundancy_score = float("inf")
                self.last_discard_reason = "low_covisibility"
                break

            redundancy = 0.0
            for other_kf in self.keyframes:
                if other_kf is c_kf:
                    continue
                dist = np.linalg.norm(c_kf.T_W_C[:3, 3] - other_kf.T_W_C[:3, 3])
                redundancy += 1.0 / (dist + 1e-5)

            dist_to_latest = np.linalg.norm(c_kf.T_W_C[:3, 3] - latest_kf.T_W_C[:3, 3])
            contribution = max(reference_count, visible_count)
            score = redundancy * np.sqrt(dist_to_latest)
            score /= np.sqrt(contribution + 1.0)
            if score > max_redundancy_score:
                max_redundancy_score = score
                best_drop_idx = i

        dropped_kf = self.keyframes[best_drop_idx]
        self._transfer_visible_landmarks(dropped_kf, latest_kf)
        self.keyframes.pop(best_drop_idx)
        self._remove_keyframe_observations(dropped_kf.kf_id)
        self.last_discarded_kf_id = int(dropped_kf.kf_id)
        if not self.last_discard_reason:
            self.last_discard_reason = "redundant"
        self.keyframe_low_usefulness_frames.pop(dropped_kf.kf_id, None)
        self.keyframe_usefulness_ratio.pop(dropped_kf.kf_id, None)
        self.last_prefilter_kept_counts.pop(dropped_kf.kf_id, None)
        self.last_prefilter_rejected_counts.pop(dropped_kf.kf_id, None)

    def _transfer_visible_landmarks(self, dropped_kf: MonoKeyframe, latest_kf: MonoKeyframe) -> None:
        if len(dropped_kf.points_w) == 0:
            return

        existing_latest_ids = {int(fid) for fid in latest_kf.landmark_ids}
        active_ids = set()
        for kf in self.keyframes:
            if kf is dropped_kf:
                continue
            active_ids.update(int(fid) for fid in kf.landmark_ids)

        uv, _depths, visible = _project_points(dropped_kf.points_w, latest_kf.T_W_C, self.K, latest_kf.image.shape)
        transferable = []
        for idx, fid in enumerate(dropped_kf.landmark_ids):
            fid_i = int(fid)
            if visible[idx] and fid_i not in existing_latest_ids and fid_i not in active_ids:
                transferable.append(idx)

        if not transferable:
            return

        idxs = np.asarray(transferable, dtype=int)
        ids = dropped_kf.landmark_ids[idxs].astype(int)
        pixels = uv[idxs].astype(np.float64)

        exp_diff = np.exp(latest_kf.affine_a - dropped_kf.affine_a)
        bias_diff = latest_kf.affine_b - dropped_kf.affine_b
        transferred_intensities = exp_diff * dropped_kf.intensities[idxs] + bias_diff

        latest_kf.landmark_ids = np.concatenate([latest_kf.landmark_ids, ids])
        latest_kf.points_w = np.vstack([latest_kf.points_w, dropped_kf.points_w[idxs]])
        latest_kf.intensities = np.vstack([latest_kf.intensities, transferred_intensities.astype(np.float32)])

        for point_w, fid, pixel in zip(dropped_kf.points_w[idxs], ids, pixels):
            dropped_obs = dropped_kf.observations.get(int(fid))
            covariance = np.eye(2, dtype=np.float64) if dropped_obs is None else dropped_obs.covariance
            intensity = latest_kf.intensities[latest_kf.landmark_ids.tolist().index(int(fid))]
            latest_kf.observations[int(fid)] = MonoObservation(
                landmark_id=int(fid),
                pixel=np.asarray(pixel, dtype=np.float64).copy(),
                point_w=np.asarray(point_w, dtype=np.float64).copy(),
                covariance=np.asarray(covariance, dtype=np.float64).copy(),
            )
            feature = MonoFeature(
                landmark_id=int(fid),
                pixel=np.asarray(pixel, dtype=np.float64).copy(),
                bearing=bearing_from_pixel(pixel, self.K),
                intensity=np.asarray(intensity, dtype=np.float32).copy(),
                point_w=np.asarray(point_w, dtype=np.float64).copy(),
                covariance=np.asarray(covariance, dtype=np.float64).copy(),
                host_kf_id=int(dropped_kf.kf_id),
            )
            latest_kf.features[int(fid)] = feature
            track = self.landmarks.get(int(fid))
            if track is None:
                track = MonoLandmarkTrack(
                    landmark_id=int(fid),
                    point_w=np.asarray(point_w, dtype=np.float64).copy(),
                    host_kf_id=int(dropped_kf.kf_id),
                )
                self.landmarks[int(fid)] = track
            track.point_w = np.asarray(point_w, dtype=np.float64).copy()
            track.observations[int(latest_kf.kf_id)] = feature

    def _make_keyframe_features(
        self,
        kf_id: int,
        image_shape: tuple[int, int],
        T_W_C: np.ndarray,
        landmark_ids: np.ndarray,
        points_w: np.ndarray,
        intensities: np.ndarray,
        observations: dict[int, np.ndarray] | None,
        covariance: np.ndarray,
    ) -> tuple[dict[int, MonoFeature], dict[int, MonoObservation]]:
        features = {}
        obs_graph = {}
        observation_pixels = {} if observations is None else {
            int(fid): np.asarray(pixel, dtype=np.float64)
            for fid, pixel in observations.items()
        }
        for idx, fid in enumerate(landmark_ids):
            fid_i = int(fid)
            point_w = np.asarray(points_w[idx], dtype=np.float64)
            pixel = observation_pixels.get(fid_i)
            if pixel is None:
                pixel = _project(point_w, T_W_C, self.K, image_shape)
            if pixel is None:
                continue
            pixel = np.asarray(pixel, dtype=np.float64).copy()
            cov = np.asarray(covariance, dtype=np.float64).copy()
            if fid_i in observation_pixels:
                obs_graph[fid_i] = MonoObservation(
                    landmark_id=fid_i,
                    pixel=pixel.copy(),
                    point_w=point_w.copy(),
                    covariance=cov.copy(),
                )
            features[fid_i] = MonoFeature(
                landmark_id=fid_i,
                pixel=pixel.copy(),
                bearing=bearing_from_pixel(pixel, self.K),
                intensity=np.asarray(intensities[idx], dtype=np.float32).copy(),
                point_w=point_w.copy(),
                covariance=cov,
                host_kf_id=int(kf_id),
            )
        return features, obs_graph

    def _add_landmark_observations(self, kf: MonoKeyframe) -> None:
        points_by_id = {int(fid): np.asarray(kf.points_w[idx], dtype=np.float64) for idx, fid in enumerate(kf.landmark_ids)}
        for fid, feature in kf.features.items():
            point_w = np.asarray(feature.point_w if feature.point_w is not None else points_by_id[fid], dtype=np.float64)
            track = self.landmarks.get(fid)
            if track is None:
                track = MonoLandmarkTrack(
                    landmark_id=fid,
                    point_w=point_w.copy(),
                    host_kf_id=int(kf.kf_id),
                )
                self.landmarks[fid] = track
            track.point_w = point_w.copy()
            track.observations[int(kf.kf_id)] = feature

    def _remove_keyframe_observations(self, kf_id: int) -> None:
        for fid in list(self.landmarks):
            track = self.landmarks[fid]
            track.observations.pop(int(kf_id), None)
            if not track.observations:
                del self.landmarks[fid]

    def sync_landmark_tracks_from_keyframes(self) -> None:
        latest_points = {}
        for kf in self.keyframes:
            for idx, fid in enumerate(kf.landmark_ids):
                latest_points[int(fid)] = np.asarray(kf.points_w[idx], dtype=np.float64).copy()

        for fid, point_w in latest_points.items():
            track = self.landmarks.get(fid)
            if track is None:
                track = MonoLandmarkTrack(
                    landmark_id=fid,
                    point_w=point_w.copy(),
                    host_kf_id=-1,
                )
                self.landmarks[fid] = track
            track.point_w = point_w.copy()
            for feature in track.observations.values():
                feature.point_w = point_w.copy()

        active_ids = set(latest_points)
        for fid in list(self.landmarks):
            if fid not in active_ids:
                del self.landmarks[fid]

    def direct_references(self, T_W_C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.last_reference_counts = {}
        self.last_visible_counts = {}
        self.last_direct_reference_landmark_ids = np.empty((0,), dtype=int)
        self.last_direct_reference_kf_ids = np.empty((0,), dtype=int)
        if not self.keyframes:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, PATTERN_SIZE), dtype=np.float32),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=int),
            )

        selected_keyframes = self._select_reference_keyframes(T_W_C)
        rows = []
        seen = set()
        per_kf_budget = max(1, int(np.ceil(self.config.max_direct_refs / max(len(selected_keyframes), 1))))
        for kf, visible_refs, _distance in selected_keyframes:
            selected_in_kf = 0
            step = max(1, len(visible_refs) // per_kf_budget)
            for ref in visible_refs[::step]:
                fid_i = int(ref["landmark_id"])
                if fid_i in seen:
                    continue
                seen.add(fid_i)
                rows.append((
                    ref["point_w"],
                    ref["intensity"],
                    kf.affine_a,
                    kf.affine_b,
                    fid_i,
                    kf.kf_id,
                ))
                selected_in_kf += 1
                if len(rows) >= self.config.max_direct_refs:
                    break
                if selected_in_kf >= per_kf_budget:
                    break
            self.last_reference_counts[kf.kf_id] = selected_in_kf
            if len(rows) >= self.config.max_direct_refs:
                break

        if not rows:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, PATTERN_SIZE), dtype=np.float32),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=int),
            )
        points, intensities, a_vals, b_vals, ids, kf_ids = zip(*rows)
        self.last_direct_reference_landmark_ids = np.asarray(ids, dtype=int)
        self.last_direct_reference_kf_ids = np.asarray(kf_ids, dtype=int)
        return (
            np.asarray(points, dtype=np.float64),
            np.asarray(intensities, dtype=np.float32),
            np.asarray(a_vals, dtype=np.float64),
            np.asarray(b_vals, dtype=np.float64),
            np.asarray(ids, dtype=int),
        )

    def visible_references(self, T_W_C: np.ndarray, image_shape: tuple[int, int]) -> list[dict]:
        self.last_reference_counts = {}
        self.last_visible_counts = {}
        self.last_direct_reference_landmark_ids = np.empty((0,), dtype=int)
        self.last_direct_reference_kf_ids = np.empty((0,), dtype=int)
        if not self.keyframes:
            return []

        selected_keyframes = self._select_reference_keyframes(T_W_C, image_shape)
        refs = []
        seen = set()
        per_kf_budget = max(1, int(np.ceil(self.config.max_direct_refs / max(len(selected_keyframes), 1))))
        for kf, visible_refs, _distance in selected_keyframes:
            selected_in_kf = 0
            step = max(1, len(visible_refs) // per_kf_budget)
            points_w = np.asarray([ref["point_w"] for ref in visible_refs[::step]], dtype=np.float64)
            uv, _depths, visible = _project_points(points_w, T_W_C, self.K, image_shape) if len(points_w) else (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=bool),
            )
            for ref, projected, is_visible in zip(visible_refs[::step], uv, visible):
                if not is_visible:
                    continue
                fid_i = int(ref["landmark_id"])
                if fid_i in seen:
                    continue
                seen.add(fid_i)
                feature = kf.features.get(fid_i)
                refs.append({
                    "landmark_id": fid_i,
                    "host_kf_id": int(kf.kf_id),
                    "point_w": np.asarray(ref["point_w"], dtype=np.float64).copy(),
                    "host_pixel": np.asarray(feature.pixel if feature is not None else projected, dtype=np.float64).copy(),
                    "projected_pixel": np.asarray(projected, dtype=np.float64).copy(),
                    "intensity": np.asarray(ref["intensity"], dtype=np.float32).copy(),
                    "affine_a": float(kf.affine_a),
                    "affine_b": float(kf.affine_b),
                    "covariance": np.asarray(
                        feature.covariance if feature is not None else np.eye(2, dtype=np.float64),
                        dtype=np.float64,
                    ).copy(),
                })
                selected_in_kf += 1
                if len(refs) >= self.config.max_direct_refs or selected_in_kf >= per_kf_budget:
                    break
            self.last_reference_counts[kf.kf_id] = selected_in_kf
            if len(refs) >= self.config.max_direct_refs:
                break

        if refs:
            self.last_direct_reference_landmark_ids = np.asarray([ref["landmark_id"] for ref in refs], dtype=int)
            self.last_direct_reference_kf_ids = np.asarray([ref["host_kf_id"] for ref in refs], dtype=int)
        return refs

    def record_direct_reference_prefilter(
        self,
        kept_landmark_ids: np.ndarray,
        reference_landmark_ids: np.ndarray | None = None,
        reference_kf_ids: np.ndarray | None = None,
    ) -> None:
        self.last_prefilter_kept_counts = {}
        self.last_prefilter_rejected_counts = {}
        ref_landmark_ids = self.last_direct_reference_landmark_ids if reference_landmark_ids is None else np.asarray(reference_landmark_ids, dtype=int)
        ref_kf_ids = self.last_direct_reference_kf_ids if reference_kf_ids is None else np.asarray(reference_kf_ids, dtype=int)
        if len(ref_kf_ids) == 0:
            return

        kept_set = {int(fid) for fid in kept_landmark_ids}
        active_kf_ids = {kf.kf_id for kf in self.keyframes}
        for kf_id in active_kf_ids:
            self.last_prefilter_kept_counts[kf_id] = 0
            self.last_prefilter_rejected_counts[kf_id] = 0

        # last_direct_reference_kf_ids is aligned with the ids returned by direct_references.
        # The selected refs are unique by landmark id, so id membership is enough here.
        for kf_id, fid in zip(ref_kf_ids, ref_landmark_ids):
            if int(fid) in kept_set:
                self.last_prefilter_kept_counts[int(kf_id)] = self.last_prefilter_kept_counts.get(int(kf_id), 0) + 1
            else:
                self.last_prefilter_rejected_counts[int(kf_id)] = self.last_prefilter_rejected_counts.get(int(kf_id), 0) + 1

        for kf in self.keyframes:
            kf_id = kf.kf_id
            kept = self.last_prefilter_kept_counts.get(kf_id, 0)
            rejected = self.last_prefilter_rejected_counts.get(kf_id, 0)
            total = kept + rejected
            if total == 0:
                continue
            ratio = kept / max(total, 1)
            old_ratio = self.keyframe_usefulness_ratio.get(kf_id, ratio)
            self.keyframe_usefulness_ratio[kf_id] = 0.8 * old_ratio + 0.2 * ratio
            if total >= self.config.min_useful_reference_count and ratio < self.config.min_useful_reference_ratio:
                self.keyframe_low_usefulness_frames[kf_id] = self.keyframe_low_usefulness_frames.get(kf_id, 0) + 1
            else:
                self.keyframe_low_usefulness_frames[kf_id] = max(0, self.keyframe_low_usefulness_frames.get(kf_id, 0) - 1)

    def _select_reference_keyframes(
        self,
        T_W_C: np.ndarray,
        image_shape: tuple[int, int] | None = None,
    ) -> list[tuple[MonoKeyframe, list[dict], float]]:
        selected = []
        current_t = T_W_C[:3, 3]
        for kf in self.keyframes:
            refs = self.keyframe_reference_records(kf)
            if not refs:
                self.last_visible_counts[kf.kf_id] = 0
                continue
            points_w = np.asarray([ref["point_w"] for ref in refs], dtype=np.float64)
            shape = kf.image.shape if image_shape is None else image_shape
            _uv, _depths, visible = _project_points(points_w, T_W_C, self.K, shape)
            visible_refs = [ref for ref, is_visible in zip(refs, visible) if is_visible]
            self.last_visible_counts[kf.kf_id] = int(len(visible_refs))
            if len(visible_refs) == 0:
                continue
            distance = float(np.linalg.norm(kf.T_W_C[:3, 3] - current_t))
            selected.append((kf, visible_refs, distance))

        selected.sort(key=lambda item: (-len(item[1]), item[2], -item[0].frame_id))
        return selected[:max(1, int(self.config.max_reference_keyframes))]

    def keyframe_reference_records(self, kf: MonoKeyframe) -> list[dict]:
        refs = []
        for idx, fid in enumerate(kf.landmark_ids):
            fid_i = int(fid)
            feature = kf.features.get(fid_i)
            track = self.landmarks.get(fid_i)
            point_w = None
            intensity = None
            if track is not None:
                point_w = track.point_w
            if feature is not None:
                if point_w is None:
                    point_w = feature.point_w
                if len(feature.intensity) > 0:
                    intensity = feature.intensity
            if point_w is None:
                point_w = kf.points_w[idx]
            if intensity is None:
                intensity = kf.intensities[idx]
            refs.append({
                "landmark_id": fid_i,
                "point_idx": int(idx),
                "point_w": np.asarray(point_w, dtype=np.float64),
                "intensity": np.asarray(intensity, dtype=np.float32),
            })
        return refs

    def active_landmark_count(self) -> int:
        if self.landmarks:
            return len(self.landmarks)
        ids = set()
        for kf in self.keyframes:
            ids.update(int(fid) for fid in kf.landmark_ids)
        return len(ids)

    def observation_count(self) -> int:
        return sum(len(kf.observations) for kf in self.keyframes)

    def landmark_observation_count(self) -> int:
        return sum(len(track.observations) for track in self.landmarks.values())

    def point_cloud(self, max_points: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows = []
        seen = set()
        for kf in reversed(self.keyframes):
            for ref in self.keyframe_reference_records(kf):
                fid_i = int(ref["landmark_id"])
                if fid_i in seen:
                    continue
                seen.add(fid_i)
                rows.append((
                    ref["point_w"],
                    float(np.mean(ref["intensity"])),
                    fid_i,
                    kf.kf_id,
                ))
                if max_points is not None and len(rows) >= max_points:
                    break
            if max_points is not None and len(rows) >= max_points:
                break

        if not rows:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=int),
            )
        points, intensities, ids, kf_ids = zip(*rows)
        return (
            np.asarray(points, dtype=np.float64),
            np.asarray(intensities, dtype=np.float32),
            np.asarray(ids, dtype=int),
            np.asarray(kf_ids, dtype=int),
        )


def _is_visible(point_w: np.ndarray, T_W_C: np.ndarray, K: np.ndarray, image_shape: tuple[int, int]) -> bool:
    return _project(point_w, T_W_C, K, image_shape) is not None


def _project(point_w: np.ndarray, T_W_C: np.ndarray, K: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray | None:
    uv, _depths, visible = _project_points(np.asarray([point_w], dtype=np.float64), T_W_C, K, image_shape)
    if visible[0]:
        return uv[0]
    return None


def _project_points(points_w: np.ndarray, T_W_C: np.ndarray, K: np.ndarray, image_shape: tuple[int, int]):
    if len(points_w) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=bool)
    T_C_W = np.linalg.inv(T_W_C)
    points_c = points_w @ T_C_W[:3, :3].T + T_C_W[:3, 3]
    z = points_c[:, 2]
    z_safe = np.clip(z, 0.01, None)
    u = K[0, 0] * points_c[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * points_c[:, 1] / z_safe + K[1, 2]
    h, w = image_shape
    visible = (z > 0.01) & (u >= 1) & (u < w - 2) & (v >= 1) & (v < h - 2)
    return np.stack([u, v], axis=1), z, visible
