from dataclasses import dataclass, field
from enum import Enum

import numpy as np


MIN_PARALLAX_SIN = 0.02


class LandmarkStatus(str, Enum):
    IMMATURE = "immature"
    MATURE = "mature"
    REJECTED = "rejected"
    MARGINALIZED = "marginalized"


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 0:
        raise ValueError("cannot normalize zero vector")
    return v / n


def tangent_basis(bearing: np.ndarray) -> np.ndarray:
    b = _normalize(np.asarray(bearing, dtype=np.float64))
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(b @ seed)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])

    u1 = _normalize(np.cross(seed, b))
    u2 = np.cross(b, u1)
    return np.column_stack([u1, u2])


def bearing_from_pixel(pixel: np.ndarray, K: np.ndarray) -> np.ndarray:
    u, v = np.asarray(pixel, dtype=np.float64)
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    return _normalize(np.array([x, y, 1.0]))


def project(point_c: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = point_c[2]
    if z <= 0:
        raise ValueError("point is behind the camera")
    return np.array([
        K[0, 0] * point_c[0] / z + K[0, 2],
        K[1, 1] * point_c[1] / z + K[1, 2],
    ])


def projection_jacobian(point_c: np.ndarray, K: np.ndarray) -> np.ndarray:
    x, y, z = point_c
    if z <= 0:
        raise ValueError("point is behind the camera")
    return np.array([
        [K[0, 0] / z, 0.0, -K[0, 0] * x / (z * z)],
        [0.0, K[1, 1] / z, -K[1, 1] * y / (z * z)],
    ])


def point_and_jacobian(b0: np.ndarray, U: np.ndarray, eta: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    v = b0 + U @ eta
    n = np.linalg.norm(v)
    if n <= 0 or rho <= 0:
        raise ValueError("invalid bearing chart state")
    b = v / n
    point = b / rho
    db_deta = (np.eye(3) - np.outer(b, b)) @ U / n
    jac = np.zeros((3, 3), dtype=np.float64)
    jac[:, :2] = db_deta / rho
    jac[:, 2] = -b / (rho * rho)
    return point, jac


def two_ray_ranges(
    b_anchor: np.ndarray,
    b_other: np.ndarray,
    R_anchor_other: np.ndarray,
    t_anchor_other: np.ndarray,
) -> tuple[float, float] | None:
    b = _normalize(b_anchor)
    a = R_anchor_other @ _normalize(b_other)
    bb = float(b @ b)
    ba = float(b @ a)
    aa = float(a @ a)
    det = bb * aa - ba * ba
    if abs(det) < 1e-12:
        return None
    bt = float(b @ t_anchor_other)
    at = float(a @ t_anchor_other)
    range_anchor = (aa * bt - ba * at) / det
    range_other = (ba * bt - bb * at) / det
    return range_anchor, range_other


@dataclass
class LandmarkObservation:
    frame_id: int
    pixel: np.ndarray
    covariance: np.ndarray


@dataclass
class LandmarkFilter:
    landmark_id: int
    anchor_frame_id: int
    anchor_T_W_C: np.ndarray
    anchor_bearing: np.ndarray
    state: np.ndarray
    covariance: np.ndarray
    status: LandmarkStatus = LandmarkStatus.IMMATURE
    consecutive_failures: int = 0
    observations: list[LandmarkObservation] = field(default_factory=list)

    @classmethod
    def from_anchor_pixel(
        cls,
        landmark_id: int,
        anchor_frame_id: int,
        pixel: np.ndarray,
        K: np.ndarray,
        initial_depth: float,
        depth_sigma: float,
        pixel_sigma: float = 1.0,
        anchor_T_W_C: np.ndarray | None = None,
    ) -> "LandmarkFilter":
        if initial_depth <= 0:
            raise ValueError("initial_depth must be positive")
        if depth_sigma <= 0:
            raise ValueError("depth_sigma must be positive")

        b0 = bearing_from_pixel(pixel, K)
        rho = 1.0 / initial_depth
        sigma_rho = depth_sigma / (initial_depth * initial_depth)
        P = np.diag([pixel_sigma / K[0, 0], pixel_sigma / K[1, 1], sigma_rho, pixel_sigma, pixel_sigma]) ** 2
        return cls(
            landmark_id=landmark_id,
            anchor_frame_id=anchor_frame_id,
            anchor_T_W_C=np.eye(4) if anchor_T_W_C is None else anchor_T_W_C.copy(),
            anchor_bearing=b0,
            state=np.array([0.0, 0.0, rho, 0.0, 0.0], dtype=np.float64),
            covariance=P,
        )

    @property
    def U(self) -> np.ndarray:
        return tangent_basis(self.anchor_bearing)

    def anchor_point(self) -> np.ndarray:
        eta = self.state[:2]
        rho = max(float(self.state[2]), 1e-9)
        point, _ = point_and_jacobian(self.anchor_bearing, self.U, eta, rho)
        return point

    def anchor_point_and_jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        return point_and_jacobian(self.anchor_bearing, self.U, self.state[:2], max(float(self.state[2]), 1e-9))

    def predict_pixel(self, T_C_A: np.ndarray, K: np.ndarray) -> np.ndarray:
        point_c = T_C_A[:3, :3] @ self.anchor_point() + T_C_A[:3, 3]
        return project(point_c, K) + self.state[3:5]

    def current_position(self, T_C_A: np.ndarray) -> np.ndarray:
        return T_C_A[:3, :3] @ self.anchor_point() + T_C_A[:3, 3]

    def depth_variance(self, T_C_A: np.ndarray) -> float:
        _, j_anchor = self.anchor_point_and_jacobian()
        j_current = T_C_A[:3, :3] @ j_anchor
        p_xyz = j_current @ self.covariance[:3, :3] @ j_current.T
        return float(max(p_xyz[2, 2], 0.0))

    def range_variance(self, T_C_A: np.ndarray) -> float:
        point_c = self.current_position(T_C_A)
        r = np.linalg.norm(point_c)
        if r <= 1e-9:
            return float("inf")
        _, j_anchor = self.anchor_point_and_jacobian()
        j_current = T_C_A[:3, :3] @ j_anchor
        h = (point_c / r).reshape(1, 3) @ j_current
        return float(max((h @ self.covariance[:3, :3] @ h.T)[0, 0], 0.0))

    def update(
        self,
        frame_id: int,
        pixel: np.ndarray,
        T_C_A: np.ndarray,
        K: np.ndarray,
        measurement_cov: np.ndarray,
        bias_rw_sigma: float = 0.05,
        gate_chi2: float = 9.21,
    ) -> bool:
        self.covariance[3, 3] += bias_rw_sigma * bias_rw_sigma
        self.covariance[4, 4] += bias_rw_sigma * bias_rw_sigma

        try:
            pred = self.predict_pixel(T_C_A, K)
            H = self.measurement_jacobian(T_C_A, K)
        except ValueError:
            self.status = LandmarkStatus.REJECTED
            return False
        y = np.asarray(pixel, dtype=np.float64) - pred
        S = H @ self.covariance @ H.T + measurement_cov
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.status = LandmarkStatus.REJECTED
            return False

        nis = float(y.T @ S_inv @ y)
        if nis > gate_chi2:
            # Pure-Gaussian outlier rejection via the Mahalanobis gate; the old
            # Gauss-Beta inlier mixture is gone (outlier tracks are rejected upstream
            # by fundamental-RANSAC, drifting tracks by this gate over time).
            self.consecutive_failures += 1
            return False

        K_gain = self.covariance @ H.T @ S_inv
        self.state = self.state + K_gain @ y
        I = np.eye(5)
        self.covariance = (I - K_gain @ H) @ self.covariance @ (I - K_gain @ H).T + K_gain @ measurement_cov @ K_gain.T
        self.consecutive_failures = 0
        self.observations.append(LandmarkObservation(frame_id, np.asarray(pixel, dtype=np.float64), measurement_cov.copy()))
        if len(self.observations) >= 3:
            self.status = LandmarkStatus.MATURE
        return True

    def apply_range_process_noise(
        self,
        T_C_A: np.ndarray,
        range_var: float,
    ) -> None:
        if range_var <= 0:
            return
        point_c = self.current_position(T_C_A)
        r = np.linalg.norm(point_c)
        if r <= 1e-9:
            return
        _, j_anchor = self.anchor_point_and_jacobian()
        j_current = T_C_A[:3, :3] @ j_anchor
        try:
            j_inv = np.linalg.inv(j_current)
        except np.linalg.LinAlgError:
            return
        r_hat = point_c / r
        q_current = range_var * np.outer(r_hat, r_hat)
        q_chart = j_inv @ q_current @ j_inv.T
        p_new = self.covariance[:3, :3] + q_chart
        self.covariance[:3, :3] = 0.5 * (p_new + p_new.T)

    def measurement_jacobian(self, T_C_A: np.ndarray, K: np.ndarray) -> np.ndarray:
        p_anchor, j_anchor = self.anchor_point_and_jacobian()
        point_c = T_C_A[:3, :3] @ p_anchor + T_C_A[:3, 3]
        j_proj = projection_jacobian(point_c, K)
        H = np.zeros((2, 5), dtype=np.float64)
        H[:, :3] = j_proj @ T_C_A[:3, :3] @ j_anchor
        H[:, 3:5] = np.eye(2)
        return H

    def _numeric_jacobian(self, T_C_A: np.ndarray, K: np.ndarray) -> np.ndarray:
        H = np.zeros((2, 5), dtype=np.float64)
        base = self.predict_pixel(T_C_A, K)
        eps = np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4])
        x0 = self.state.copy()
        for i in range(5):
            self.state = x0.copy()
            self.state[i] += eps[i]
            H[:, i] = (self.predict_pixel(T_C_A, K) - base) / eps[i]
        self.state = x0
        return H


@dataclass
class Sparse3DSettings:
    sigma_pixel: float = 0.5
    initial_depth_sigma: float = 2.0
    min_depth: float = 0.2
    max_depth: float = 100.0
    birth_min_flow_px: float = 0.5
    min_parallax_sin: float = MIN_PARALLAX_SIN
    min_track_length: int = 3
    conv_inlier_ratio: float = 0.5
    conv_depth_variance: float = 10.0
    max_pool_size: int = 2000
    max_missed_frames: int = 30
    bias_walk_sigma: float = 0.05
    mahalanobis_gate_chi2: float = 9.21
    range_walk_var: float = 0.0


@dataclass
class PendingFeature:
    frame_id: int
    T_W_C: np.ndarray
    pixel: np.ndarray


class Sparse3DFilterBank:
    """Small Python analogue of the Rust Sparse3D bearing-inverse-depth filter."""

    def __init__(self, K: np.ndarray, settings: Sparse3DSettings | None = None):
        self.K = K
        self.settings = settings or Sparse3DSettings()
        self.features: dict[int, LandmarkFilter] = {}
        self.pending: dict[int, PendingFeature] = {}
        self.prev_pixels: dict[int, np.ndarray] = {}
        self.prev_frame_id: int | None = None
        self.prev_T_W_C: np.ndarray | None = None
        self.retired_ids: set[int] = set()
        self.missed_frames: dict[int, int] = {}

    def update(
        self,
        frame_id: int,
        pixels: dict[int, np.ndarray],
        T_W_C: np.ndarray,
        remove_missing: bool = True,
        preserve_previous: bool = False,
    ) -> None:
        curr = {
            int(fid): np.asarray(pixel, dtype=np.float64)
            for fid, pixel in pixels.items()
            if int(fid) not in self.retired_ids
        }

        for fid, lm in list(self.features.items()):
            if fid not in curr:
                if remove_missing:
                    lm.status = LandmarkStatus.MARGINALIZED
                    del self.features[fid]
                    self.missed_frames.pop(fid, None)
                else:
                    missed = self.missed_frames.get(fid, 0) + 1
                    self.missed_frames[fid] = missed
                    if missed > self.settings.max_missed_frames:
                        lm.status = LandmarkStatus.MARGINALIZED
                        del self.features[fid]
                        self.missed_frames.pop(fid, None)
                continue

            self.missed_frames[fid] = 0
            T_C_A = np.linalg.inv(T_W_C) @ lm.anchor_T_W_C
            if self.settings.range_walk_var > 0.0:
                point_c = lm.current_position(T_C_A)
                lm.apply_range_process_noise(T_C_A, self.settings.range_walk_var * float(point_c @ point_c))
            ok = lm.update(
                frame_id,
                curr[fid],
                T_C_A,
                self.K,
                np.eye(2) * self.settings.sigma_pixel * self.settings.sigma_pixel,
                bias_rw_sigma=self.settings.bias_walk_sigma,
                gate_chi2=self.settings.mahalanobis_gate_chi2,
            )
            if not ok and lm.consecutive_failures >= 3:
                lm.status = LandmarkStatus.REJECTED
                del self.features[fid]
                self.missed_frames.pop(fid, None)

        if self.prev_T_W_C is not None and self.prev_frame_id is not None:
            for fid in sorted(curr):
                if fid in self.features:
                    continue
                if fid in self.retired_ids:
                    continue
                if fid not in self.prev_pixels:
                    continue
                if fid not in self.pending:
                    if len(self.features) + len(self.pending) >= self.settings.max_pool_size:
                        continue
                    self.pending[fid] = PendingFeature(self.prev_frame_id, self.prev_T_W_C.copy(), self.prev_pixels[fid].copy())

                pending = self.pending[fid]
                lm = self._try_birth(fid, pending, frame_id, curr[fid], T_W_C)
                if lm is not None:
                    self.features[fid] = lm
                    self.missed_frames[fid] = 0
                    del self.pending[fid]

        self.pending = {fid: p for fid, p in self.pending.items() if fid in curr}
        if not preserve_previous:
            self.prev_pixels = curr
            self.prev_frame_id = frame_id
            self.prev_T_W_C = T_W_C.copy()

    def retire(self, feature_ids) -> None:
        ids = {int(fid) for fid in feature_ids}
        if not ids:
            return
        self.retired_ids.update(ids)
        for fid in ids:
            self.features.pop(fid, None)
            self.pending.pop(fid, None)
            self.prev_pixels.pop(fid, None)
            self.missed_frames.pop(fid, None)

    def _try_birth(
        self,
        fid: int,
        pending: PendingFeature,
        frame_id: int,
        pixel: np.ndarray,
        T_W_C: np.ndarray,
    ) -> LandmarkFilter | None:
        if np.linalg.norm(pixel - pending.pixel) < self.settings.birth_min_flow_px:
            return None

        b_anchor = bearing_from_pixel(pending.pixel, self.K)
        b_current = bearing_from_pixel(pixel, self.K)
        T_A_C = np.linalg.inv(pending.T_W_C) @ T_W_C
        R_A_C = T_A_C[:3, :3]
        t_A_C = T_A_C[:3, 3]
        if np.linalg.norm(np.cross(b_anchor, R_A_C @ b_current)) < self.settings.min_parallax_sin:
            return None

        ranges = two_ray_ranges(b_anchor, b_current, R_A_C, t_A_C)
        if ranges is None:
            return None
        range_anchor, range_current = ranges
        if (
            not np.isfinite(range_anchor)
            or not np.isfinite(range_current)
            or range_anchor < self.settings.min_depth
            or range_current < self.settings.min_depth
            or range_anchor > self.settings.max_depth
            or range_current > self.settings.max_depth
        ):
            return None

        lm = LandmarkFilter.from_anchor_pixel(
            landmark_id=fid,
            anchor_frame_id=pending.frame_id,
            pixel=pending.pixel,
            K=self.K,
            initial_depth=range_anchor,
            depth_sigma=self.settings.initial_depth_sigma,
            pixel_sigma=self.settings.sigma_pixel,
            anchor_T_W_C=pending.T_W_C,
        )
        T_C_A = np.linalg.inv(T_W_C) @ pending.T_W_C
        if not lm.update(
            frame_id,
            pixel,
            T_C_A,
            self.K,
            np.eye(2) * self.settings.sigma_pixel * self.settings.sigma_pixel,
            bias_rw_sigma=self.settings.bias_walk_sigma,
            gate_chi2=self.settings.mahalanobis_gate_chi2,
        ):
            return None
        return lm

    def has_track(self, fid: int) -> bool:
        return fid in self.features or fid in self.pending

    def feature(self, fid: int) -> LandmarkFilter | None:
        return self.features.get(fid)

    def feature_count(self) -> int:
        return len(self.features)

    def query(self, fid: int, T_W_C: np.ndarray | None = None, require_converged: bool = True) -> tuple[float, float]:
        lm = self.features.get(fid)
        if lm is None:
            return -1.0, float("inf")
        T_C_A = np.linalg.inv(T_W_C) @ lm.anchor_T_W_C if T_W_C is not None else np.eye(4)
        point_c = lm.current_position(T_C_A)
        depth = float(point_c[2])
        depth_var = lm.depth_variance(T_C_A)
        if not require_converged:
            if depth <= 0 or not np.isfinite(depth) or not np.isfinite(depth_var):
                return -1.0, float("inf")
            return depth, depth_var
        if (
            depth <= 0
            or len(lm.observations) < self.settings.min_track_length
            or depth_var > self.settings.conv_depth_variance
        ):
            return -1.0, float("inf")
        return depth, depth_var

    def mature_landmarks(
        self,
        T_W_C: np.ndarray,
        image: np.ndarray | None = None,
        affine_a: float = 0.0,
        affine_b: float = 0.0,
        require_converged: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points_w = []
        intensities = []
        a_vals = []
        b_vals = []
        ids = []

        T_C_W = np.linalg.inv(T_W_C)
        h = image.shape[0] if image is not None else None
        w = image.shape[1] if image is not None else None
        pattern_dx = np.array([0, 1, -1, 0, 0])
        pattern_dy = np.array([0, 0, 0, 1, -1])

        for fid, lm in self.features.items():
            depth, _ = self.query(fid, T_W_C, require_converged=require_converged)
            if depth <= 0:
                continue
            point_w = lm.anchor_T_W_C[:3, :3] @ lm.anchor_point() + lm.anchor_T_W_C[:3, 3]
            point_c = T_C_W[:3, :3] @ point_w + T_C_W[:3, 3]
            if point_c[2] <= 0:
                continue

            if image is not None:
                uv = project(point_c, self.K)
                u = int(round(uv[0]))
                v = int(round(uv[1]))
                if u < 1 or v < 1 or u >= w - 1 or v >= h - 1:
                    continue
                patch = image[v + pattern_dy, u + pattern_dx].astype(np.float32)
            else:
                patch = np.zeros(5, dtype=np.float32)

            points_w.append(point_w)
            intensities.append(patch)
            a_vals.append(affine_a)
            b_vals.append(affine_b)
            ids.append(fid)

        if not points_w:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 5), dtype=np.float32),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=int),
            )

        return (
            np.asarray(points_w, dtype=np.float64),
            np.asarray(intensities, dtype=np.float32),
            np.asarray(a_vals, dtype=np.float64),
            np.asarray(b_vals, dtype=np.float64),
            np.asarray(ids, dtype=int),
        )
