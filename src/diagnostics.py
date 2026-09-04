import numpy as np
import logging
import cv2

logging.basicConfig(level=logging.INFO, format='[DEBUG-MONITOR] %(message)s')
logger = logging.getLogger('vo_monitor')


def stereo_flow_reprojection_error(
    prev_image,
    image,
    prev_depth,
    K,
    prev_T_W_C,
    T_W_C,
    point_cloud,
    max_points=1000,
    min_depth=0.2,
    max_depth=20.0,
):
    points_w = point_cloud[0] if isinstance(point_cloud, tuple) else point_cloud
    points_w = np.asarray(points_w, dtype=np.float64)
    if (
        prev_image is None
        or image is None
        or prev_depth is None
        or prev_T_W_C is None
        or len(points_w) == 0
    ):
        return _empty_stereo_flow_reprojection_stats()

    prev_image_u8 = prev_image.astype(np.uint8)
    image_u8 = image.astype(np.uint8)
    depth = np.asarray(prev_depth, dtype=np.float32)
    h, w = prev_image_u8.shape
    T_prev_C_W = np.linalg.inv(prev_T_W_C)
    pts_prev_c = points_w @ T_prev_C_W[:3, :3].T + T_prev_C_W[:3, 3]
    z_prev = pts_prev_c[:, 2]
    z_safe = np.clip(z_prev, 1e-9, None)
    uv_prev = np.column_stack([
        K[0, 0] * pts_prev_c[:, 0] / z_safe + K[0, 2],
        K[1, 1] * pts_prev_c[:, 1] / z_safe + K[1, 2],
    ])
    valid = (
        (z_prev > min_depth)
        & (uv_prev[:, 0] >= 2)
        & (uv_prev[:, 0] < w - 2)
        & (uv_prev[:, 1] >= 2)
        & (uv_prev[:, 1] < h - 2)
    )
    if not np.any(valid):
        return _empty_stereo_flow_reprojection_stats(projected=int(len(points_w)))

    candidate_idx = np.flatnonzero(valid)
    if len(candidate_idx) > max_points:
        candidate_idx = candidate_idx[:max_points]
    p0 = uv_prev[candidate_idx].astype(np.float32)
    u0 = np.clip(np.round(p0[:, 0]).astype(int), 0, w - 1)
    v0 = np.clip(np.round(p0[:, 1]).astype(int), 0, h - 1)
    stereo_depth = depth[v0, u0].astype(np.float64)
    valid_depth = (stereo_depth > min_depth) & (stereo_depth < max_depth) & np.isfinite(stereo_depth)
    if not np.any(valid_depth):
        return _empty_stereo_flow_reprojection_stats(projected=int(np.sum(valid)), sampled=len(candidate_idx))

    candidate_idx = candidate_idx[valid_depth]
    p0 = p0[valid_depth]
    stereo_depth = stereo_depth[valid_depth]
    p1, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_image_u8,
        image_u8,
        p0.reshape(-1, 1, 2),
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    if p1 is None or status is None:
        return _empty_stereo_flow_reprojection_stats(
            projected=int(np.sum(valid)),
            sampled=int(len(candidate_idx)),
            valid_depth=int(len(candidate_idx)),
        )
    p1 = p1.reshape(-1, 2).astype(np.float64)
    tracked = status.reshape(-1).astype(bool)
    tracked &= (p1[:, 0] >= 2) & (p1[:, 0] < w - 2) & (p1[:, 1] >= 2) & (p1[:, 1] < h - 2)
    if not np.any(tracked):
        return _empty_stereo_flow_reprojection_stats(
            projected=int(np.sum(valid)),
            sampled=int(len(candidate_idx)),
            valid_depth=int(len(candidate_idx)),
        )

    tracked_idx = candidate_idx[tracked]
    flow_pixels = p1[tracked]
    T_C_W = np.linalg.inv(T_W_C)
    pts_cur_c = points_w[tracked_idx] @ T_C_W[:3, :3].T + T_C_W[:3, 3]
    z_cur = pts_cur_c[:, 2]
    z_cur_safe = np.clip(z_cur, 1e-9, None)
    mono_pixels = np.column_stack([
        K[0, 0] * pts_cur_c[:, 0] / z_cur_safe + K[0, 2],
        K[1, 1] * pts_cur_c[:, 1] / z_cur_safe + K[1, 2],
    ])
    visible_cur = (
        (z_cur > min_depth)
        & (mono_pixels[:, 0] >= 2)
        & (mono_pixels[:, 0] < w - 2)
        & (mono_pixels[:, 1] >= 2)
        & (mono_pixels[:, 1] < h - 2)
    )
    if not np.any(visible_cur):
        return _empty_stereo_flow_reprojection_stats(
            projected=int(np.sum(valid)),
            sampled=int(len(candidate_idx)),
            valid_depth=int(len(candidate_idx)),
            tracked=int(np.sum(tracked)),
        )

    errors = np.linalg.norm(mono_pixels[visible_cur] - flow_pixels[visible_cur], axis=1)
    depth_ratio = z_prev[tracked_idx][visible_cur] / stereo_depth[tracked][visible_cur]
    return {
        "projected": int(np.sum(valid)),
        "sampled": int(len(candidate_idx)),
        "valid_depth": int(len(candidate_idx)),
        "tracked": int(np.sum(tracked)),
        "compared": int(len(errors)),
        "median_px": float(np.median(errors)) if len(errors) else 0.0,
        "p90_px": float(np.percentile(errors, 90)) if len(errors) else 0.0,
        "median_depth_ratio": float(np.median(depth_ratio)) if len(depth_ratio) else 0.0,
    }


def _empty_stereo_flow_reprojection_stats(projected=0, sampled=0, valid_depth=0, tracked=0):
    return {
        "projected": int(projected),
        "sampled": int(sampled),
        "valid_depth": int(valid_depth),
        "tracked": int(tracked),
        "compared": 0,
        "median_px": 0.0,
        "p90_px": 0.0,
        "median_depth_ratio": 0.0,
    }

class VOMonitor:
    def __init__(self):
        self.last_T_W_C = None
        
    def check_pose_jump(self, frame_id, stage, T_W_C, threshold=0.5):
        if self.last_T_W_C is not None:
            dt = np.linalg.norm(T_W_C[:3, 3] - self.last_T_W_C[:3, 3])
            if dt > threshold:
                logger.error(f"FRAME {frame_id} | POSE JUMP DETECTED in {stage}! dt = {dt:.3f} meters")
                logger.error(f"Old translation: {self.last_T_W_C[:3, 3]}")
                logger.error(f"New translation: {T_W_C[:3, 3]}")
        
    def set_baseline(self, T_W_C):
        self.last_T_W_C = T_W_C.copy()

    def check_landmark_consistency(self, frame_id, stage, pts, ints, a, b):
        if len(pts) != len(ints) or len(pts) != len(a) or len(pts) != len(b):
            logger.error(f"FRAME {frame_id} | ARRAY SIZE MISMATCH in {stage}!")
            logger.error(f"pts: {len(pts)}, ints: {len(ints)}, a: {len(a)}, b: {len(b)}")
            
        if len(pts) > 0:
            nan_pts = np.sum(np.isnan(pts))
            if nan_pts > 0:
                logger.error(f"FRAME {frame_id} | NANs DETECTED in {stage}! {nan_pts} NaN values in points.")

    def check_landmark_jump(self, frame_id, pts_before, pts_after, threshold=0.2):
        if len(pts_before) == len(pts_after) and len(pts_before) > 0:
            diffs = np.linalg.norm(pts_after - pts_before, axis=1)
            mean_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            if mean_diff > threshold or max_diff > threshold * 5:
                logger.error(f"FRAME {frame_id} | LANDMARK JUMP DETECTED during BA! Mean shift: {mean_diff:.4f}m, Max shift: {max_diff:.4f}m")

    def check_alignment_jump(self, frame_id, s, R, t):
        if not hasattr(self, 'last_s'):
            self.last_s = s
            self.last_R = R
            self.last_t = t
        else:
            ds = abs(s - self.last_s)
            dt = np.linalg.norm(t - self.last_t)
            dR = np.linalg.norm(R - self.last_R)
            if ds > 0.05 or dt > 0.1 or dR > 0.05:
                logger.warning(f"FRAME {frame_id} | ALIGNMENT JUMP! ds: {ds:.4f}, dt: {dt:.4f}, dR: {dR:.4f}")
            self.last_s = s
            self.last_R = R
            self.last_t = t
