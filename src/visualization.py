import cv2
import numpy as np


def colorize_scalar(values, colormap=cv2.COLORMAP_JET, invert=False):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    lo = np.percentile(values, 5)
    hi = np.percentile(values, 95)
    if hi > lo:
        norm = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    else:
        norm = np.zeros_like(values)
    if invert:
        norm = 1.0 - norm
    colors_bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
    return colors_bgr.reshape(-1, 3)[:, ::-1]


def project_points(T_W_C, points_w, K, image_shape):
    if len(points_w) == 0:
        return np.empty((0, 2)), np.empty((0,)), np.empty((0,), dtype=bool)
    T_C_W = np.linalg.inv(T_W_C)
    pts_c = points_w @ T_C_W[:3, :3].T + T_C_W[:3, 3]
    z = pts_c[:, 2]
    valid_z = z > 0.01
    z_safe = np.clip(z, 0.01, None)
    u = K[0, 0] * pts_c[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * pts_c[:, 1] / z_safe + K[1, 2]
    h, w = image_shape
    valid = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return np.stack([u, v], axis=1), z, valid


def _log_keyframes(rr, K, image_shape, keyframes):
    rr.log("world/estimated/keyframes", rr.Clear(recursive=True))
    frustums = []
    colors = []
    scale = 0.18
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = image_shape
    corners_c = np.array([
        [0.0, 0.0, 0.0],
        [(-cx) / fx * scale, (-cy) / fy * scale, scale],
        [(w - cx) / fx * scale, (-cy) / fy * scale, scale],
        [(w - cx) / fx * scale, (h - cy) / fy * scale, scale],
        [(-cx) / fx * scale, (h - cy) / fy * scale, scale],
    ])
    for kf_idx, kf in enumerate(keyframes):
        kf_path = f"world/estimated/keyframes/kf_{kf_idx}"
        rr.log(kf_path, rr.Transform3D(translation=kf.T_W_C[:3, 3], mat3x3=kf.T_W_C[:3, :3]))
        rr.log(f"{kf_path}/image", rr.Pinhole(
            image_from_camera=K,
            resolution=[image_shape[1], image_shape[0]],
        ))
        corners_w = corners_c @ kf.T_W_C[:3, :3].T + kf.T_W_C[:3, 3]
        o, c0, c1, c2, c3 = corners_w
        frustums.extend([
            np.array([o, c0]),
            np.array([o, c1]),
            np.array([o, c2]),
            np.array([o, c3]),
            np.array([c0, c1, c2, c3, c0]),
        ])
        color = [255, 220, 80] if kf_idx == len(keyframes) - 1 else [160, 160, 160]
        colors.extend([color] * 5)

    if frustums:
        rr.log("world/estimated/keyframes/frustums", rr.LineStrips3D(
            frustums,
            colors=np.asarray(colors, dtype=np.uint8),
            radii=0.003,
        ))


def _log_image_landmarks(rr, mode, T_W_C, K, image_shape, point_cloud, tracked_points):
    if mode == "mono":
        overlay_points = point_cloud[0]
        show_depth_labels = True
    else:
        overlay_points = tracked_points
        show_depth_labels = False

    pts_2d, depths, visible = project_points(T_W_C, overlay_points, K, image_shape)
    if not np.any(visible):
        rr.log("world/estimated/camera/image/tracked_points", rr.Clear(recursive=False))
        rr.log("world/estimated/camera/image/landmark_depths", rr.Clear(recursive=False))
        return

    pts_visible = pts_2d[visible]
    depths_visible = depths[visible]
    rr.log("world/estimated/camera/image/tracked_points", rr.Points2D(
        pts_visible,
        colors=colorize_scalar(depths_visible, invert=True),
        radii=2.0,
    ))

    if not show_depth_labels:
        rr.log("world/estimated/camera/image/landmark_depths", rr.Clear(recursive=False))
        return

    label_count = min(120, len(pts_visible))
    label_idx = np.linspace(0, len(pts_visible) - 1, label_count, dtype=int)
    depth_labels = [f"{depths_visible[i]:.2f}m" for i in label_idx]
    rr.log("world/estimated/camera/image/landmark_depths", rr.Points2D(
        pts_visible[label_idx],
        colors=np.full((label_count, 3), 255, dtype=np.uint8),
        radii=1.0,
        labels=depth_labels,
    ))


def _circle_strips_2d(points, radius=4.0, segments=16):
    if len(points) == 0:
        return []
    theta = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    unit = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return [p + radius * unit for p in points]


def _log_ekf_image_landmarks(rr, ekf_landmarks):
    if ekf_landmarks is None or len(ekf_landmarks["pixels"]) == 0:
        rr.log("world/estimated/camera/image/ekf_landmarks", rr.Clear(recursive=False))
        return

    rr.log("world/estimated/camera/image/ekf_landmarks", rr.LineStrips2D(
        _circle_strips_2d(ekf_landmarks["pixels"]),
        colors=np.tile(np.array([[80, 220, 255]], dtype=np.uint8), (len(ekf_landmarks["pixels"]), 1)),
        radii=1.0,
    ))


def _log_keyframe_klt_residuals(rr, keyframe_klt_residuals):
    if keyframe_klt_residuals is None or len(keyframe_klt_residuals["segments"]) == 0:
        rr.log("world/estimated/camera/image/keyframe_klt_residuals", rr.Clear(recursive=False))
        return

    residuals = keyframe_klt_residuals["residuals"]
    colors = colorize_scalar(residuals, colormap=cv2.COLORMAP_TURBO)
    rr.log("world/estimated/camera/image/keyframe_klt_residuals", rr.LineStrips2D(
        keyframe_klt_residuals["segments"],
        colors=colors,
        radii=1.0,
    ))


def _log_point_cloud(rr, point_cloud):
    points, intensities, _, _ = point_cloud
    if len(points) == 0:
        rr.log("world/estimated/points", rr.Clear(recursive=False))
        return

    intensity = np.clip(intensities, 0, 255).astype(np.uint8)
    colors = np.stack([intensity, intensity, intensity], axis=-1)
    rr.log("world/estimated/points", rr.Points3D(points, colors=colors))


def _log_ekf_patch_glyphs(rr, T_W_C, ekf_landmarks, patch_size=0.08, std_scale=0.25):
    if ekf_landmarks is None or len(ekf_landmarks["points_w"]) == 0:
        rr.log("world/estimated/ekf_landmarks", rr.Clear(recursive=True))
        return

    points_w = ekf_landmarks["points_w"]
    points_c = ekf_landmarks["points_c"]
    depth_std = ekf_landmarks["depth_std"]
    R_W_C = T_W_C[:3, :3]
    right_w = R_W_C[:, 0]
    down_w = R_W_C[:, 1]
    forward_w = R_W_C[:, 2]

    patch_strips = []
    normal_strips = []
    colors = []
    normal_colors = []
    for point_w, point_c, std in zip(points_w, points_c, depth_std):
        scale = patch_size * max(float(point_c[2]), 0.2)
        dx = right_w * scale
        dy = down_w * scale
        patch_strips.append(np.array([
            point_w - dx - dy,
            point_w + dx - dy,
            point_w + dx + dy,
            point_w - dx + dy,
            point_w - dx - dy,
        ]))

        normal_len = min(max(float(std) * std_scale, 0.03), 1.0)
        normal_strips.append(np.array([point_w, point_w + forward_w * normal_len]))
        colors.append([80, 220, 255])
        normal_colors.append([255, 190, 60])

    rr.log("world/estimated/ekf_landmarks/patches", rr.LineStrips3D(
        patch_strips,
        colors=np.asarray(colors, dtype=np.uint8),
        radii=0.002,
    ))
    rr.log("world/estimated/ekf_landmarks/depth_std_normals", rr.LineStrips3D(
        normal_strips,
        colors=np.asarray(normal_colors, dtype=np.uint8),
        radii=0.004,
    ))


def projected_depth_discontinuity(T_W_C, point_cloud, K, image_shape, max_pixel_gap=8.0):
    points = point_cloud[0]
    pts_2d, depths, visible = project_points(T_W_C, points, K, image_shape)
    if np.sum(visible) < 2:
        return {
            "visible": int(np.sum(visible)),
            "pairs": 0,
            "median_abs": 0.0,
            "p90_abs": 0.0,
            "median_rel": 0.0,
        }

    pts_visible = pts_2d[visible]
    depths_visible = depths[visible]
    row_bin = np.floor(pts_visible[:, 1] / max_pixel_gap).astype(int)
    order = np.lexsort((pts_visible[:, 0], row_bin))
    pts_sorted = pts_visible[order]
    depth_sorted = depths_visible[order]
    row_sorted = row_bin[order]

    same_row = row_sorted[1:] == row_sorted[:-1]
    dx = np.abs(pts_sorted[1:, 0] - pts_sorted[:-1, 0])
    close = same_row & (dx <= max_pixel_gap)
    if not np.any(close):
        return {
            "visible": int(len(pts_visible)),
            "pairs": 0,
            "median_abs": 0.0,
            "p90_abs": 0.0,
            "median_rel": 0.0,
        }

    d0 = depth_sorted[:-1][close]
    d1 = depth_sorted[1:][close]
    abs_jump = np.abs(d1 - d0)
    rel_jump = abs_jump / np.maximum(np.minimum(np.abs(d0), np.abs(d1)), 1e-6)
    return {
        "visible": int(len(pts_visible)),
        "pairs": int(len(abs_jump)),
        "median_abs": float(np.median(abs_jump)),
        "p90_abs": float(np.percentile(abs_jump, 90)),
        "median_rel": float(np.median(rel_jump)),
    }


def _mono_keyframe_stats(keyframes):
    stats_md = "## Keyframe Statistics\n\n"
    stats_md += "| KF ID | Frame | Landmarks | Obs | Direct Inliers | Reason | Pos (x, y, z) |\n"
    stats_md += "|-------|-------|-----------|-----|----------------|--------|---------------|\n"
    for kf in keyframes:
        x, y, z = kf.T_W_C[:3, 3]
        stats_md += (
            f"| {kf.kf_id:04d} | {kf.frame_id:5d} | {len(kf.landmark_ids):9d} | "
            f"{len(kf.observations):3d} | {kf.direct_inliers:14d} | {kf.insertion_reason} | "
            f"({x:5.2f}, {y:5.2f}, {z:5.2f}) |\n"
        )
    return stats_md


def _stereo_keyframe_stats(keyframes, covisibility_map):
    stats_md = "## Keyframe Statistics\n\n"
    stats_md += "| KF ID | Pts Owned | Covisible | Pos (x, y, z) |\n"
    stats_md += "|-------|-----------|-----------|---------------|\n"
    for kf in keyframes:
        x, y, z = kf.T_W_C[:3, 3]
        stats_md += (
            f"| {kf.kf_id:04d} | {len(kf.pts_3d_c):4d} | "
            f"{covisibility_map.get(kf.kf_id, 0):4d} | ({x:5.2f}, {y:5.2f}, {z:5.2f}) |\n"
        )
    return stats_md


def _log_keyframe_stats(rr, mode, keyframes, frame_idx, covisibility_map):
    if not keyframes:
        rr.log("metrics/keyframe_stats", rr.Clear(recursive=False))
        return

    if mode == "mono":
        stats_md = _mono_keyframe_stats(keyframes)
        heading = "MONO KF STATS"
    else:
        stats_md = _stereo_keyframe_stats(keyframes, covisibility_map)
        heading = "KF STATS"

    rr.log("metrics/keyframe_stats", rr.TextDocument(text=stats_md, media_type=rr.MediaType.MARKDOWN))
    if frame_idx > 0 and frame_idx % 50 == 0:
        print(f"\n--- FRAME {frame_idx} {heading} ---")
        print(stats_md)


def log_vo_visualization(
    rr,
    frame_idx,
    mode,
    image,
    K,
    T_W_C,
    trajectory,
    keyframes,
    point_cloud,
    tracked_points=None,
    covisibility_map=None,
    ekf_landmarks=None,
    keyframe_klt_residuals=None,
):
    tracked_points = np.empty((0, 3), dtype=np.float64) if tracked_points is None else tracked_points
    covisibility_map = {} if covisibility_map is None else covisibility_map

    rr.log("world/estimated/trajectory", rr.LineStrips3D([trajectory], colors=[[255, 100, 100]]))
    rr.log("world/estimated/camera", rr.Transform3D(translation=T_W_C[:3, 3], mat3x3=T_W_C[:3, :3]))

    t_view = T_W_C[:3, 3] + T_W_C[:3, :3] @ np.array([0.0, -0.5, -1.5])
    rr.log("world/view_camera", rr.Transform3D(translation=t_view, mat3x3=T_W_C[:3, :3]))
    rr.log("world/estimated/camera/image", rr.Pinhole(
        image_from_camera=K,
        resolution=[image.shape[1], image.shape[0]],
    ))

    _log_keyframes(rr, K, image.shape, keyframes)
    rr.log("world/estimated/camera/image", rr.Image(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)))
    _log_image_landmarks(rr, mode, T_W_C, K, image.shape, point_cloud, tracked_points)
    _log_ekf_image_landmarks(rr, ekf_landmarks)
    _log_keyframe_klt_residuals(rr, keyframe_klt_residuals)
    _log_point_cloud(rr, point_cloud)
    _log_ekf_patch_glyphs(rr, T_W_C, ekf_landmarks)
    _log_keyframe_stats(rr, mode, keyframes, frame_idx, covisibility_map)
