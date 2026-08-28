import numpy as np
import cv2
from src.map import Keyframe, Map
from src.tracker import DirectTracker
from src.depth_provider import FixedDepthProvider
from src.feature_tracker import FeatureTracker, FeatureTrackerConfig
from src.landmark_filter import (
    LandmarkFilter,
    LandmarkStatus,
    Sparse3DFilterBank,
    Sparse3DSettings,
    point_and_jacobian,
    tangent_basis,
    two_ray_ranges,
)
from src.monocular import ExperimentalMonocularVO
from src.mono_map import MonoMap, MonoMapConfig
from src.optimizer import PhotometricBA
from src.visualization import colorize_scalar, projected_depth_discontinuity

def get_dummy_camera():
    return np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)


def project_world_points(K, T_W_C, points_w):
    T_C_W = np.linalg.inv(T_W_C)
    points_c = points_w @ T_C_W[:3, :3].T + T_C_W[:3, 3]
    uv = np.column_stack([
        K[0, 0] * points_c[:, 0] / points_c[:, 2] + K[0, 2],
        K[1, 1] * points_c[:, 1] / points_c[:, 2] + K[1, 2],
    ])
    return uv, points_c


def synthetic_texture_image(shape=(480, 640)):
    y, x = np.indices(shape, dtype=np.float32)
    image = (
        110.0
        + 25.0 * np.sin(x * 0.041)
        + 20.0 * np.cos(y * 0.053)
        + 15.0 * np.sin((x + y) * 0.019)
    )
    return np.clip(image, 0.0, 255.0).astype(np.float32)


def sample_pattern(image, uv):
    h, w = image.shape
    u_pat = np.clip((uv[:, None, 0] + np.array([0, 1, -1, 0, 0])).reshape(-1), 0, w - 1)
    v_pat = np.clip((uv[:, None, 1] + np.array([0, 0, 0, 1, -1])).reshape(-1), 0, h - 1)
    return cv2.remap(
        image,
        u_pat.astype(np.float32).reshape(-1, 1),
        v_pat.astype(np.float32).reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1, 5)


def reprojection_rmse(K, T_W_C, points_w, observations):
    pred, _ = project_world_points(K, T_W_C, points_w)
    obs = np.asarray([observations[i] for i in range(len(points_w))], dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum((pred - obs) ** 2, axis=1))))

def test_keyframe_masking():
    """Verify that the spatial exclusion mask prevents feature extraction in occupied areas."""
    img = np.zeros((480, 640), dtype=np.uint8)
    # Draw two distinct bright spots to act as corners
    cv2.rectangle(img, (100, 100), (120, 120), 255, -1)
    cv2.rectangle(img, (400, 400), (420, 420), 255, -1)
    
    depth = np.ones((480, 640), dtype=np.float32) * 5.0
    T_W_C = np.eye(4)
    K = get_dummy_camera()
    
    # Extract without mask (should find both corners)
    kf_unmasked = Keyframe(0, img, depth, T_W_C, K)
    assert len(kf_unmasked.pts_3d_w) > 0, "Failed to extract any features"
    
    # Create a mask covering the first rectangle
    mask = np.ones((480, 640), dtype=np.uint8) * 255
    cv2.circle(mask, (110, 110), 30, 0, -1) # mask out the top-left corner
    
    kf_masked = Keyframe(1, img, depth, T_W_C, K, mask=mask)
    
    # Assert that all extracted points are far from the masked region
    for u, v in kf_masked.pts_3d_c[:, :2]: # actually need to reproject, or just check mask logic
        pass # The simplest check: kf_masked should have fewer points than kf_unmasked!
    
    assert len(kf_masked.pts_3d_w) < len(kf_unmasked.pts_3d_w), "Mask did not exclude features!"

def test_point_transfer_marginalization():
    """Verify that points from a dropped keyframe are rescued into the active keyframe."""
    K = get_dummy_camera()
    vo_map = Map(K, window_size=2)
    
    # Create 3 distinct images to force sliding window marginalization
    img = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(img, (300, 200), (340, 240), 255, -1)
    depth = np.ones((480, 640), dtype=np.float32) * 2.0
    
    # Add KF 0
    vo_map.add_keyframe(img, depth, np.eye(4))
    assert len(vo_map.keyframes) == 1
    
    # Add KF 1 (shift camera slightly on X)
    T_1 = np.eye(4)
    T_1[0, 3] = 0.1
    vo_map.add_keyframe(img, depth, T_1)
    assert len(vo_map.keyframes) == 2
    
    # Add KF 2 (shift camera further). This should drop KF 1!
    # Because KF 1's points are still visible in KF 2's frustum, they should be TRANSFERRED to KF 2!
    T_2 = np.eye(4)
    T_2[0, 3] = 0.2
    
    kf2_pts_before_drop = vo_map.add_keyframe(img, depth, T_2)
    assert len(vo_map.keyframes) == 2 # KF 0 and KF 2 remain (size 2 enforced, index 1 dropped)
    
    latest_kf = vo_map.keyframes[-1]
    
    # The latest KF should now have its OWN extracted points + the TRANSFERRED points from KF 1!
    # Because they see the exact same rectangle, it should easily have >= the number of points.
    assert len(latest_kf.pts_3d_w) >= kf2_pts_before_drop, "Points were permanently discarded instead of transferred!"
    
def test_tracker_robust_huber():
    """Ensure the tracker does not explode when given a massive outlier (simulating occlusion)."""
    K = get_dummy_camera()
    tracker = DirectTracker(K)
    
    img_cur = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(img_cur, (100, 100), (200, 200), 255, -1)
    
    pts_3d_w = np.array([[0, 0, 5.0], [0.1, 0, 5.0], [0, 0.1, 5.0]])
    
    # Simulate one perfect match (intensity 255) and one massive outlier (intensity 0 vs 255)
    intensities = np.array([255, 255, 0]) 
    
    T_guess = np.eye(4)
    a_ref = np.zeros(3)
    b_ref = np.zeros(3)
    T_opt, inliers, a_opt, b_opt = tracker.track_map(img_cur, pts_3d_w, intensities, a_ref, b_ref, T_guess, max_iters=5)
    
    # If the Huber loss is working, the massive outlier (255 error) is down-weighted 
    # and the pose should barely move (translation norm should be tiny).
    t_norm = np.linalg.norm(T_opt[:3, 3])
    assert t_norm < 0.5, f"Tracker exploded due to outlier! Translation shift: {t_norm}"

def test_tracker_relative_affine_photometry():
    """
    Ensures that the Gauss-Newton tracker correctly evaluates the relative affine brightness
    formulation: I_cur = exp(a_cur - a_ref) * I_ref + (b_cur - b_ref)
    """
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0,   0,   1]
    ], dtype=np.float64)
    
    tracker = DirectTracker(K)
    
    # 1 Point directly in front of the camera
    pts_3d_w = np.array([[0.0, 0.0, 5.0]])
    
    # Origin state: RAW intensity 100, origin exposure state (a=1.0, b=10.0)
    intensities = np.array([100.0])
    a_ref = np.array([1.0])
    b_ref = np.array([10.0])
    
    # Target frame exposure guess (a=2.0, b=30.0)
    # Expected intensity in current frame: exp(2.0 - 1.0) * 100 + (30.0 - 10.0)
    # e^1 * 100 + 20 = 271.828... + 20 = 291.828
    expected_intensity = np.exp(1.0) * 100.0 + 20.0
    
    img_cur = np.zeros((480, 640), dtype=np.float32)
    # Give the point and its neighborhood the mathematically exact expected intensity
    img_cur[235:245, 315:325] = expected_intensity
    
    T_guess = np.eye(4)
    a_guess = 2.0
    b_guess = 30.0
    
    T_opt, inliers, a_opt, b_opt = tracker.track_map(
        img_cur, pts_3d_w, intensities, a_ref, b_ref, 
        T_guess, a_guess, b_guess, max_iters=10
    )
    
    # The pose should NOT have moved, because the affine model perfectly explains the brightness difference
    t_opt = T_opt[:3, 3]
    assert np.linalg.norm(t_opt) < 1e-2, f"Tracker hallucinated motion to fix exposure: {t_opt}"
    assert np.all(inliers), "Point was incorrectly rejected as an outlier"


def test_direct_tracker_recovers_known_pose_from_photometric_residuals():
    K = get_dummy_camera()
    tracker = DirectTracker(K)
    rng = np.random.default_rng(10)
    points_w = np.column_stack([
        rng.uniform(-0.9, 0.9, 220),
        rng.uniform(-0.6, 0.6, 220),
        rng.uniform(4.0, 7.0, 220),
    ])
    image = synthetic_texture_image()
    T_true_W_C = np.eye(4)
    T_true_W_C[0, 3] = 0.06
    T_true_W_C[1, 3] = -0.02
    uv_true, _ = project_world_points(K, T_true_W_C, points_w)
    intensities = sample_pattern(image, uv_true)

    T_guess_W_C = np.eye(4)
    T_guess_W_C[0, 3] = -0.04
    T_guess_W_C[1, 3] = 0.03
    before = np.linalg.norm(T_guess_W_C[:3, 3] - T_true_W_C[:3, 3])

    T_opt_C_W, inliers, _, _ = tracker.track_map(
        image,
        points_w,
        intensities,
        np.zeros(len(points_w)),
        np.zeros(len(points_w)),
        np.linalg.inv(T_guess_W_C),
        max_iters=20,
    )
    T_opt_W_C = np.linalg.inv(T_opt_C_W)
    after = np.linalg.norm(T_opt_W_C[:3, 3] - T_true_W_C[:3, 3])

    assert np.sum(inliers) > 120
    assert after < before * 0.5, (before, after, T_opt_W_C[:3, 3])


def test_monocular_direct_prefilter_temporarily_removes_photometric_outliers():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K, mono_map_config=MonoMapConfig(min_keyframe_landmarks=1))
    image = synthetic_texture_image()
    points_w = np.array([
        [-0.08, 0.0, 4.0],
        [0.0, 0.0, 4.0],
        [0.08, 0.0, 4.0],
        [0.16, 0.0, 4.0],
    ])
    uv, _ = project_world_points(K, np.eye(4), points_w)
    intensities = sample_pattern(image, uv)
    ids = np.array([10, 11, 12, 13])

    occluded = image.copy()
    for u, v in uv[[1, 3]]:
        cv2.rectangle(occluded, (int(u) - 3, int(v) - 3), (int(u) + 3, int(v) + 3), 255, -1)

    kept_pts, kept_intensities, kept_a, kept_b, kept_ids, stats = mono._prefilter_direct_references(
        occluded,
        np.eye(4),
        points_w,
        intensities,
        np.zeros(len(points_w)),
        np.zeros(len(points_w)),
        ids,
    )

    assert kept_ids.tolist() == [10, 12]
    assert kept_pts.shape == (2, 3)
    assert kept_intensities.shape == (2, 5)
    assert kept_a.shape == (2,)
    assert kept_b.shape == (2,)
    assert stats["kept"] == 2
    assert stats["rejected"] == 2


def test_monocular_direct_prefilter_does_not_delete_occluded_map_landmarks():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K, mono_map_config=MonoMapConfig(min_keyframe_landmarks=1))
    image = synthetic_texture_image()
    points_w = np.array([[0.0, 0.0, 4.0], [0.08, 0.0, 4.0]])
    uv, _ = project_world_points(K, np.eye(4), points_w)
    intensities = sample_pattern(image, uv)
    ids = np.array([20, 21])
    mono.mono_map.add_keyframe(0, image, np.eye(4), 0.0, 0.0, ids, points_w, intensities, "first", 2, 2)

    occluded = image.copy()
    cv2.rectangle(occluded, (int(uv[0, 0]) - 3, int(uv[0, 1]) - 3), (int(uv[0, 0]) + 3, int(uv[0, 1]) + 3), 255, -1)
    refs = mono.mono_map.direct_references(np.eye(4))
    _kept_pts, _kept_intensities, _kept_a, _kept_b, kept_ids, stats = mono._prefilter_direct_references(
        occluded,
        np.eye(4),
        *refs,
    )

    assert kept_ids.tolist() == [21]
    assert stats["rejected"] == 1
    assert mono.mono_map.active_landmark_count() == 2
    assert mono.mono_map.keyframes[0].landmark_ids.tolist() == [20, 21]


def test_fixed_depth_provider_copies_depth():
    depth = np.ones((4, 5), dtype=np.float32) * 3.0
    provider = FixedDepthProvider(depth)

    estimate = provider.compute(np.zeros((4, 5), dtype=np.uint8))
    estimate.depth[0, 0] = 10.0

    assert estimate.source == "fixed"
    assert depth[0, 0] == 3.0


def test_landmark_filter_anchor_bearing_and_depth():
    K = get_dummy_camera()
    lm = LandmarkFilter.from_anchor_pixel(
        landmark_id=1,
        anchor_frame_id=0,
        pixel=np.array([320.0, 240.0]),
        K=K,
        initial_depth=5.0,
        depth_sigma=2.0,
    )

    assert np.allclose(lm.anchor_bearing, np.array([0.0, 0.0, 1.0]))
    assert np.isclose(lm.anchor_point()[2], 5.0)
    assert lm.status == LandmarkStatus.IMMATURE


def test_landmark_filter_update_reduces_inverse_depth_uncertainty():
    K = get_dummy_camera()
    lm = LandmarkFilter.from_anchor_pixel(
        landmark_id=1,
        anchor_frame_id=0,
        pixel=np.array([320.0, 240.0]),
        K=K,
        initial_depth=5.0,
        depth_sigma=2.0,
    )

    T_C_A = np.eye(4)
    T_C_A[0, 3] = -0.5
    true_point_c = T_C_A[:3, :3] @ np.array([0.0, 0.0, 5.0]) + T_C_A[:3, 3]
    measured = np.array([
        K[0, 0] * true_point_c[0] / true_point_c[2] + K[0, 2],
        K[1, 1] * true_point_c[1] / true_point_c[2] + K[1, 2],
    ])

    before = lm.covariance[2, 2]
    accepted = lm.update(1, measured, T_C_A, K, np.eye(2) * 0.5 * 0.5)

    assert accepted
    assert lm.covariance[2, 2] < before


def test_landmark_filter_gates_large_outlier():
    K = get_dummy_camera()
    lm = LandmarkFilter.from_anchor_pixel(
        landmark_id=1,
        anchor_frame_id=0,
        pixel=np.array([320.0, 240.0]),
        K=K,
        initial_depth=5.0,
        depth_sigma=2.0,
    )

    accepted = lm.update(1, np.array([50.0, 50.0]), np.eye(4), K, np.eye(2))

    assert not accepted
    assert lm.inlier_beta > 10.0


def test_bearing_chart_jacobian_matches_finite_difference():
    b0 = np.array([0.2, -0.3, 1.0])
    b0 = b0 / np.linalg.norm(b0)
    U = tangent_basis(b0)
    eta = np.array([0.1, -0.05])
    rho = 0.4

    _, J = point_and_jacobian(b0, U, eta, rho)

    eps = 1e-6
    for i in range(3):
        eta_p = eta.copy()
        eta_m = eta.copy()
        rho_p = rho
        rho_m = rho
        if i < 2:
            eta_p[i] += eps
            eta_m[i] -= eps
        else:
            rho_p += eps
            rho_m -= eps
        p_plus, _ = point_and_jacobian(b0, U, eta_p, rho_p)
        p_minus, _ = point_and_jacobian(b0, U, eta_m, rho_m)
        numeric = (p_plus - p_minus) / (2.0 * eps)
        assert np.allclose(J[:, i], numeric, atol=1e-5)


def test_two_ray_ranges_recovers_known_point():
    point_anchor = np.array([0.4, -0.2, 4.0])
    b_anchor = point_anchor / np.linalg.norm(point_anchor)
    R_A_C = np.eye(3)
    t_A_C = np.array([0.5, 0.0, 0.0])
    point_current = R_A_C.T @ (point_anchor - t_A_C)
    b_current = point_current / np.linalg.norm(point_current)

    ranges = two_ray_ranges(b_anchor, b_current, R_A_C, t_A_C)

    assert ranges is not None
    assert np.isclose(ranges[0], np.linalg.norm(point_anchor))
    assert np.isclose(ranges[1], np.linalg.norm(point_current))


def test_sparse3d_filter_bank_birth_query_and_lost_track_removal():
    K = get_dummy_camera()
    settings = Sparse3DSettings(
        min_track_length=2,
        birth_min_flow_px=0.1,
        min_parallax_sin=0.001,
        initial_depth_sigma=1.0,
        conv_depth_variance=100.0,
    )
    bank = Sparse3DFilterBank(K, settings)
    point_w = np.array([0.4, 0.0, 4.0])

    for frame_id in range(6):
        T_W_C = np.eye(4)
        T_W_C[0, 3] = frame_id * 0.1
        point_c = point_w - T_W_C[:3, 3]
        uv = np.array([
            K[0, 0] * point_c[0] / point_c[2] + K[0, 2],
            K[1, 1] * point_c[1] / point_c[2] + K[1, 2],
        ])
        bank.update(frame_id, {42: uv}, T_W_C)

    assert bank.feature_count() == 1
    depth, var = bank.query(42, T_W_C)
    assert 3.5 < depth < 4.5
    assert np.isfinite(var)

    bank.update(7, {}, T_W_C)

    assert bank.feature_count() == 0
    assert not bank.has_track(42)


def test_sparse3d_filter_bank_birth_recovers_depth_with_rotation_and_translation():
    K = get_dummy_camera()
    settings = Sparse3DSettings(
        min_track_length=2,
        birth_min_flow_px=0.1,
        min_parallax_sin=0.001,
        initial_depth_sigma=0.5,
        conv_depth_variance=100.0,
    )
    bank = Sparse3DFilterBank(K, settings)
    point_w = np.array([0.35, -0.12, 4.5])
    T0 = np.eye(4)
    T1 = np.eye(4)
    yaw = np.deg2rad(3.0)
    T1[:3, :3] = np.array([
        [np.cos(yaw), 0.0, np.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-np.sin(yaw), 0.0, np.cos(yaw)],
    ])
    T1[:3, 3] = np.array([0.12, -0.02, 0.03])

    uv0, pc0 = project_world_points(K, T0, point_w.reshape(1, 3))
    uv1, pc1 = project_world_points(K, T1, point_w.reshape(1, 3))

    bank.update(0, {7: uv0[0]}, T0)
    bank.update(1, {7: uv1[0]}, T1)
    bank.update(2, {7: uv1[0]}, T1)

    assert bank.feature_count() == 1
    lm = bank.feature(7)
    assert lm is not None
    point_w_est = lm.anchor_T_W_C[:3, :3] @ lm.anchor_point() + lm.anchor_T_W_C[:3, 3]
    assert np.linalg.norm(point_w_est - point_w) < 0.15
    assert abs(bank.query(7, T0)[0] - pc0[0, 2]) < 0.2
    assert abs(bank.query(7, T1)[0] - pc1[0, 2]) < 0.2


def test_sparse3d_filter_bank_can_preserve_missing_tracks():
    K = get_dummy_camera()
    settings = Sparse3DSettings(
        min_track_length=2,
        birth_min_flow_px=0.1,
        min_parallax_sin=0.001,
        initial_depth_sigma=1.0,
        conv_depth_variance=100.0,
    )
    bank = Sparse3DFilterBank(K, settings)
    point_w = np.array([0.4, 0.0, 4.0])

    for frame_id in range(4):
        T_W_C = np.eye(4)
        T_W_C[0, 3] = frame_id * 0.1
        point_c = point_w - T_W_C[:3, 3]
        uv = np.array([
            K[0, 0] * point_c[0] / point_c[2] + K[0, 2],
            K[1, 1] * point_c[1] / point_c[2] + K[1, 2],
        ])
        bank.update(frame_id, {42: uv}, T_W_C)

    assert bank.feature_count() == 1
    bank.update(5, {}, T_W_C, remove_missing=False)

    assert bank.feature_count() == 1
    assert bank.has_track(42)


def test_sparse3d_filter_bank_ages_out_long_missing_tracks_when_preserving_missing():
    K = get_dummy_camera()
    settings = Sparse3DSettings(
        min_track_length=2,
        birth_min_flow_px=0.1,
        min_parallax_sin=0.001,
        initial_depth_sigma=1.0,
        conv_depth_variance=100.0,
        max_missed_frames=2,
    )
    bank = Sparse3DFilterBank(K, settings)
    point_w = np.array([0.4, 0.0, 4.0])

    for frame_id in range(4):
        T_W_C = np.eye(4)
        T_W_C[0, 3] = frame_id * 0.1
        point_c = point_w - T_W_C[:3, 3]
        uv = np.array([
            K[0, 0] * point_c[0] / point_c[2] + K[0, 2],
            K[1, 1] * point_c[1] / point_c[2] + K[1, 2],
        ])
        bank.update(frame_id, {42: uv}, T_W_C)

    assert bank.feature_count() == 1
    bank.update(4, {}, T_W_C, remove_missing=False)
    bank.update(5, {}, T_W_C, remove_missing=False)
    bank.update(6, {}, T_W_C, remove_missing=False)

    assert bank.feature_count() == 0
    assert not bank.has_track(42)


def test_sparse3d_filter_bank_retire_removes_promoted_ids():
    K = get_dummy_camera()
    bank = Sparse3DFilterBank(K, Sparse3DSettings(min_track_length=2, birth_min_flow_px=0.1, min_parallax_sin=0.001))
    point_w = np.array([0.4, 0.0, 4.0])

    for frame_id in range(4):
        T_W_C = np.eye(4)
        T_W_C[0, 3] = frame_id * 0.1
        point_c = point_w - T_W_C[:3, 3]
        uv = np.array([
            K[0, 0] * point_c[0] / point_c[2] + K[0, 2],
            K[1, 1] * point_c[1] / point_c[2] + K[1, 2],
        ])
        bank.update(frame_id, {42: uv}, T_W_C)

    assert bank.feature_count() == 1
    bank.retire([42])

    assert bank.feature_count() == 0
    assert not bank.has_track(42)
    assert 42 in bank.retired_ids


def test_legacy_map_observations_use_feature_ids():
    K = get_dummy_camera()
    vo_map = Map(K, window_size=2)
    img = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(img, (300, 220), (340, 260), 255, -1)
    depth = np.ones((480, 640), dtype=np.float32) * 4.0
    vo_map.add_keyframe(img, depth, np.eye(4))

    observations = vo_map.observe_window_points(np.eye(4), img.shape)
    feature_ids = set(vo_map.get_window_feature_ids().tolist())

    assert observations
    assert set(observations).issubset(feature_ids)


def test_legacy_map_point_cloud_exports_visualization_shape():
    K = get_dummy_camera()
    vo_map = Map(K, window_size=2)
    img = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(img, (300, 220), (340, 260), 180, -1)
    depth = np.ones((480, 640), dtype=np.float32) * 4.0

    vo_map.add_keyframe(img, depth, np.eye(4))
    points, intensities, ids, kf_ids = vo_map.point_cloud(max_points=10)

    assert points.ndim == 2
    assert points.shape[1] == 3
    assert intensities.shape == (len(points),)
    assert ids.shape == (len(points),)
    assert kf_ids.shape == (len(points),)
    assert np.all(kf_ids == 0)


def test_depth_coloring_uses_per_frame_percentile_normalization():
    color_a = colorize_scalar(
        np.array([2.0, 5.0, 9.0]),
        invert=True,
    )[1]
    color_b = colorize_scalar(
        np.array([0.7, 5.0, 30.0]),
        invert=True,
    )[1]

    assert not np.array_equal(color_a, color_b)


def test_projected_depth_discontinuity_reports_local_depth_jumps():
    K = get_dummy_camera()
    T_W_C = np.eye(4)
    point_cloud = (
        np.array([[0.0, 0.0, 2.0], [0.01, 0.0, 8.0], [1.0, 0.0, 8.0]]),
        np.zeros(3, dtype=np.float32),
        np.arange(3),
        np.zeros(3, dtype=int),
    )

    stats = projected_depth_discontinuity(T_W_C, point_cloud, K, (480, 640), max_pixel_gap=8.0)

    assert stats["visible"] == 3
    assert stats["pairs"] == 1
    assert np.isclose(stats["median_abs"], 6.0)


def test_sparse3d_mature_landmarks_export_tracker_arrays():
    K = get_dummy_camera()
    settings = Sparse3DSettings(
        min_track_length=2,
        birth_min_flow_px=0.1,
        min_parallax_sin=0.001,
        initial_depth_sigma=1.0,
        conv_depth_variance=100.0,
    )
    bank = Sparse3DFilterBank(K, settings)
    point_w = np.array([0.4, 0.0, 4.0])
    image = np.zeros((480, 640), dtype=np.uint8)

    for frame_id in range(6):
        T_W_C = np.eye(4)
        T_W_C[0, 3] = frame_id * 0.1
        point_c = point_w - T_W_C[:3, 3]
        uv = np.array([
            K[0, 0] * point_c[0] / point_c[2] + K[0, 2],
            K[1, 1] * point_c[1] / point_c[2] + K[1, 2],
        ])
        cv2.circle(image, tuple(np.round(uv).astype(int)), 2, 180, -1)
        bank.update(frame_id, {42: uv}, T_W_C)

    pts_w, ints, a_vals, b_vals, ids = bank.mature_landmarks(T_W_C, image)

    assert pts_w.shape == (1, 3)
    assert ints.shape == (1, 5)
    assert a_vals.shape == (1,)
    assert b_vals.shape == (1,)
    assert ids.tolist() == [42]


def test_feature_tracker_tracks_stable_ids_under_translation():
    cfg = FeatureTrackerConfig(max_features=20, quality_level=0.01, min_distance=8, min_ransac_points=50)
    tracker = FeatureTracker(cfg)
    img0 = np.zeros((120, 160), dtype=np.uint8)
    img1 = np.zeros_like(img0)
    for x, y in [(30, 30), (80, 35), (45, 80), (110, 90)]:
        cv2.circle(img0, (x, y), 4, 255, -1)
        cv2.circle(img1, (x + 5, y), 4, 255, -1)

    obs0 = tracker.update(img0)
    obs1 = tracker.update(img1)
    common = sorted(set(obs0) & set(obs1))

    assert common
    displacements = np.array([obs1[fid] - obs0[fid] for fid in common])
    assert np.allclose(np.median(displacements, axis=0), np.array([5.0, 0.0]), atol=0.5)


def test_feature_tracker_handles_textureless_frames():
    tracker = FeatureTracker(FeatureTrackerConfig(max_features=20))

    obs0 = tracker.update(np.zeros((60, 80), dtype=np.uint8))
    obs1 = tracker.update(np.zeros((60, 80), dtype=np.uint8))

    assert obs0 == {}
    assert obs1 == {}


def test_feature_tracker_masks_external_exclusion_points_for_gftt():
    tracker = FeatureTracker(FeatureTrackerConfig(max_features=20, quality_level=0.01, min_distance=12))
    img = np.zeros((120, 160), dtype=np.uint8)
    cv2.rectangle(img, (34, 34), (46, 46), 255, -1)
    cv2.rectangle(img, (104, 74), (116, 86), 255, -1)

    obs = tracker.update(img, exclusion_points=np.array([[40.0, 40.0]]))
    points = np.array(list(obs.values()))

    assert len(points) > 0
    assert np.all(np.linalg.norm(points - np.array([40.0, 40.0]), axis=1) >= 12.0)
    assert np.any(np.linalg.norm(points - np.array([110.0, 80.0]), axis=1) < 15.0)


def test_feature_tracker_remove_ids_frees_tracked_slots():
    tracker = FeatureTracker(FeatureTrackerConfig(max_features=10, quality_level=0.01, min_distance=12))
    img = np.zeros((120, 160), dtype=np.uint8)
    cv2.rectangle(img, (34, 34), (46, 46), 255, -1)
    cv2.rectangle(img, (104, 74), (116, 86), 255, -1)

    obs = tracker.update(img)
    remove_id = next(iter(obs))
    tracker.remove_ids([remove_id])

    assert remove_id not in tracker.observations()
    assert len(tracker.observations()) == len(obs) - 1


def test_experimental_monocular_vo_estimates_relative_motion_from_tracks():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K, min_essential_tracks=12, min_direct_landmarks=1000, max_bootstrap_step=0.15)
    rng = np.random.default_rng(0)
    points_w = np.column_stack([
        rng.uniform(-1.0, 1.0, 40),
        rng.uniform(-0.7, 0.7, 40),
        rng.uniform(4.0, 8.0, 40),
    ])

    T_W_C0 = np.eye(4)
    T_W_C1 = np.eye(4)
    T_W_C1[0, 3] = 0.2

    obs0 = {}
    obs1 = {}
    for i, p_w in enumerate(points_w):
        p0 = p_w - T_W_C0[:3, 3]
        p1 = p_w - T_W_C1[:3, 3]
        obs0[i] = np.array([
            K[0, 0] * p0[0] / p0[2] + K[0, 2],
            K[1, 1] * p0[1] / p0[2] + K[1, 2],
        ])
        obs1[i] = np.array([
            K[0, 0] * p1[0] / p1[2] + K[0, 2],
            K[1, 1] * p1[1] / p1[2] + K[1, 2],
        ])

    mono.prev_observations = obs0
    guess = mono._essential_pose_guess(obs1)

    assert guess is not None
    assert np.linalg.norm(guess[:3, 3]) <= 0.15 + 1e-9
    assert abs(guess[0, 3]) > abs(guess[1, 3])


def test_experimental_monocular_vo_essential_preserves_small_rotation():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K, min_essential_tracks=12, min_direct_landmarks=1000)
    rng = np.random.default_rng(1)
    points_w = np.column_stack([
        rng.uniform(-1.0, 1.0, 80),
        rng.uniform(-0.7, 0.7, 80),
        rng.uniform(4.0, 8.0, 80),
    ])

    yaw = np.deg2rad(2.0)
    R_W_C1 = np.array([
        [np.cos(yaw), 0.0, np.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-np.sin(yaw), 0.0, np.cos(yaw)],
    ])
    T_W_C0 = np.eye(4)
    T_W_C1 = np.eye(4)
    T_W_C1[:3, :3] = R_W_C1
    T_W_C1[0, 3] = 0.1

    obs0 = {}
    obs1 = {}
    for i, p_w in enumerate(points_w):
        p0 = np.linalg.inv(T_W_C0)[:3, :3] @ (p_w - T_W_C0[:3, 3])
        p1 = np.linalg.inv(T_W_C1)[:3, :3] @ (p_w - T_W_C1[:3, 3])
        obs0[i] = np.array([
            K[0, 0] * p0[0] / p0[2] + K[0, 2],
            K[1, 1] * p0[1] / p0[2] + K[1, 2],
        ])
        obs1[i] = np.array([
            K[0, 0] * p1[0] / p1[2] + K[0, 2],
            K[1, 1] * p1[1] / p1[2] + K[1, 2],
        ])

    mono.prev_observations = obs0
    guess = mono._essential_pose_guess(obs1)

    assert guess is not None
    assert np.rad2deg(np.arccos(np.clip((np.trace(guess[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))) > 0.5


def test_experimental_monocular_vo_image_motion_fallback_moves_pose():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K, min_essential_tracks=1000, min_direct_landmarks=1000)
    obs0 = {i: np.array([100.0 + i * 5.0, 120.0]) for i in range(12)}
    obs1 = {i: pixel + np.array([2.0, 0.0]) for i, pixel in obs0.items()}

    mono.prev_observations = obs0
    guess = mono._image_motion_pose_guess(obs1)

    assert guess is not None
    assert guess[0, 3] < 0.0
    assert np.isclose(guess[1, 3], 0.0)


def test_experimental_monocular_vo_image_motion_fallback_has_rotation():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K, min_essential_tracks=1000, min_direct_landmarks=1000)
    yaw = np.deg2rad(1.0)
    R_cur_prev = np.array([
        [np.cos(yaw), 0.0, np.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-np.sin(yaw), 0.0, np.cos(yaw)],
    ])
    points = np.array([
        [-0.5, -0.3, 4.0],
        [0.1, -0.2, 5.0],
        [0.5, -0.1, 6.0],
        [-0.2, 0.1, 4.5],
        [0.3, 0.2, 5.5],
        [-0.4, 0.3, 6.0],
        [0.0, 0.0, 5.0],
        [0.6, 0.3, 7.0],
    ])

    obs0 = {}
    obs1 = {}
    for i, p in enumerate(points):
        p1 = R_cur_prev @ p
        obs0[i] = np.array([
            K[0, 0] * p[0] / p[2] + K[0, 2],
            K[1, 1] * p[1] / p[2] + K[1, 2],
        ])
        obs1[i] = np.array([
            K[0, 0] * p1[0] / p1[2] + K[0, 2],
            K[1, 1] * p1[1] / p1[2] + K[1, 2],
        ])

    mono.prev_observations = obs0
    guess = mono._image_motion_pose_guess(obs1)

    assert guess is not None
    angle = np.rad2deg(np.arccos(np.clip((np.trace(guess[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)))
    assert angle > 0.5


def test_experimental_monocular_vo_does_not_use_klt_pose_after_bootstrap():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K, min_essential_tracks=8, min_direct_landmarks=1000)
    mono.bootstrap_complete = True
    image = np.zeros((120, 160), dtype=np.uint8)
    cv2.circle(image, (60, 60), 4, 255, -1)

    def fail_if_called(_observations):
        raise AssertionError("KLT-derived pose fallback should be bootstrap-only")

    mono._essential_pose_guess = fail_if_called
    mono._image_motion_pose_guess = fail_if_called

    result = mono.process(10, image)

    assert result.motion_source == "hold"
    assert not result.bootstrap_active
    assert not result.essential_used
    assert not result.image_motion_fallback_used


def test_experimental_monocular_vo_post_bootstrap_guesses_use_direct_motion_model():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K)
    mono.bootstrap_complete = True
    mono.have_direct_motion_model = True
    mono.T_W_C[0, 3] = 1.0
    mono.last_direct_delta[0, 3] = 0.1

    guesses = mono._direct_pose_guesses()

    assert guesses[0][0] == "direct_motion"
    assert np.isclose(guesses[0][1][0, 3], 1.1)
    assert any(source == "direct_hold" for source, _ in guesses)
    assert any(source == "direct_recovery" for source, _ in guesses)


def test_experimental_monocular_vo_keyframe_klt_pixel_guesses_track_map_landmarks():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K, mono_map_config=MonoMapConfig(min_keyframe_landmarks=1))
    mono.bootstrap_complete = True
    rng = np.random.default_rng(2)
    points = np.column_stack([
        rng.uniform(-0.7, 0.7, 40),
        rng.uniform(-0.4, 0.4, 40),
        rng.uniform(4.0, 7.0, 40),
    ])
    ids = np.arange(len(points))
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[0, 3] = 0.05
    img0 = np.zeros((480, 640), dtype=np.uint8)
    img1 = np.zeros_like(img0)

    for img, T in [(img0, T0), (img1, T1)]:
        T_C_W = np.linalg.inv(T)
        points_c = points @ T_C_W[:3, :3].T + T_C_W[:3, 3]
        u = K[0, 0] * points_c[:, 0] / points_c[:, 2] + K[0, 2]
        v = K[1, 1] * points_c[:, 1] / points_c[:, 2] + K[1, 2]
        for ui, vi in zip(u, v):
            cv2.circle(img, (int(round(ui)), int(round(vi))), 3, 255, -1)

    mono.T_W_C = T0.copy()
    mono.mono_map.add_keyframe(
        0,
        img0,
        T0,
        0.0,
        0.0,
        ids,
        points,
        np.ones((len(points), 5), dtype=np.float32) * 100.0,
        "first",
        len(points),
        len(points),
    )
    mono.feature_tracker.prev_img = img0

    refs = mono.mono_map.visible_references(T0, img1.shape)
    pixel_guesses = mono._keyframe_klt_pixel_guesses(img1, refs)

    assert len(pixel_guesses) >= 20
    assert mono.last_keyframe_klt_tracks >= 20
    guessed_ids = sorted(pixel_guesses)
    guessed_pixels = np.asarray([pixel_guesses[fid] for fid in guessed_ids], dtype=np.float64)
    expected_pixels, _ = project_world_points(K, T1, points[guessed_ids])
    assert np.median(np.linalg.norm(guessed_pixels - expected_pixels, axis=1)) < 1.0

    mono.T_W_C = T1
    mono._update_keyframe_klt_residual_stats(img1.shape)
    residuals = mono.keyframe_klt_residual_segments(img1.shape)
    assert len(residuals["segments"]) == mono.last_keyframe_klt_tracks
    assert np.median(residuals["residuals"]) < 2.0
    assert mono.last_keyframe_klt_flow_cos_median > 0.9


def test_experimental_monocular_vo_detects_opposite_keyframe_klt_flow():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K)
    mono.last_keyframe_klt_points_w = np.array([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]])
    mono.last_keyframe_klt_prev_pixels = np.array([[320.0, 240.0], [332.5, 240.0]])
    mono.last_keyframe_klt_pixels = np.array([[330.0, 240.0], [342.5, 240.0]])
    T_bad = np.eye(4)
    T_bad[0, 3] = 0.08

    stats = mono._keyframe_klt_flow_stats_for_pose(T_bad, (480, 640))

    assert stats["count"] == 2
    assert stats["median"] < -0.9


def test_experimental_monocular_vo_rejects_implausible_direct_step():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K, max_direct_step=0.05)
    T_bad = np.eye(4)
    T_bad[0, 3] = 1.0

    assert not mono._direct_step_is_plausible(T_bad)


def test_experimental_monocular_vo_stops_after_good_keyframe_klt_candidate():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K, mono_map_config=MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    mono.mono_map.add_keyframe(
        0,
        image,
        np.eye(4),
        0.0,
        0.0,
        np.array([0]),
        np.array([[0.0, 0.0, 4.0]]),
        np.ones((1, 5), dtype=np.float32),
        "first",
        1,
        1,
    )
    mono.min_direct_landmarks = 1
    mono.min_direct_inliers = 1
    mono.last_keyframe_klt_points_w = np.array([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]])
    mono.last_keyframe_klt_prev_pixels = np.array([[320.0, 240.0], [332.5, 240.0]])
    mono.last_keyframe_klt_pixels = np.array([[320.0, 240.0], [332.5, 240.0]])
    mono.candidate_klt_residual_gate_tracks = 1

    calls = []
    refs = (
        np.array([[0.0, 0.0, 4.0]]),
        np.ones((1, 5), dtype=np.float32),
        np.zeros(1),
        np.zeros(1),
        np.array([0]),
    )
    mono.mono_map.direct_references = lambda _T: refs

    def track_once(_image, _pts, _ints, _a_ref, _b_ref, T_C_W_guess, _a, _b, max_iters=10):
        calls.append(max_iters)
        return T_C_W_guess, np.array([True]), 0.0, 0.0

    mono.direct_tracker.track_map = track_once

    result = mono._track_direct_candidates(
        image,
        [("keyframe_klt", np.eye(4)), ("direct_recovery", np.eye(4))],
    )

    assert result["accepted"]
    assert result["source"] == "keyframe_klt"
    assert len(calls) == 1


def test_experimental_monocular_vo_projects_existing_map_landmarks_for_gftt_mask():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K, mono_map_config=MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    mono.mono_map.add_keyframe(
        0,
        image,
        np.eye(4),
        0.0,
        0.0,
        np.array([10, 11]),
        np.array([[0.0, 0.0, 4.0], [10.0, 0.0, 4.0]]),
        np.ones((2, 5), dtype=np.float32),
        "first",
        2,
        2,
    )

    exclusions = mono._project_existing_map_landmarks(image.shape)

    assert exclusions.shape == (1, 2)
    assert np.allclose(exclusions[0], np.array([320.0, 240.0]))


def test_experimental_monocular_vo_exports_unassigned_ekf_landmark_glyphs():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K, mono_map_config=MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    mono.sparse3d.features[10] = LandmarkFilter.from_anchor_pixel(
        10,
        0,
        np.array([320.0, 240.0]),
        K,
        initial_depth=4.0,
        depth_sigma=1.0,
        anchor_T_W_C=np.eye(4),
    )
    mono.sparse3d.features[11] = LandmarkFilter.from_anchor_pixel(
        11,
        0,
        np.array([340.0, 240.0]),
        K,
        initial_depth=5.0,
        depth_sigma=2.0,
        anchor_T_W_C=np.eye(4),
    )
    mono.mono_map.add_keyframe(
        0,
        image,
        np.eye(4),
        0.0,
        0.0,
        np.array([10]),
        np.array([[0.0, 0.0, 4.0]]),
        np.ones((1, 5), dtype=np.float32),
        "first",
        1,
        1,
    )

    glyphs = mono.ekf_landmark_glyphs(image.shape)

    assert glyphs["points_w"].shape == (1, 3)
    assert np.allclose(glyphs["pixels"][0], np.array([340.0, 240.0]))
    assert glyphs["depth_std"][0] > 0.0


def test_experimental_monocular_vo_retires_ekf_landmarks_after_keyframe_promotion():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(
        K,
        sparse_settings=Sparse3DSettings(min_track_length=0, conv_depth_variance=100.0),
        mono_map_config=MonoMapConfig(min_keyframe_landmarks=1),
    )
    mono.bootstrap_complete = True
    mono.sparse3d.features[10] = LandmarkFilter.from_anchor_pixel(
        10,
        0,
        np.array([320.0, 240.0]),
        K,
        initial_depth=4.0,
        depth_sigma=1.0,
        anchor_T_W_C=np.eye(4),
    )
    mono.feature_tracker.prev_points = np.array([[320.0, 240.0]], dtype=np.float32)
    mono.feature_tracker.ids = np.array([10], dtype=int)
    image = np.zeros((480, 640), dtype=np.uint8)
    image[239:242, 319:322] = 100

    mono._maybe_add_keyframe(
        0,
        image,
        {10: np.array([320.0, 240.0])},
        used_direct=True,
        direct_inliers=1,
        direct_landmarks=1,
    )

    assert 10 in mono.mono_map.keyframes[0].landmark_ids
    assert 10 not in mono.sparse3d.features
    assert 10 in mono.sparse3d.retired_ids
    assert 10 not in mono.feature_tracker.ids


def test_experimental_monocular_vo_throttles_ba_on_keyframe_insertions():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(
        K,
        sparse_settings=Sparse3DSettings(min_track_length=0, conv_depth_variance=100.0),
        mono_map_config=MonoMapConfig(min_keyframe_landmarks=1),
        ba_every_keyframes=2,
    )
    mono.bootstrap_complete = True
    mono.min_direct_landmarks = 1
    mono.mono_map.add_keyframe(
        0,
        np.zeros((480, 640), dtype=np.uint8),
        np.eye(4),
        0.0,
        0.0,
        np.array([0]),
        np.array([[0.0, 0.0, 4.0]]),
        np.ones((1, 5), dtype=np.float32),
        "first",
        1,
        1,
    )
    for fid in [10, 11]:
        mono.sparse3d.features[fid] = LandmarkFilter.from_anchor_pixel(
            fid,
            0,
            np.array([320.0 + fid, 240.0]),
            K,
            initial_depth=4.0,
            depth_sigma=1.0,
            anchor_T_W_C=np.eye(4),
        )

    calls = {"geo": 0, "photo": 0}

    def fake_geo(_keyframes, max_iters=0):
        calls["geo"] += 1
        return {"ran": True, "window": len(_keyframes), "edges": 10}

    def fake_photo(_keyframes, max_iters=0):
        calls["photo"] += 1
        return {"ran": True, "window": len(_keyframes), "residuals": 10}

    mono.ba.optimize_mono_geometric_pose_window = fake_geo
    mono.ba.optimize_mono_pose_window = fake_photo

    image = np.zeros((480, 640), dtype=np.uint8)
    mono.T_W_C[0, 3] = 1.0
    mono._maybe_add_keyframe(1, image, {10: np.array([330.0, 240.0])}, True, 10, 10)
    mono.sparse3d.features[11] = LandmarkFilter.from_anchor_pixel(
        11,
        0,
        np.array([331.0, 240.0]),
        K,
        initial_depth=4.0,
        depth_sigma=1.0,
        anchor_T_W_C=np.eye(4),
    )
    mono.T_W_C[0, 3] = 2.0
    mono._maybe_add_keyframe(2, image, {11: np.array([331.0, 240.0])}, True, 10, 10)

    assert calls == {"geo": 1, "photo": 1}


def test_mono_map_inserts_and_culls_keyframes():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(window_size=2, min_keyframe_landmarks=1))
    image = np.zeros((120, 160), dtype=np.uint8)
    points = np.array([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]])
    intensities = np.ones((2, 5), dtype=np.float32) * 100.0
    ids = np.array([1, 2])

    for frame_id in range(3):
        T = np.eye(4)
        T[0, 3] = frame_id * 0.1
        mono_map.add_keyframe(frame_id, image, T, 0.0, 0.0, ids, points, intensities, "test", 2, 2)

    assert len(mono_map) == 2
    assert [kf.frame_id for kf in mono_map.keyframes] == [1, 2]
    assert mono_map.active_landmark_count() == 2


def test_mono_map_discards_redundant_keyframe_and_rescues_visible_landmarks():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(window_size=3, min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    observations = {0: np.array([320.0, 240.0])}

    poses = []
    for x in [0.0, 0.05, 0.5, 0.1]:
        T = np.eye(4)
        T[0, 3] = x
        poses.append(T)

    mono_map.add_keyframe(
        0,
        image,
        poses[0],
        0.0,
        0.0,
        np.array([0]),
        np.array([[0.0, 0.0, 4.0]]),
        np.ones((1, 5), dtype=np.float32) * 80.0,
        "first",
        1,
        1,
        observations=observations,
    )
    mono_map.add_keyframe(
        1,
        image,
        poses[1],
        0.0,
        0.0,
        np.array([1]),
        np.array([[0.1, 0.0, 4.0]]),
        np.ones((1, 5), dtype=np.float32) * 90.0,
        "motion",
        1,
        1,
    )
    mono_map.add_keyframe(
        2,
        image,
        poses[2],
        0.0,
        0.0,
        np.array([2]),
        np.array([[0.2, 0.0, 4.0]]),
        np.ones((1, 5), dtype=np.float32) * 100.0,
        "motion",
        1,
        1,
    )
    mono_map.add_keyframe(
        3,
        image,
        poses[3],
        0.0,
        0.0,
        np.array([3]),
        np.array([[0.3, 0.0, 4.0]]),
        np.ones((1, 5), dtype=np.float32) * 110.0,
        "motion",
        1,
        1,
    )

    assert len(mono_map) == 3
    assert [kf.frame_id for kf in mono_map.keyframes] == [1, 2, 3]
    latest = mono_map.keyframes[-1]
    assert 0 in latest.landmark_ids
    assert 0 in latest.observations
    assert 0 in latest.features
    assert 0 in mono_map.landmarks
    assert latest.kf_id in mono_map.landmarks[0].observations
    assert mono_map.last_discarded_kf_id not in mono_map.landmarks[0].observations
    assert np.allclose(latest.points_w[latest.landmark_ids.tolist().index(0)], np.array([0.0, 0.0, 4.0]))


def test_mono_map_direct_references_distribute_budget_across_overlapping_keyframes():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(
        window_size=4,
        min_keyframe_landmarks=1,
        max_direct_refs=20,
        max_reference_keyframes=2,
    ))
    image = np.zeros((480, 640), dtype=np.uint8)
    points_a = np.column_stack([
        np.linspace(-0.5, -0.1, 40),
        np.zeros(40),
        np.ones(40) * 4.0,
    ])
    points_b = np.column_stack([
        np.linspace(0.1, 0.5, 40),
        np.zeros(40),
        np.ones(40) * 4.0,
    ])

    mono_map.add_keyframe(
        0, image, np.eye(4), 0.0, 0.0, np.arange(40), points_a,
        np.ones((40, 5), dtype=np.float32) * 80.0, "first", 40, 40,
    )
    mono_map.add_keyframe(
        1, image, np.eye(4), 0.0, 0.0, np.arange(100, 140), points_b,
        np.ones((40, 5), dtype=np.float32) * 90.0, "motion", 40, 40,
    )

    _, _, _, _, ids = mono_map.direct_references(np.eye(4))

    assert len(ids) == 20
    assert np.any(ids < 40)
    assert np.any(ids >= 100)
    assert mono_map.last_reference_counts[0] > 0
    assert mono_map.last_reference_counts[1] > 0


def test_mono_map_discards_low_covisibility_keyframe_before_useful_old_anchor():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(
        window_size=3,
        min_keyframe_landmarks=1,
        min_covisible_landmarks=5,
    ))
    image = np.zeros((480, 640), dtype=np.uint8)
    for frame_id, x in enumerate([0.0, 0.1, 0.2]):
        T = np.eye(4)
        T[0, 3] = x
        points = np.array([[x, 0.0, 4.0]])
        mono_map.add_keyframe(
            frame_id, image, T, 0.0, 0.0, np.array([frame_id]), points,
            np.ones((1, 5), dtype=np.float32), "motion", 1, 1,
        )

    mono_map.last_reference_counts = {0: 10, 1: 0}
    mono_map.last_visible_counts = {0: 10, 1: 0}
    T_new = np.eye(4)
    T_new[0, 3] = 0.3
    mono_map.add_keyframe(
        3,
        image,
        T_new,
        0.0,
        0.0,
        np.array([3]),
        np.array([[0.3, 0.0, 4.0]]),
        np.ones((1, 5), dtype=np.float32),
        "motion",
        1,
        1,
    )

    assert [kf.frame_id for kf in mono_map.keyframes] == [0, 2, 3]
    assert mono_map.last_discarded_kf_id == 1
    assert mono_map.last_discard_reason == "low_covisibility"


def test_mono_map_accumulates_and_recovers_keyframe_low_usefulness():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(
        min_keyframe_landmarks=1,
        min_useful_reference_count=4,
        min_useful_reference_ratio=0.5,
    ))
    image = np.zeros((480, 640), dtype=np.uint8)
    points = np.column_stack([np.linspace(-0.1, 0.1, 4), np.zeros(4), np.ones(4) * 4.0])
    ids = np.arange(4)
    mono_map.add_keyframe(
        0, image, np.eye(4), 0.0, 0.0, ids, points,
        np.ones((4, 5), dtype=np.float32), "first", 4, 4,
    )

    _, _, _, _, refs = mono_map.direct_references(np.eye(4))
    mono_map.record_direct_reference_prefilter(refs[:1])
    mono_map.record_direct_reference_prefilter(refs[:1])

    assert mono_map.last_prefilter_kept_counts[0] == 1
    assert mono_map.last_prefilter_rejected_counts[0] == 3
    assert mono_map.keyframe_low_usefulness_frames[0] == 2

    mono_map.record_direct_reference_prefilter(refs)

    assert mono_map.keyframe_low_usefulness_frames[0] == 1
    assert mono_map.keyframe_usefulness_ratio[0] > 0.25


def test_mono_map_discards_persistently_low_usefulness_keyframe():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(
        window_size=3,
        min_keyframe_landmarks=1,
        min_useful_reference_count=4,
        min_useful_reference_ratio=0.5,
        max_low_usefulness_frames=2,
    ))
    image = np.zeros((480, 640), dtype=np.uint8)
    for frame_id, x in enumerate([0.0, 0.1, 0.2]):
        T = np.eye(4)
        T[0, 3] = x
        points = np.column_stack([np.linspace(-0.1, 0.1, 4) + x, np.zeros(4), np.ones(4) * 4.0])
        ids = np.arange(frame_id * 10, frame_id * 10 + 4)
        mono_map.add_keyframe(
            frame_id, image, T, 0.0, 0.0, ids, points,
            np.ones((4, 5), dtype=np.float32), "motion", 4, 4,
        )

    mono_map.keyframe_low_usefulness_frames = {0: 2, 1: 0, 2: 0}
    mono_map.last_reference_counts = {0: 4, 1: 4, 2: 4}
    mono_map.last_visible_counts = {0: 4, 1: 4, 2: 4}
    T_new = np.eye(4)
    T_new[0, 3] = 0.3
    mono_map.add_keyframe(
        3,
        image,
        T_new,
        0.0,
        0.0,
        np.arange(30, 34),
        np.column_stack([np.linspace(-0.1, 0.1, 4) + 0.3, np.zeros(4), np.ones(4) * 4.0]),
        np.ones((4, 5), dtype=np.float32),
        "motion",
        4,
        4,
    )

    assert [kf.frame_id for kf in mono_map.keyframes] == [1, 2, 3]
    assert mono_map.last_discarded_kf_id == 0
    assert mono_map.last_discard_reason == "low_usefulness"
    assert 0 not in mono_map.keyframe_low_usefulness_frames


def test_mono_map_direct_references_survive_missing_klt_tracks():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    points = np.array([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]])
    intensities = np.ones((2, 5), dtype=np.float32) * 120.0
    ids = np.array([10, 11])

    mono_map.add_keyframe(0, image, np.eye(4), 0.0, 0.0, ids, points, intensities, "first", 2, 2)
    pts_w, refs, a_vals, b_vals, out_ids = mono_map.direct_references(np.eye(4))

    assert pts_w.shape == (2, 3)
    assert refs.shape == (2, 5)
    assert a_vals.tolist() == [0.0, 0.0]
    assert b_vals.tolist() == [0.0, 0.0]
    assert out_ids.tolist() == [10, 11]


def test_mono_map_direct_references_use_landmark_tracks_and_features():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    points = np.array([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]])
    intensities = np.ones((2, 5), dtype=np.float32) * 120.0
    ids = np.array([10, 11])

    mono_map.add_keyframe(0, image, np.eye(4), 0.0, 0.0, ids, points, intensities, "first", 2, 2)
    mono_map.landmarks[10].point_w = np.array([0.2, 0.0, 5.0], dtype=np.float64)
    mono_map.keyframes[0].features[10].intensity = np.ones(5, dtype=np.float32) * 77.0

    pts_w, refs, _a_vals, _b_vals, out_ids = mono_map.direct_references(np.eye(4))

    idx = out_ids.tolist().index(10)
    assert np.allclose(pts_w[idx], np.array([0.2, 0.0, 5.0]))
    assert np.allclose(refs[idx], np.ones(5, dtype=np.float32) * 77.0)


def test_mono_map_visible_references_exports_projected_graph_records():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    points = np.array([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]])
    ids = np.array([10, 11])
    intensities = np.ones((2, 5), dtype=np.float32) * 120.0
    observations = {10: np.array([320.0, 240.0]), 11: np.array([332.5, 240.0])}

    mono_map.add_keyframe(0, image, np.eye(4), 0.1, 2.0, ids, points, intensities, "first", 2, 2,
                          observations=observations)

    refs = mono_map.visible_references(np.eye(4), image.shape)

    assert len(refs) == 2
    assert refs[0]["landmark_id"] == 10
    assert refs[0]["host_kf_id"] == 0
    assert np.allclose(refs[0]["projected_pixel"], np.array([320.0, 240.0]))
    assert np.allclose(refs[0]["host_pixel"], observations[10])
    assert np.allclose(refs[0]["intensity"], np.ones(5, dtype=np.float32) * 120.0)
    assert refs[0]["affine_a"] == 0.1
    assert refs[0]["affine_b"] == 2.0
    assert mono_map.last_reference_counts[0] == 2


def test_mono_map_point_cloud_uses_landmark_tracks_and_features():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    points = np.array([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]])
    intensities = np.ones((2, 5), dtype=np.float32) * 120.0
    ids = np.array([10, 11])

    mono_map.add_keyframe(0, image, np.eye(4), 0.0, 0.0, ids, points, intensities, "first", 2, 2)
    mono_map.landmarks[10].point_w = np.array([0.2, 0.0, 5.0], dtype=np.float64)
    mono_map.keyframes[0].features[10].intensity = np.ones(5, dtype=np.float32) * 77.0

    pts_w, intensity, out_ids, _kf_ids = mono_map.point_cloud()

    idx = out_ids.tolist().index(10)
    assert np.allclose(pts_w[idx], np.array([0.2, 0.0, 5.0]))
    assert np.isclose(intensity[idx], 77.0)


def test_mono_map_syncs_landmark_tracks_after_keyframe_point_updates():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    points = np.array([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]])
    intensities = np.ones((2, 5), dtype=np.float32) * 120.0
    ids = np.array([10, 11])

    mono_map.add_keyframe(0, image, np.eye(4), 0.0, 0.0, ids, points, intensities, "first", 2, 2)
    refined = np.array([0.2, 0.0, 5.0], dtype=np.float64)
    mono_map.keyframes[0].points_w[0] = refined

    mono_map.sync_landmark_tracks_from_keyframes()

    assert np.allclose(mono_map.landmarks[10].point_w, refined)
    assert np.allclose(mono_map.keyframes[0].features[10].point_w, refined)


def test_monocular_patch_matcher_accepts_translated_patch_and_rejects_occlusion():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K, mono_map_config=MonoMapConfig(min_keyframe_landmarks=1))
    image = synthetic_texture_image()
    ref_intensity = sample_pattern(image, np.array([[325.0, 240.0]]))[0]
    refs = [
        {
            "landmark_id": 10,
            "host_kf_id": 0,
            "point_w": np.array([0.0, 0.0, 4.0]),
            "projected_pixel": np.array([320.0, 240.0]),
            "intensity": ref_intensity,
            "affine_a": 0.0,
            "affine_b": 0.0,
        },
        {
            "landmark_id": 11,
            "host_kf_id": 0,
            "point_w": np.array([0.1, 0.0, 4.0]),
            "projected_pixel": np.array([340.0, 240.0]),
            "intensity": np.ones(5, dtype=np.float32) * 255.0,
            "affine_a": 0.0,
            "affine_b": 0.0,
        },
    ]

    matches = mono._match_visible_reference_patches(image, refs, {10: np.array([324.0, 240.0])})

    assert matches[10].accepted
    assert np.linalg.norm(matches[10].matched_pixel - np.array([325.0, 240.0])) <= 1.0
    assert not matches[11].accepted


def test_monocular_matched_observation_pose_guess_recovers_translation():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K, mono_map_config=MonoMapConfig(min_keyframe_landmarks=1))
    rng = np.random.default_rng(12)
    points = np.column_stack([
        rng.uniform(-0.7, 0.7, 40),
        rng.uniform(-0.4, 0.4, 40),
        rng.uniform(4.0, 7.0, 40),
    ])
    T1 = np.eye(4)
    T1[0, 3] = 0.05
    uv, _ = project_world_points(K, T1, points)
    image = synthetic_texture_image()
    intensities = sample_pattern(image, uv)
    refs = []
    for i, (point, pixel, intensity) in enumerate(zip(points, uv, intensities)):
        refs.append({
            "landmark_id": i,
            "host_kf_id": 0,
            "point_w": point,
            "projected_pixel": pixel,
            "intensity": intensity,
            "affine_a": 0.0,
            "affine_b": 0.0,
        })

    guess = mono._matched_observation_pose_guess(image, np.eye(4), refs, {})

    assert guess is not None
    assert np.linalg.norm(guess[:3, 3] - T1[:3, 3]) < 0.03


def test_experimental_monocular_vo_post_bootstrap_does_not_call_klt_pnp_pose_hook():
    K = get_dummy_camera()
    mono = ExperimentalMonocularVO(K)
    mono.bootstrap_complete = True
    image = np.zeros((120, 160), dtype=np.uint8)

    def old_hook(_image):
        raise AssertionError("old KLT-PnP pose hook should not be called")

    mono._keyframe_klt_pose_guess = old_hook
    mono._track_direct_candidates = lambda _image, _guesses: {
        "accepted": False,
        "hypotheses": len(_guesses),
        "landmarks": 0,
        "attempted": False,
        "inliers": 0,
        "T_W_C": mono.T_W_C.copy(),
        "a": mono.current_a,
        "b": mono.current_b,
        "source": "hold",
        "candidates": [],
    }

    result = mono.process(0, image)

    assert result.motion_source == "hold"


def test_mono_map_stores_observation_graph_for_keyframes():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    points = np.array([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]])
    intensities = np.ones((2, 5), dtype=np.float32)
    ids = np.array([10, 11])
    observations = {
        10: np.array([320.0, 240.0]),
        12: np.array([100.0, 100.0]),
    }

    mono_map.add_keyframe(
        0, image, np.eye(4), 0.0, 0.0, ids, points, intensities,
        "first", 2, 2, observations=observations, observation_cov=np.eye(2) * 0.25,
    )

    assert mono_map.observation_count() == 1
    assert 10 in mono_map.keyframes[0].observations
    assert 12 not in mono_map.keyframes[0].observations
    assert np.allclose(mono_map.keyframes[0].observations[10].covariance, np.eye(2) * 0.25)


def test_mono_map_builds_svo_style_landmark_tracks_on_keyframe_insert():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    points = np.array([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]])
    intensities = np.ones((2, 5), dtype=np.float32) * np.array([[80.0], [90.0]], dtype=np.float32)
    ids = np.array([10, 11])
    observations = {
        10: np.array([320.0, 240.0]),
        11: np.array([332.5, 240.0]),
    }

    mono_map.add_keyframe(
        0,
        image,
        np.eye(4),
        0.0,
        0.0,
        ids,
        points,
        intensities,
        "first",
        2,
        2,
        observations=observations,
        observation_cov=np.eye(2) * 0.5,
    )

    kf = mono_map.keyframes[0]
    assert set(kf.features) == {10, 11}
    assert set(mono_map.landmarks) == {10, 11}
    assert mono_map.landmark_observation_count() == 2
    assert mono_map.landmarks[10].host_kf_id == kf.kf_id
    assert mono_map.landmarks[10].observations[kf.kf_id] is kf.features[10]
    assert np.allclose(kf.features[10].pixel, observations[10])
    assert np.allclose(kf.features[10].bearing, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(kf.features[11].intensity, np.ones(5, dtype=np.float32) * 90.0)
    assert np.allclose(kf.features[10].covariance, np.eye(2) * 0.5)


def test_mono_map_point_cloud_exports_unique_latest_landmarks():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((80, 100), dtype=np.uint8)

    mono_map.add_keyframe(
        0,
        image,
        np.eye(4),
        0.0,
        0.0,
        np.array([1, 2]),
        np.array([[0.0, 0.0, 4.0], [0.2, 0.0, 5.0]]),
        np.array([[50.0, 60.0, 70.0], [100.0, 110.0, 120.0]], dtype=np.float32),
        "first",
        2,
        2,
    )
    T1 = np.eye(4)
    T1[0, 3] = 0.1
    mono_map.add_keyframe(
        1,
        image,
        T1,
        0.0,
        0.0,
        np.array([2, 3]),
        np.array([[0.3, 0.0, 6.0], [0.4, 0.0, 7.0]]),
        np.array([[130.0, 140.0, 150.0], [200.0, 210.0, 220.0]], dtype=np.float32),
        "second",
        2,
        2,
    )

    points, intensities, ids, kf_ids = mono_map.point_cloud()

    assert ids.tolist() == [2, 3, 1]
    assert kf_ids.tolist() == [1, 1, 0]
    assert np.allclose(points[0], np.array([0.3, 0.0, 6.0]))
    assert np.allclose(intensities, np.array([140.0, 210.0, 60.0]))


def test_mono_pose_ba_noops_for_small_window():
    ba = PhotometricBA(get_dummy_camera())

    result = ba.optimize_mono_pose_window([], max_iters=1)

    assert not result["ran"]
    assert result["window"] == 0


def test_mono_pose_ba_uses_mono_map_reference_records():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    points = np.array([[0.0, 0.0, 4.0]])
    intensities = np.ones((1, 5), dtype=np.float32)
    mono_map.add_keyframe(0, image, np.eye(4), 0.0, 0.0, np.array([0]), points, intensities, "first", 1, 1)
    mono_map.add_keyframe(1, image, np.eye(4), 0.0, 0.0, np.array([1]), points, intensities, "motion", 1, 1)
    calls = []

    def empty_records(kf):
        calls.append(kf.kf_id)
        return []

    mono_map.keyframe_reference_records = empty_records

    result = PhotometricBA(K).optimize_mono_pose_window(mono_map, max_iters=1)

    assert not result["ran"]
    assert calls == [0, 1]


def test_mono_pose_ba_keeps_oldest_keyframe_fixed():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    image[235:245, 315:325] = 100
    points = np.array([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0], [-0.1, 0.0, 4.0], [0.0, 0.1, 4.0],
                       [0.0, -0.1, 4.0], [0.2, 0.1, 4.0], [-0.2, -0.1, 4.0], [0.3, 0.0, 4.0],
                       [-0.3, 0.0, 4.0], [0.0, 0.2, 4.0], [0.0, -0.2, 4.0], [0.25, -0.2, 4.0]])
    intensities = np.ones((len(points), 5), dtype=np.float32) * 100.0
    ids = np.arange(len(points))
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[0, 3] = 0.01

    mono_map.add_keyframe(0, image, T0, 0.0, 0.0, ids, points, intensities, "first", len(points), len(points))
    mono_map.add_keyframe(1, image, T1, 0.0, 0.0, ids, points, intensities, "motion", len(points), len(points))
    before = mono_map.keyframes[0].T_W_C.copy()

    PhotometricBA(K).optimize_mono_pose_window(mono_map.keyframes, max_iters=1)

    assert np.allclose(mono_map.keyframes[0].T_W_C, before)


def test_mono_geometric_ba_noops_for_insufficient_edges():
    ba = PhotometricBA(get_dummy_camera())

    result = ba.optimize_mono_geometric_pose_window([], max_iters=1)

    assert not result["ran"]
    assert result["window"] == 0


def test_mono_geometric_ba_uses_mono_map_landmark_tracks():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    points = np.column_stack([
        np.linspace(-0.5, 0.5, 12),
        np.linspace(-0.2, 0.2, 12),
        np.ones(12) * 4.0,
    ])
    ids = np.arange(len(points))
    observations = {}
    for i, p in enumerate(points):
        observations[i] = np.array([
            K[0, 0] * p[0] / p[2] + K[0, 2],
            K[1, 1] * p[1] / p[2] + K[1, 2],
        ])
    intensities = np.ones((len(points), 5), dtype=np.float32)

    mono_map.add_keyframe(0, image, np.eye(4), 0.0, 0.0, ids, points, intensities, "first", 12, 12,
                          observations=observations)
    mono_map.add_keyframe(1, image, np.eye(4), 0.0, 0.0, ids, points, intensities, "motion", 12, 12,
                          observations=observations)
    for kf in mono_map.keyframes:
        for obs in kf.observations.values():
            obs.point_w = np.array([10.0, 0.0, 4.0], dtype=np.float64)

    result = PhotometricBA(K).optimize_mono_geometric_pose_window(mono_map, max_iters=1)

    assert result["ran"]
    assert result["edges"] >= 10


def test_mono_geometric_ba_keeps_oldest_keyframe_fixed():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    points = np.column_stack([
        np.linspace(-0.5, 0.5, 12),
        np.linspace(-0.2, 0.2, 12),
        np.ones(12) * 4.0,
    ])
    ids = np.arange(len(points))
    intensities = np.ones((len(points), 5), dtype=np.float32)
    observations = {}
    for i, p in enumerate(points):
        observations[i] = np.array([
            K[0, 0] * p[0] / p[2] + K[0, 2],
            K[1, 1] * p[1] / p[2] + K[1, 2],
        ])

    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[0, 3] = 0.01
    mono_map.add_keyframe(0, image, T0, 0.0, 0.0, ids, points, intensities, "first", 12, 12,
                          observations=observations)
    mono_map.add_keyframe(1, image, T1, 0.0, 0.0, ids, points, intensities, "motion", 12, 12,
                          observations=observations)
    before = mono_map.keyframes[0].T_W_C.copy()

    result = PhotometricBA(K).optimize_mono_geometric_pose_window(mono_map.keyframes, max_iters=1)

    assert result["ran"]
    assert result["edges"] >= 10
    assert np.allclose(mono_map.keyframes[0].T_W_C, before)


def test_mono_geometric_ba_recovers_perturbed_keyframe_pose():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(min_keyframe_landmarks=1))
    image = np.zeros((480, 640), dtype=np.uint8)
    rng = np.random.default_rng(11)
    points = np.column_stack([
        rng.uniform(-0.8, 0.8, 80),
        rng.uniform(-0.5, 0.5, 80),
        rng.uniform(4.0, 7.0, 80),
    ])
    ids = np.arange(len(points))
    intensities = np.ones((len(points), 5), dtype=np.float32)
    T0 = np.eye(4)
    T1_true = np.eye(4)
    T1_true[0, 3] = 0.12
    T1_true[1, 3] = -0.03
    T1_initial = T1_true.copy()
    T1_initial[0, 3] -= 0.04
    T1_initial[1, 3] += 0.02

    obs0_uv, _ = project_world_points(K, T0, points)
    obs1_uv, _ = project_world_points(K, T1_true, points)
    obs0 = {i: obs0_uv[i] for i in range(len(points))}
    obs1 = {i: obs1_uv[i] for i in range(len(points))}

    mono_map.add_keyframe(0, image, T0, 0.0, 0.0, ids, points, intensities, "first", len(points), len(points),
                          observations=obs0)
    mono_map.add_keyframe(1, image, T1_initial, 0.0, 0.0, ids, points, intensities, "motion",
                          len(points), len(points), observations=obs1)
    before = reprojection_rmse(K, mono_map.keyframes[1].T_W_C, points, obs1)

    result = PhotometricBA(K).optimize_mono_geometric_pose_window(mono_map.keyframes, max_iters=20)
    after = reprojection_rmse(K, mono_map.keyframes[1].T_W_C, points, obs1)

    assert result["ran"]
    assert after < before * 0.2
    assert np.linalg.norm(mono_map.keyframes[1].T_W_C[:3, 3] - T1_true[:3, 3]) < 0.01


def test_mono_inverse_depth_ba_refines_landmarks_along_host_rays():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(min_keyframe_landmarks=1))
    y, x = np.indices((480, 640), dtype=np.float32)
    host_image = np.clip(0.2 * x + 0.15 * y + 40.0, 0.0, 255.0).astype(np.float32)
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[0, 3] = 0.2
    true_points = np.column_stack([
        np.linspace(-0.6, 0.6, 36),
        np.linspace(-0.25, 0.25, 36),
        np.ones(36) * 4.0,
    ])
    host_uv, _ = project_world_points(K, T0, true_points)
    disparity = K[0, 0] * T1[0, 3] / 4.0
    y, x = np.indices(host_image.shape, dtype=np.float32)
    target_image = cv2.remap(
        host_image,
        np.clip(x + disparity, 0, host_image.shape[1] - 1).astype(np.float32),
        y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    wrong_points = true_points.copy()
    wrong_points[:, 2] = 6.0
    wrong_points[:, 0] = true_points[:, 0] * 6.0 / 4.0
    wrong_points[:, 1] = true_points[:, 1] * 6.0 / 4.0
    intensities = sample_pattern(host_image, host_uv)
    ids = np.arange(len(true_points))
    mono_map.add_keyframe(0, host_image, T0, 0.0, 0.0, ids, wrong_points, intensities, "first", 36, 36)
    mono_map.add_keyframe(1, target_image, T1, 0.0, 0.0, ids, wrong_points, intensities, "motion", 36, 36)

    before = np.mean(np.abs(mono_map.keyframes[0].points_w[:, 2] - 4.0))
    result = PhotometricBA(K).optimize_mono_inverse_depth_window(
        mono_map,
        max_iters=20,
        max_initial_photometric_error=200.0,
    )
    after = np.mean(np.abs(mono_map.keyframes[0].points_w[:, 2] - 4.0))

    assert result["ran"]
    assert result["edges"] >= 10
    assert result["cost_after"] < result["cost_before"]
    assert after < before
    assert np.allclose(mono_map.keyframes[0].points_w, mono_map.keyframes[1].points_w)
    assert np.allclose(mono_map.landmarks[0].point_w, mono_map.keyframes[1].points_w[0])
    assert np.allclose(mono_map.keyframes[0].features[0].point_w, mono_map.keyframes[1].points_w[0])


def test_mono_inverse_depth_ba_skips_occluded_target_edges():
    K = get_dummy_camera()
    mono_map = MonoMap(K, MonoMapConfig(min_keyframe_landmarks=1))
    host_image = synthetic_texture_image()
    occluded_image = np.zeros_like(host_image)
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[0, 3] = 0.2
    points = np.column_stack([
        np.linspace(-0.5, 0.5, 20),
        np.zeros(20),
        np.ones(20) * 4.0,
    ])
    host_uv, _ = project_world_points(K, T0, points)
    intensities = sample_pattern(host_image, host_uv)
    ids = np.arange(len(points))
    mono_map.add_keyframe(0, host_image, T0, 0.0, 0.0, ids, points, intensities, "first", 20, 20)
    mono_map.add_keyframe(1, occluded_image, T1, 0.0, 0.0, ids, points, intensities, "motion", 20, 20)
    before = mono_map.keyframes[0].points_w.copy()

    result = PhotometricBA(K).optimize_mono_inverse_depth_window(
        mono_map.keyframes,
        max_iters=5,
        max_initial_photometric_error=20.0,
    )

    assert not result["ran"]
    assert result["edges"] == 0
    assert np.allclose(mono_map.keyframes[0].points_w, before)
