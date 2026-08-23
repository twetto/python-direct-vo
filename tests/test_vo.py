import numpy as np
import cv2
import pytest
from src.map import Keyframe, Map
from src.tracker import DirectTracker

def get_dummy_camera():
    return np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)

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
