import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent / "neuro-dataloaders"))
from neuro_dataloaders.euroc import EuRoCLoader

def build_stereo_rectification_maps(cam0, cam1):
    # T_body_camX transforms points from body to camera X
    # Relative transform from cam0 to cam1: T_c1_c0 = T_body_cam1 * inv(T_body_cam0)
    T_c1_c0 = cam1.T_body_cam @ np.linalg.inv(cam0.T_body_cam)
    R = T_c1_c0[:3, :3]
    T = T_c1_c0[:3, 3].reshape(3, 1)

    res = cam0.resolution
    
    # OpenCV Stereo Rectification
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cam0.K, cam0.dist_coeffs,
        cam1.K, cam1.dist_coeffs,
        res, R, T,
        alpha=0.0 # 0.0 means crop all black pixels
    )

    # Compute undistortion & rectification maps
    map1_x, map1_y = cv2.initUndistortRectifyMap(cam0.K, cam0.dist_coeffs, R1, P1, res, cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(cam1.K, cam1.dist_coeffs, R2, P2, res, cv2.CV_32FC1)

    # Return maps, focal length (from P1), and baseline (derived from P2)
    focal_length = P1[0, 0]
    # P2[0, 3] is (fx * -baseline)
    baseline = -P2[0, 3] / focal_length 
    
    cx_rect = P1[0, 2]
    cy_rect = P1[1, 2]

    return (map1_x, map1_y), (map2_x, map2_y), focal_length, baseline, R1, cx_rect, cy_rect

def compute_stereo_depth(img_left, img_right, f, b):
    window_size = 5
    min_disp = 0
    num_disp = 16 * 4

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 1 * window_size**2,
        P2=32 * 1 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
    
    depth = np.zeros_like(disparity)
    valid = disparity > 0
    depth[valid] = (f * b) / disparity[valid]
    
    return disparity, depth

if __name__ == "__main__":
    DATASET_PATH = "C:/Users/twetto/Downloads/machine_hall/machine_hall/MH_01_easy/MH_01_easy"
    
    # We load RAW images (undistort=False) because we will manually rectify them
    loader_left = EuRoCLoader(DATASET_PATH, cam_id=0, undistort=False)
    loader_right = EuRoCLoader(DATASET_PATH, cam_id=1, undistort=False)

    print("Building stereo rectification maps...")
    maps_left, maps_right, f, b = build_stereo_rectification_maps(loader_left.camera, loader_right.camera)
    print(f"Rectified Focal Length: {f:.2f}, Baseline: {b:.4f}m")

    for sample_left, sample_right in zip(loader_left, loader_right):
        if abs(sample_left.timestamp - sample_right.timestamp) > 0.01:
            continue

        # 1. Apply Stereo Rectification (Warping so epipolar lines are perfectly horizontal)
        img_left_rect = cv2.remap(sample_left.image, maps_left[0], maps_left[1], cv2.INTER_LINEAR)
        img_right_rect = cv2.remap(sample_right.image, maps_right[0], maps_right[1], cv2.INTER_LINEAR)

        # 2. Compute Depth
        disp, depth = compute_stereo_depth(img_left_rect, img_right_rect, f, b)

        # Visualization
        disp_vis = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        cv2.imshow("Left Rectified", img_left_rect)
        cv2.imshow("Disparity Map", disp_vis)
        
        if cv2.waitKey(30) == 27:
            break
