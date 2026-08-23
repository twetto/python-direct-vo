import sys
from pathlib import Path
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import argparse

# Hook up the dataloader
sys.path.append(str(Path(__file__).parent.parent / "neuro-dataloaders"))
from neuro_dataloaders.euroc import EuRoCLoader
from src.stereo import build_stereo_rectification_maps, compute_stereo_depth
from src.profiler import profiler
from src.tracker import DirectTracker
from src.map import Map
from src.optimizer import PhotometricBA
from src.diagnostics import VOMonitor

def umeyama_alignment(X, Y):
    """
    Estimates the Sim(3) transform from X to Y.
    Returns (s, R, t) such that Y \approx s * R @ X + t
    """
    if len(X) < 3:
        return 1.0, np.eye(3), np.zeros(3)
    
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean
    
    Sigma = X_centered.T @ Y_centered / len(X)
    U, D, Vt = np.linalg.svd(Sigma)
    
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
        
    R = Vt.T @ S @ U.T
    
    # Enforce strictly isometric SE(3) alignment (s = 1.0).
    # Since this is Stereo VO, the scale is inherently metric.
    # Allowing a scale factor here creates non-isometric scaling, 
    # which breaks Rerun Pinhole rendering.
    s = 1.0
        
    t = Y_mean - (R @ X_mean)
    return s, R, t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', help='Run without OpenCV GUI')
    args = parser.parse_args()

    # DATASET_PATH = "C:/Users/twetto/Downloads/machine_hall/machine_hall/MH_01_easy/MH_01_easy"
    DATASET_PATH = "C:/Users/twetto/Downloads/vicon_room1/vicon_room1/V1_01_easy"
    
    # Initialize Rerun GUI with a fresh app id to clear ghost blueprints
    print("Connecting to Rerun Viewer (Real-Time TCP Stream)...")
    rr.init("python_direct_vo", spawn=False)
    rr.connect_grpc()
    
    # Configure Rerun view coordinates (OpenCV camera convention is Z-forward, X-right, Y-down)
    try:
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    except Exception:
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)
        
    print("Setting up Rerun Blueprint for Third-Person View...")
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
                name="Camera View (Raw)",
                origin="world/estimated/camera/image"
            ),
            rrb.Spatial3DView(
                name="World (Third Person)",
                origin="world",
                eye_controls=rrb.EyeControls3D(
                    tracking_entity="world/view_camera"
                )
            ),
            column_shares=[1, 2]
        )
    )
    rr.send_blueprint(blueprint)
    
    # Log a dummy transparent pinhole to force Rerun to adopt full pose tracking
    rr.log(
        "world/view_camera", 
        rr.Pinhole(resolution=[640, 480], focal_length=[300, 300], image_plane_distance=0.0),
        static=True
    )
    
    print("Loading EuRoC Dataset...")
    loader_left = EuRoCLoader(DATASET_PATH, cam_id=0, undistort=False)
    loader_right = EuRoCLoader(DATASET_PATH, cam_id=1, undistort=False)

    print("Initializing Stereo Rectification...")
    maps_left, maps_right, f, b, R1, cx_rect, cy_rect = build_stereo_rectification_maps(loader_left.camera, loader_right.camera)
    
    # K_rect has focal length f and principal point from the RECTIFIED projection matrix!
    K_rect = np.array([
        [f, 0, cx_rect],
        [0, f, cy_rect],
        [0, 0, 1]
    ])
    
    from src.map import Map
    from src.optimizer import PhotometricBA
    from src.diagnostics import VOMonitor

    tracker = DirectTracker(K_rect)
    vo_map = Map(K_rect)
    ba = PhotometricBA(K_rect)
    monitor = VOMonitor()
    
    T_W_C = np.eye(4)
    current_a = 0.0
    current_b = 0.0
    prev_img_rect = None
    
    traj_positions = []
    traj_positions_gt = []
    align_X = []
    align_Y = []
    
    print("Starting Direct Visual Odometry with Rerun...")
    for frame_idx, (sample_left, sample_right) in enumerate(zip(loader_left, loader_right)):
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}...")
        rr.set_time("frame", sequence=frame_idx)
        
        if abs(sample_left.timestamp - sample_right.timestamp) > 0.01:
            continue
            
        # Get GT Unrectified Camera Pose
        T_W_C_gt_unrect = sample_left.T_world_body @ loader_left.camera.T_cam_body
        
        # Transform the GT pose into the Rectified Camera frame (R1 maps unrectified -> rectified)
        # T_world_rectified = T_world_unrectified @ R_unrectified_from_rectified
        T_unrect_rect = np.eye(4)
        T_unrect_rect[:3, :3] = R1.T
        T_W_C_gt = T_W_C_gt_unrect @ T_unrect_rect
        traj_positions_gt.append(T_W_C_gt[:3, 3])
        
        # Add GT SE(3) basis points for robust alignment
        align_Y.append(T_W_C_gt[:3, 3])
        align_Y.append(T_W_C_gt[:3, 3] + T_W_C_gt[:3, 0] * 0.1)
        align_Y.append(T_W_C_gt[:3, 3] + T_W_C_gt[:3, 1] * 0.1)
        align_Y.append(T_W_C_gt[:3, 3] + T_W_C_gt[:3, 2] * 0.1)
        
        # Log Ground Truth Trajectory & Camera
        rr.log("world/gt/trajectory", rr.LineStrips3D([traj_positions_gt], colors=[[0, 255, 255]]))
        rr.log("world/gt/camera", rr.Transform3D(translation=T_W_C_gt[:3, 3], mat3x3=T_W_C_gt[:3, :3]))

        profiler.start("0. Data & Rectification")
        img_left_rect = cv2.remap(sample_left.image, maps_left[0], maps_left[1], cv2.INTER_LINEAR)
        img_right_rect = cv2.remap(sample_right.image, maps_right[0], maps_right[1], cv2.INTER_LINEAR)
        _, depth = compute_stereo_depth(img_left_rect, img_right_rect, f, b)
        # --- 1. Local Map Tracking (Sliding Window) ---
        window_pts, window_ints, window_a, window_b, window_ids = vo_map.get_window_points()
        profiler.stop("0. Data & Rectification")
        
        monitor.check_landmark_consistency(frame_idx, "PRE-TRACK", window_pts, window_ints, window_a, window_b)
        
        if len(window_pts) > 50:
            profiler.start("1. Coarse Tracking (KLT)")
            T_C_W_guess = np.linalg.inv(T_W_C)
            
            # --- 1a. Coarse Geometric Guess (LK Optical Flow + PnP) ---
            if prev_img_rect is not None:
                R_prev, t_prev = T_C_W_guess[:3, :3], T_C_W_guess[:3, 3]
                pts_c_prev = window_pts @ R_prev.T + t_prev
                valid_z = pts_c_prev[:, 2] > 0.01
                
                if np.sum(valid_z) > 0:
                    u_prev = K_rect[0,0] * pts_c_prev[valid_z, 0] / pts_c_prev[valid_z, 2] + K_rect[0,2]
                    v_prev = K_rect[1,1] * pts_c_prev[valid_z, 1] / pts_c_prev[valid_z, 2] + K_rect[1,2]
                    
                    H_img, W_img = prev_img_rect.shape
                    in_bounds = (u_prev >= 0) & (u_prev < W_img) & (v_prev >= 0) & (v_prev < H_img)
                    
                    valid_idx = np.where(valid_z)[0]
                    valid_z[valid_idx[~in_bounds]] = False

                if np.sum(valid_z) > 20:
                    u_prev = K_rect[0,0] * pts_c_prev[valid_z, 0] / pts_c_prev[valid_z, 2] + K_rect[0,2]
                    v_prev = K_rect[1,1] * pts_c_prev[valid_z, 1] / pts_c_prev[valid_z, 2] + K_rect[1,2]
                    p0 = np.stack([u_prev, v_prev], axis=1).astype(np.float32)
                    
                    # Histogram Equalize images for extremely robust LK tracking under exposure changes
                    prev_img_eq = cv2.equalizeHist(prev_img_rect)
                    img_left_eq = cv2.equalizeHist(img_left_rect)
                    
                    p1, st, err = cv2.calcOpticalFlowPyrLK(
                        prev_img_eq, img_left_eq, p0, None,
                        winSize=(21, 21), maxLevel=3
                    )
                    
                    st = st.flatten() == 1
                    
                    # --- Fundamental Matrix Outlier Rejection ---
                    if np.sum(st) >= 8:
                        F, mask_f = cv2.findFundamentalMat(p0[st], p1[st], cv2.FM_RANSAC, 3.0, 0.99)
                        if mask_f is not None:
                            mask_f = mask_f.flatten() == 1
                            valid_indices = np.where(st)[0]
                            st[valid_indices[~mask_f]] = False
                            
                    if np.sum(st) > 20:
                        success, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                            window_pts[valid_z][st], p1[st], K_rect, None, flags=cv2.SOLVEPNP_EPNP
                        )
                        if success and inliers_pnp is not None and len(inliers_pnp) > 15:
                            R_pnp, _ = cv2.Rodrigues(rvec)
                            T_C_W_guess[:3, :3] = R_pnp
                            T_C_W_guess[:3, 3] = tvec.flatten()
                            
                            monitor.check_pose_jump(frame_idx, "PNP", np.linalg.inv(T_C_W_guess))
            
            # --- 1b. Fine Photometric Tracking (Gauss-Newton with Affine Brightness) ---
            profiler.stop("1. Coarse Tracking (KLT)")
            profiler.start("2. Fine Tracking (GN)")
            T_C_W, inliers, current_a, current_b = tracker.track_map(
                img_left_rect, 
                window_pts, 
                window_ints, 
                window_a,
                window_b,
                T_C_W_guess, 
                current_a,
                current_b,
                max_iters=10
            )
            
            T_W_C = np.linalg.inv(T_C_W)
            monitor.check_pose_jump(frame_idx, "TRACKER_GN", T_W_C)
            monitor.set_baseline(T_W_C)
            
            # --- 1.5. Prune Dead Keyframes (Low Covisibility) ---
            MIN_COVISIBLE_POINTS = 20
            
            if np.any(inliers):
                active_ids = window_ids[inliers]
                unique_ids, counts = np.unique(active_ids, return_counts=True)
                covisibility_map = dict(zip(unique_ids, counts))
                
                active_kfs = []
                for kf in vo_map.keyframes:
                    covisible_count = covisibility_map.get(kf.kf_id, 0)
                    # Protect the latest KF, drop any that fall below the threshold
                    if kf is vo_map.keyframes[-1]:
                        active_kfs.append(kf)
                    elif covisible_count >= MIN_COVISIBLE_POINTS:
                        active_kfs.append(kf)
                    else:
                        print(f"[Map Maintenance] Discarding KF {kf.kf_id:04d} due to low covisibility ({covisible_count} < {MIN_COVISIBLE_POINTS}).")
                vo_map.keyframes = active_kfs
            
        else:
            inliers = np.zeros(len(window_pts), dtype=bool)

        # --- 2. Map Expansion & Optimization ---
        # If we successfully tracked, check if we need a new keyframe
        need_new_kf = False
        if len(vo_map.keyframes) == 0:
            need_new_kf = True
        elif np.any(inliers):
            # Calculate geometric disparity (parallax) to nearest keyframes
            direct_covisibility = np.sum(inliers)
            
            latest_kf = vo_map.keyframes[-1]
            dist_to_latest = np.linalg.norm(T_W_C[:3, 3] - latest_kf.T_W_C[:3, 3])
            
            # Spawn a new keyframe if:
            # 1. We moved enough baseline for good parallax (20cm)
            # 2. We lost more than 35% of the local map points
            # 3. We are running dangerously low on active inliers (< 250)
            if dist_to_latest > 0.15:  
                need_new_kf = True
            elif direct_covisibility < (0.65 * len(window_pts)):
                need_new_kf = True
            elif direct_covisibility < 250:
                need_new_kf = True
                    
        profiler.stop("2. Fine Tracking (GN)")
        profiler.start("3. Map Maintenance (Add/Prune)")
        if need_new_kf:
            vo_map.add_keyframe(img_left_rect, depth, T_W_C, current_a, current_b)
            
            pts_before, _, _, _, _ = vo_map.get_window_points()
            
            # Jointly optimize the sliding window (Photometric Local Bundle Adjustment)
            profiler.stop("3. Map Maintenance (Add/Prune)")
            profiler.start("4. Bundle Adjustment")
            ba.optimize_window(vo_map.keyframes, max_iters=3)
            
            pts_after, ints_after, a_after, b_after, _ = vo_map.get_window_points()
            monitor.check_landmark_consistency(frame_idx, "POST-BA", pts_after, ints_after, a_after, b_after)
            monitor.check_landmark_jump(frame_idx, pts_before, pts_after)
            
            # Ensure the current tracking pose reflects the newly optimized active keyframe pose
            T_W_C = vo_map.keyframes[-1].T_W_C.copy()
            monitor.check_pose_jump(frame_idx, "POST-BA_SYNC", T_W_C)
            monitor.set_baseline(T_W_C)
            
        traj_positions.append(T_W_C[:3, 3])
        
        # Add Estimated SE(3) basis points for robust alignment
        align_X.append(T_W_C[:3, 3])
        align_X.append(T_W_C[:3, 3] + T_W_C[:3, 0] * 0.1)
        align_X.append(T_W_C[:3, 3] + T_W_C[:3, 1] * 0.1)
        align_X.append(T_W_C[:3, 3] + T_W_C[:3, 2] * 0.1)
        
        # --- 3. Umeyama Alignment (Estimated to GT) ---
        if len(traj_positions) >= 5:
            X = np.array(align_X)
            Y = np.array(align_Y[-len(align_X):])
            s, R_um, t_um = umeyama_alignment(X, Y)
            
            monitor.check_alignment_jump(frame_idx, s, R_um, t_um)
            
            # Map GT into Estimated Space so the GT trajectory visually overlays the native VO trajectory
            rr.log("world/estimated", rr.Clear(recursive=False))
            
            gt_transform = rr.Transform3D(
                translation= -R_um.T @ t_um / s,
                mat3x3= R_um.T / s
            )
            rr.log("world/gt", gt_transform)
        else:
            rr.log("world/gt", rr.Clear(recursive=False))
            
        # --- 4. 3D World View Logging (Aligned) ---
        profiler.stop("3. Map Maintenance (Add/Prune)")
        profiler.stop("4. Bundle Adjustment")
        profiler.start("5. Rerun Visualization")
        rr.log("world/estimated/trajectory", rr.LineStrips3D([traj_positions], colors=[[255, 100, 100]]))
        rr.log("world/estimated/camera", rr.Transform3D(translation=T_W_C[:3, 3], mat3x3=T_W_C[:3, :3]))
        
        # Third Person View Chase Camera (Offset -Y (Up) and -Z (Behind))
        t_view = T_W_C[:3, 3] + T_W_C[:3, :3] @ np.array([0.0, -0.5, -1.5])
        rr.log("world/view_camera", rr.Transform3D(translation=t_view, mat3x3=T_W_C[:3, :3]))
        rr.log("world/estimated/camera/image", rr.Pinhole(
            image_from_camera=K_rect, resolution=[img_left_rect.shape[1], img_left_rect.shape[0]]))
            

        
        # Log the 7 Sliding Window Keyframes as frustums in the 3D view
        for kf_idx, kf in enumerate(vo_map.keyframes):
            kf_path = f"world/estimated/keyframes/kf_{kf_idx}"
            rr.log(kf_path, rr.Transform3D(translation=kf.T_W_C[:3, 3], mat3x3=kf.T_W_C[:3, :3]))
            rr.log(f"{kf_path}/image", rr.Pinhole(
                image_from_camera=K_rect,
                resolution=[img_left_rect.shape[1], img_left_rect.shape[0]]
            ))
        
        # Camera View Overlay Logging
        # --- 6. Visualization ---
        img_rgb = cv2.cvtColor(img_left_rect, cv2.COLOR_GRAY2BGR)
        
        if not args.headless:
            cv2.imshow("Direct VO - Rectified Left", img_rgb)
            if cv2.waitKey(1) == 27:
                break
                
        # Log the clean image
        rr.log("world/estimated/camera/image", rr.Image(img_rgb[..., ::-1]))
        
        # Log the 2D projected tracking inliers onto the image panel
        if len(window_pts) > 0 and np.any(inliers):
            T_C_W = np.linalg.inv(T_W_C)
            R, t = T_C_W[:3, :3], T_C_W[:3, 3]
            
            # Project only the successfully tracked 3D points into the current 2D frame
            pts_c = window_pts[inliers] @ R.T + t
            Z = pts_c[:, 2]
            
            valid_z = Z > 0.01
            Z_valid = Z[valid_z]
            u = (K_rect[0,0] * pts_c[valid_z, 0] / Z_valid + K_rect[0,2])
            v = (K_rect[1,1] * pts_c[valid_z, 1] / Z_valid + K_rect[1,2])
            
            h, w = img_left_rect.shape
            in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            
            pts_2d = np.stack([u[in_bounds], v[in_bounds]], axis=1)
            if len(pts_2d) > 0:
                Z_final = Z_valid[in_bounds]
                
                # Dynamically normalize to the visible landmarks using percentiles 
                # (using 5th and 95th percentiles prevents outliers from squashing the colors)
                min_Z = np.percentile(Z_final, 5)
                max_Z = np.percentile(Z_final, 95)
                
                # Closer = Red (1.0), Farther = Blue (0.0)
                if max_Z > min_Z:
                    norm_Z = np.clip((max_Z - Z_final) / (max_Z - min_Z), 0.0, 1.0)
                else:
                    norm_Z = np.zeros_like(Z_final)
                    
                norm_Z_uint8 = (norm_Z * 255).astype(np.uint8)
                colors_bgr = cv2.applyColorMap(norm_Z_uint8, cv2.COLORMAP_JET)
                colors_rgb = colors_bgr.reshape(-1, 3)[:, ::-1]
                
                rr.log("world/estimated/camera/image/tracked_points", rr.Points2D(pts_2d, colors=colors_rgb, radii=2.0))
            else:
                rr.log("world/estimated/camera/image/tracked_points", rr.Clear(recursive=False))
        else:
            rr.log("world/estimated/camera/image/tracked_points", rr.Clear(recursive=False))
        
        # Log the Local Point Cloud (Sliding Window)
        window_pts, window_ints, _, _, _ = vo_map.get_window_points()
        if len(window_pts) > 0:
            window_ints_clipped = np.clip(window_ints[:, 0], 0, 255)
            colors = np.stack([window_ints_clipped, window_ints_clipped, window_ints_clipped], axis=-1).astype(np.uint8)
            rr.log("world/estimated/points", rr.Points3D(window_pts, colors=colors))
            
        # Keyframe Statistics (Markdown Table)
        if len(vo_map.keyframes) > 0:
            stats_md = "## Keyframe Statistics\n\n"
            stats_md += "| KF ID | Pts Owned | Covisible | Pos (x, y, z) |\n"
            stats_md += "|-------|-----------|-----------|---------------|\n"
            
            # Count covisibility for the current frame
            if np.any(inliers) and len(window_ids) == len(inliers):
                active_ids = window_ids[inliers]
                unique_ids, counts = np.unique(active_ids, return_counts=True)
                covisibility_map = dict(zip(unique_ids, counts))
            else:
                covisibility_map = {}
                
            for kf in vo_map.keyframes:
                pts_owned = len(kf.pts_3d_c)
                covisible = covisibility_map.get(kf.kf_id, 0)
                x, y, z = kf.T_W_C[:3, 3]
                stats_md += f"| {kf.kf_id:04d} | {pts_owned:4d} | {covisible:4d} | ({x:5.2f}, {y:5.2f}, {z:5.2f}) |\n"
                
            rr.log("metrics/keyframe_stats", rr.TextDocument(text=stats_md, media_type=rr.MediaType.MARKDOWN))
            
            if frame_idx > 0 and frame_idx % 50 == 0:
                print(f"\n--- FRAME {frame_idx} KF STATS ---")
                print(stats_md)
            
        # Telemetry Output
        num_pts = len(window_pts)
        num_inliers = np.sum(inliers)
        num_kfs = len(vo_map.keyframes)
        
        rr.log("metrics/active_landmarks", rr.Scalars([num_pts]))
        rr.log("metrics/inliers", rr.Scalars([num_inliers]))
        rr.log("metrics/keyframes", rr.Scalars([num_kfs]))
        
        if frame_idx % 10 == 0:
            print(f"FRAME {frame_idx} | Landmarks: {num_pts} | Inliers: {num_inliers} | Keyframes: {num_kfs}")
        
        # Cache image for LK flow in the next frame
        profiler.stop("5. Rerun Visualization")
        if frame_idx > 0 and frame_idx % 50 == 0:
            profiler.print_stats()
            
        prev_img_rect = img_left_rect.copy()
        
if __name__ == '__main__':
    main()
