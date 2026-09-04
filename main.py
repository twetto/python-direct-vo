import sys
from pathlib import Path
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import argparse
import json

# Hook up the dataloader
sys.path.append(str(Path(__file__).parent.parent / "neuro-dataloaders"))
from neuro_dataloaders.euroc import EuRoCLoader
from src.stereo import build_stereo_rectification_maps, compute_stereo_depth
from src.profiler import profiler
from src.tracker import DirectTracker
from src.map import Map
from src.optimizer import PhotometricBA
from src.diagnostics import VOMonitor, stereo_flow_reprojection_error
from src.feature_tracker import FeatureTracker, FeatureTrackerConfig
from src.landmark_filter import Sparse3DFilterBank, Sparse3DSettings
from src.monocular import ExperimentalMonocularVO
from src.visualization import log_vo_visualization, projected_depth_discontinuity


class NullRerun:
    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self


class DiagnosticsLog:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8") if path else None

    def write(self, record):
        if self.file is None:
            return
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()

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
    global rr, rrb
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', help='Run without OpenCV GUI')
    parser.add_argument('--enable-sparse3d', action='store_true', help='Run experimental Sparse3D monocular landmark filters')
    parser.add_argument('--mono', action='store_true', help='Run experimental left-image-only monocular VO mode')
    parser.add_argument('--no-rerun', action='store_true', help='Disable Rerun logging for smoke tests')
    parser.add_argument('--max-frames', type=int, default=None, help='Stop after this many input frames')
    parser.add_argument('--dataset-path', type=str, default=None, help='Override EuRoC dataset path')
    parser.add_argument('--diagnostics-log', type=str, default=None, help='Write per-frame diagnostics as JSONL')
    parser.add_argument('--no-mono-debug-stereo-depth', action='store_true', help='Disable stereo-depth debug metrics in mono mode')
    args = parser.parse_args()
    mono_debug_stereo_depth = bool(args.mono and not args.no_mono_debug_stereo_depth)

    # DATASET_PATH = "C:/Users/twetto/Downloads/machine_hall/machine_hall/MH_01_easy/MH_01_easy"
    DATASET_PATH = args.dataset_path or "C:/Users/twetto/Downloads/vicon_room1/vicon_room1/V1_01_easy"
    diag_log = DiagnosticsLog(args.diagnostics_log)
    
    # Initialize Rerun GUI with a fresh app id to clear ghost blueprints
    if args.no_rerun:
        rr = NullRerun()
        rrb = NullRerun()
        print("Rerun logging disabled.")
    else:
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
    loader_right = EuRoCLoader(DATASET_PATH, cam_id=1, undistort=False) if (not args.mono or mono_debug_stereo_depth) else None

    print("Initializing Stereo Rectification...")
    if args.mono and not mono_debug_stereo_depth:
        maps_left = cv2.initUndistortRectifyMap(
            loader_left.camera.K,
            loader_left.camera.dist_coeffs,
            np.eye(3),
            loader_left.camera.K,
            loader_left.camera.resolution,
            cv2.CV_32FC1,
        )
        maps_right = None
        f = loader_left.camera.K[0, 0]
        b = 0.0
        R1 = np.eye(3)
        cx_rect = loader_left.camera.K[0, 2]
        cy_rect = loader_left.camera.K[1, 2]
    else:
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
    feature_tracker = FeatureTracker(FeatureTrackerConfig(max_features=800))
    sparse3d = Sparse3DFilterBank(K_rect, Sparse3DSettings(
        min_track_length=3,
        birth_min_flow_px=0.5,
        conv_depth_variance=25.0,
    ))
    mono_vo = ExperimentalMonocularVO(K_rect, min_essential_tracks=12, min_direct_landmarks=50) if args.mono else None
    
    T_W_C = np.eye(4)
    current_a = 0.0
    current_b = 0.0
    prev_img_rect = None
    prev_depth_rect = None
    prev_T_W_C_for_stereo_flow_debug = None
    
    traj_positions = []
    traj_positions_gt = []
    align_X = []
    align_Y = []
    
    print("Starting Direct Visual Odometry with Rerun...")
    frame_iter = enumerate(loader_left) if args.mono and loader_right is None else enumerate(zip(loader_left, loader_right))
    for frame_idx, samples in frame_iter:
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break
        if args.mono and loader_right is None:
            sample_left = samples
            sample_right = None
        else:
            sample_left, sample_right = samples
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}...")
        rr.set_time("frame", sequence=frame_idx)
        
        if sample_right is not None and abs(sample_left.timestamp - sample_right.timestamp) > 0.01:
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
        if args.mono and sample_right is None:
            img_right_rect = None
            depth = np.zeros_like(img_left_rect, dtype=np.float32)
        else:
            img_right_rect = cv2.remap(sample_right.image, maps_right[0], maps_right[1], cv2.INTER_LINEAR)
            _, depth = compute_stereo_depth(img_left_rect, img_right_rect, f, b)
        # --- 1. Local Map Tracking (Sliding Window) ---
        window_pts, window_ints, window_a, window_b, window_ids = vo_map.get_window_points()
        profiler.stop("0. Data & Rectification")

        if args.mono:
            profiler.start("1. Monocular VO")
            mono_result = mono_vo.process(frame_idx, img_left_rect)
            T_W_C = mono_result.T_W_C
            inliers = np.zeros(len(window_pts), dtype=bool)
            profiler.stop("1. Monocular VO")
        else:
        
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

        if not args.mono and len(window_pts) <= 50:
            inliers = np.zeros(len(window_pts), dtype=bool)
            profiler.start("3. Map Maintenance (Add/Prune)")
            if len(vo_map.keyframes) == 0:
                vo_map.add_keyframe(img_left_rect, depth, T_W_C, current_a, current_b)

        if args.enable_sparse3d and not args.mono:
            sparse3d_obs = feature_tracker.update(img_left_rect)
            sparse3d.update(frame_idx, sparse3d_obs, T_W_C)
            
        traj_positions.append(T_W_C[:3, 3])
        if args.mono and mono_debug_stereo_depth:
            stereo_flow_reproj = stereo_flow_reprojection_error(
                prev_img_rect,
                img_left_rect,
                prev_depth_rect,
                K_rect,
                prev_T_W_C_for_stereo_flow_debug,
                T_W_C,
                mono_vo.mono_map.point_cloud(max_points=1000),
            )
        else:
            stereo_flow_reproj = stereo_flow_reprojection_error(
                None,
                None,
                None,
                K_rect,
                None,
                T_W_C,
                (np.empty((0, 3), dtype=np.float64),),
            )
        
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

        if not args.headless:
            img_rgb = cv2.cvtColor(img_left_rect, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Direct VO - Rectified Left", img_rgb)
            if cv2.waitKey(1) == 27:
                break

        skip_visualization_work = args.headless and args.no_rerun
        if args.mono and skip_visualization_work:
            vis_mode = "mono"
            vis_keyframes = mono_vo.mono_map.keyframes
            vis_point_cloud = (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=int),
            )
            vis_tracked_points = None
            covisibility_map = {}
            depth_jump = {
                "visible": 0,
                "pairs": 0,
                "median_abs": 0.0,
                "p90_abs": 0.0,
                "median_rel": 0.0,
            }
            ekf_landmarks = None
            keyframe_klt_residuals = None
        elif args.mono:
            vis_mode = "mono"
            vis_keyframes = mono_vo.mono_map.keyframes
            vis_point_cloud = mono_vo.mono_map.point_cloud(max_points=3000)
            vis_tracked_points = None
            covisibility_map = {}
            depth_jump = projected_depth_discontinuity(T_W_C, vis_point_cloud, K_rect, img_left_rect.shape)
            ekf_landmarks = mono_vo.ekf_landmark_glyphs(img_left_rect.shape)
            keyframe_klt_residuals = mono_vo.keyframe_klt_residual_segments(img_left_rect.shape)
        else:
            window_pts, window_ints, _, _, _ = vo_map.get_window_points()
            if np.any(inliers) and len(window_ids) == len(inliers):
                active_ids = window_ids[inliers]
                unique_ids, counts = np.unique(active_ids, return_counts=True)
                covisibility_map = dict(zip(unique_ids, counts))
            else:
                covisibility_map = {}
            vis_mode = "stereo"
            vis_keyframes = vo_map.keyframes
            vis_point_cloud = vo_map.point_cloud(max_points=3000)
            vis_tracked_points = window_pts[inliers] if len(window_pts) == len(inliers) else None
            ekf_landmarks = None
            keyframe_klt_residuals = None
            depth_jump = {
                "visible": 0,
                "pairs": 0,
                "median_abs": 0.0,
                "p90_abs": 0.0,
                "median_rel": 0.0,
            }

        if not skip_visualization_work:
            log_vo_visualization(
                rr,
                frame_idx,
                vis_mode,
                img_left_rect,
                K_rect,
                T_W_C,
                traj_positions,
                vis_keyframes,
                vis_point_cloud,
                tracked_points=vis_tracked_points,
                covisibility_map=covisibility_map,
                ekf_landmarks=ekf_landmarks,
                keyframe_klt_residuals=keyframe_klt_residuals,
            )
            
        # Telemetry Output
        num_pts = len(window_pts)
        num_inliers = np.sum(inliers)
        num_kfs = len(vo_map.keyframes)
        if args.mono:
            num_pts = mono_result.mature_tracks
            num_inliers = mono_result.direct_inliers
            num_kfs = mono_result.mono_keyframes
        
        rr.log("metrics/active_landmarks", rr.Scalars([num_pts]))
        rr.log("metrics/inliers", rr.Scalars([num_inliers]))
        rr.log("metrics/keyframes", rr.Scalars([num_kfs]))
        if args.enable_sparse3d and not args.mono:
            rr.log("metrics/sparse3d_tracks", rr.Scalars([sparse3d.feature_count()]))
            rr.log("metrics/sparse3d_pending", rr.Scalars([len(sparse3d.pending)]))
        if args.mono:
            rr.log("metrics/sparse3d_tracks", rr.Scalars([mono_result.mature_tracks]))
            rr.log("metrics/sparse3d_pending", rr.Scalars([mono_result.pending_tracks]))
            rr.log("metrics/mono_depth_local_jump_median_m", rr.Scalars([depth_jump["median_abs"]]))
            rr.log("metrics/mono_depth_local_jump_p90_m", rr.Scalars([depth_jump["p90_abs"]]))
            rr.log("metrics/mono_keyframe_klt_residual_median_px", rr.Scalars([mono_result.mono_keyframe_klt_residual_median_px]))
            rr.log("metrics/mono_keyframe_klt_residual_p90_px", rr.Scalars([mono_result.mono_keyframe_klt_residual_p90_px]))
            rr.log("metrics/mono_stereo_flow_reproj_median_px", rr.Scalars([stereo_flow_reproj["median_px"]]))
            rr.log("metrics/mono_stereo_flow_reproj_p90_px", rr.Scalars([stereo_flow_reproj["p90_px"]]))
            rr.log("metrics/mono_keyframe_pose_update_norm", rr.Scalars([mono_result.mono_keyframe_pose_update_norm]))
        
        if frame_idx % 10 == 0:
            sparse3d_msg = ""
            if args.mono:
                sparse3d_msg = (
                    f" | Mono Sparse3D: {mono_result.mature_tracks} mature / {mono_result.pending_tracks} pending"
                    f" | direct={mono_result.used_direct}"
                    f" | direct_try={mono_result.direct_attempted}"
                    f" | source={mono_result.motion_source}"
                    f" | bootstrap={mono_result.bootstrap_active}"
                    f" | kfs={mono_result.mono_keyframes}"
                    f" | ba={mono_result.mono_ba_ran}"
                    f" | dz_med={depth_jump['median_abs']:.2f}m"
                    f" | gftt_mask={mono_result.mono_gftt_exclusions}"
                    f" | kf_klt={mono_result.mono_keyframe_klt_inliers}/{mono_result.mono_keyframe_klt_tracks}"
                    f" | kf_res={mono_result.mono_keyframe_klt_residual_median_px:.1f}px"
                    f" | stereo_flow_res={stereo_flow_reproj['median_px']:.1f}/{stereo_flow_reproj['p90_px']:.1f}px"
                    f"({stereo_flow_reproj['compared']})"
                    f" | flow_cos={mono_result.mono_keyframe_klt_flow_cos_median:.2f}"
                    f" | kf_pose_delta={mono_result.mono_keyframe_pose_update_norm:.3f}m"
                )
            elif args.enable_sparse3d:
                sparse3d_msg = f" | Sparse3D: {sparse3d.feature_count()} mature / {len(sparse3d.pending)} pending"
            print(f"FRAME {frame_idx} | Landmarks: {num_pts} | Inliers: {num_inliers} | Keyframes: {num_kfs}{sparse3d_msg}")
        diag_log.write({
            "frame": frame_idx,
            "mode": "mono" if args.mono else "stereo",
            "timestamp": float(sample_left.timestamp),
            "position": T_W_C[:3, 3].astype(float).tolist(),
            "gt_position": T_W_C_gt[:3, 3].astype(float).tolist(),
            "active_landmarks": int(num_pts),
            "inliers": int(num_inliers),
            "keyframes": int(num_kfs),
            "sparse3d_tracks": int(mono_result.mature_tracks if args.mono else sparse3d.feature_count()),
            "sparse3d_pending": int(mono_result.pending_tracks if args.mono else len(sparse3d.pending)),
            "mono_direct_landmarks": int(mono_result.direct_landmarks) if args.mono else 0,
            "mono_direct_hypotheses": int(mono_result.direct_hypotheses) if args.mono else 0,
            "mono_direct_attempted": bool(mono_result.direct_attempted) if args.mono else False,
            "mono_direct_used": bool(mono_result.used_direct) if args.mono else False,
            "mono_essential_common_tracks": int(mono_result.essential_common_tracks) if args.mono else 0,
            "mono_essential_inliers": int(mono_result.essential_inliers) if args.mono else 0,
            "mono_essential_used": bool(mono_result.essential_used) if args.mono else False,
            "mono_image_motion_fallback_used": bool(mono_result.image_motion_fallback_used) if args.mono else False,
            "mono_bootstrap_active": bool(mono_result.bootstrap_active) if args.mono else False,
            "mono_motion_source": str(mono_result.motion_source) if args.mono else "",
            "mono_motion_step_norm": float(mono_result.motion_step_norm) if args.mono else 0.0,
            "mono_rotation_step_deg": float(mono_result.rotation_step_deg) if args.mono else 0.0,
            "mono_keyframes": int(mono_result.mono_keyframes) if args.mono else 0,
            "mono_keyframe_inserted": bool(mono_result.mono_keyframe_inserted) if args.mono else False,
            "mono_keyframe_reason": str(mono_result.mono_keyframe_reason) if args.mono else "",
            "mono_ba_ran": bool(mono_result.mono_ba_ran) if args.mono else False,
            "mono_ba_window": int(mono_result.mono_ba_window) if args.mono else 0,
            "mono_geo_ba_ran": bool(mono_result.mono_geo_ba_ran) if args.mono else False,
            "mono_geo_ba_edges": int(mono_result.mono_geo_ba_edges) if args.mono else 0,
            "mono_depth_ba_ran": bool(mono_result.mono_depth_ba_ran) if args.mono else False,
            "mono_depth_ba_landmarks": int(mono_result.mono_depth_ba_landmarks) if args.mono else 0,
            "mono_depth_ba_edges": int(mono_result.mono_depth_ba_edges) if args.mono else 0,
            "mono_depth_ba_cost_before": float(mono_result.mono_depth_ba_cost_before) if args.mono else 0.0,
            "mono_depth_ba_cost_after": float(mono_result.mono_depth_ba_cost_after) if args.mono else 0.0,
            "mono_depth_ba_updated": int(mono_result.mono_depth_ba_updated) if args.mono else 0,
            "mono_depth_ba_median_abs_log_update": float(mono_result.mono_depth_ba_median_abs_log_update) if args.mono else 0.0,
            "mono_depth_ba_max_abs_log_update": float(mono_result.mono_depth_ba_max_abs_log_update) if args.mono else 0.0,
            "mono_observations": int(mono_result.mono_observations) if args.mono else 0,
            "mono_map_landmarks": int(mono_result.mono_map_landmarks) if args.mono else 0,
            "mono_depth_jump_visible": int(depth_jump["visible"]) if args.mono else 0,
            "mono_depth_jump_pairs": int(depth_jump["pairs"]) if args.mono else 0,
            "mono_depth_jump_median_abs_m": float(depth_jump["median_abs"]) if args.mono else 0.0,
            "mono_depth_jump_p90_abs_m": float(depth_jump["p90_abs"]) if args.mono else 0.0,
            "mono_depth_jump_median_rel": float(depth_jump["median_rel"]) if args.mono else 0.0,
            "mono_gftt_exclusions": int(mono_result.mono_gftt_exclusions) if args.mono else 0,
            "mono_keyframe_klt_tracks": int(mono_result.mono_keyframe_klt_tracks) if args.mono else 0,
            "mono_keyframe_klt_inliers": int(mono_result.mono_keyframe_klt_inliers) if args.mono else 0,
            "mono_keyframe_klt_used": bool(mono_result.mono_keyframe_klt_used) if args.mono else False,
            "mono_keyframe_klt_residual_median_px": float(mono_result.mono_keyframe_klt_residual_median_px) if args.mono else 0.0,
            "mono_keyframe_klt_residual_p90_px": float(mono_result.mono_keyframe_klt_residual_p90_px) if args.mono else 0.0,
            "mono_keyframe_klt_flow_cos_median": float(mono_result.mono_keyframe_klt_flow_cos_median) if args.mono else 0.0,
            "mono_keyframe_klt_flow_cos_p10": float(mono_result.mono_keyframe_klt_flow_cos_p10) if args.mono else 0.0,
            "mono_stereo_flow_reproj_projected": int(stereo_flow_reproj["projected"]) if args.mono else 0,
            "mono_stereo_flow_reproj_sampled": int(stereo_flow_reproj["sampled"]) if args.mono else 0,
            "mono_stereo_flow_reproj_valid_depth": int(stereo_flow_reproj["valid_depth"]) if args.mono else 0,
            "mono_stereo_flow_reproj_tracked": int(stereo_flow_reproj["tracked"]) if args.mono else 0,
            "mono_stereo_flow_reproj_compared": int(stereo_flow_reproj["compared"]) if args.mono else 0,
            "mono_stereo_flow_reproj_median_px": float(stereo_flow_reproj["median_px"]) if args.mono else 0.0,
            "mono_stereo_flow_reproj_p90_px": float(stereo_flow_reproj["p90_px"]) if args.mono else 0.0,
            "mono_stereo_flow_depth_ratio_median": float(stereo_flow_reproj["median_depth_ratio"]) if args.mono else 0.0,
            "mono_svo_match_stats": mono_result.mono_svo_match_stats if args.mono else {},
            "mono_keyframe_reproj_rejected": int(mono_result.mono_keyframe_reproj_rejected) if args.mono else 0,
            "mono_keyframe_reproj_median_px": float(mono_result.mono_keyframe_reproj_median_px) if args.mono else 0.0,
            "mono_landmark_updates": int(mono_result.mono_landmark_updates) if args.mono else 0,
            "mono_landmark_update_median_m": float(mono_result.mono_landmark_update_median_m) if args.mono else 0.0,
            "mono_landmark_reproj_before_px": float(mono_result.mono_landmark_reproj_before_px) if args.mono else 0.0,
            "mono_landmark_reproj_after_px": float(mono_result.mono_landmark_reproj_after_px) if args.mono else 0.0,
            "mono_keyframe_pose_update_norm": float(mono_result.mono_keyframe_pose_update_norm) if args.mono else 0.0,
            "mono_keyframe_pose_update_rot_deg": float(mono_result.mono_keyframe_pose_update_rot_deg) if args.mono else 0.0,
            "mono_direct_candidate_stats": mono_result.mono_direct_candidate_stats if args.mono else [],
            "mono_keyframe_reference_counts": mono_result.mono_keyframe_reference_counts if args.mono else {},
            "mono_keyframe_visible_counts": mono_result.mono_keyframe_visible_counts if args.mono else {},
            "mono_keyframe_prefilter_kept_counts": mono_result.mono_keyframe_prefilter_kept_counts if args.mono else {},
            "mono_keyframe_prefilter_rejected_counts": mono_result.mono_keyframe_prefilter_rejected_counts if args.mono else {},
            "mono_keyframe_low_usefulness_frames": mono_result.mono_keyframe_low_usefulness_frames if args.mono else {},
            "mono_keyframe_usefulness_ratio": mono_result.mono_keyframe_usefulness_ratio if args.mono else {},
            "mono_direct_prefilter_kept": int(mono_result.mono_direct_prefilter_kept) if args.mono else 0,
            "mono_direct_prefilter_rejected": int(mono_result.mono_direct_prefilter_rejected) if args.mono else 0,
            "mono_keyframe_discarded_id": int(mono_result.mono_keyframe_discarded_id) if args.mono else -1,
            "mono_keyframe_discard_reason": str(mono_result.mono_keyframe_discard_reason) if args.mono else "",
        })
        
        # Cache image for LK flow in the next frame
        profiler.stop("5. Rerun Visualization")
        prev_depth_rect = depth.copy() if args.mono and mono_debug_stereo_depth else None
        prev_T_W_C_for_stereo_flow_debug = T_W_C.copy() if args.mono and mono_debug_stereo_depth else None
        if frame_idx > 0 and frame_idx % 50 == 0:
            profiler.print_stats()
            
        prev_img_rect = img_left_rect.copy()
    diag_log.close()

    # --- Scale-aware trajectory evaluation (Phase 0 measuring stick) ---
    if len(traj_positions) >= 3 and len(traj_positions_gt) >= 3:
        from src.mono_eval import evaluate, format_report
        n = min(len(traj_positions), len(traj_positions_gt))
        metrics = evaluate(np.asarray(traj_positions[:n]), np.asarray(traj_positions_gt[:n]))
        print(format_report(metrics))
        
if __name__ == '__main__':
    main()
