import numpy as np
import cv2
import scipy.ndimage as nd
import sophuspy as sp
from scipy.optimize import least_squares, minimize_scalar

from src.tracker import PATTERN_DX, PATTERN_DY, PATTERN_SIZE

class PhotometricBA:
    def __init__(self, K):
        self.K = K
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

    def _mono_window(self, mono_window):
        if hasattr(mono_window, "keyframes") and hasattr(mono_window, "keyframe_reference_records"):
            return mono_window.keyframes, mono_window
        return mono_window, None

    def _mono_reference_records(self, mono_map, kf):
        if mono_map is not None:
            return mono_map.keyframe_reference_records(kf)
        refs = []
        for point_idx, fid in enumerate(kf.landmark_ids):
            refs.append({
                "landmark_id": int(fid),
                "point_idx": int(point_idx),
                "point_w": np.asarray(kf.points_w[point_idx], dtype=np.float64),
                "intensity": np.asarray(kf.intensities[point_idx], dtype=np.float32),
            })
        return refs

    def _mono_observation_point(self, mono_map, obs):
        if mono_map is not None:
            track = mono_map.landmarks.get(int(obs.landmark_id))
            if track is not None:
                return np.asarray(track.point_w, dtype=np.float64)
        return np.asarray(obs.point_w, dtype=np.float64)

    def optimize_window(self, keyframes, max_iters=10):
        """
        Jointly optimizes the SE(3) poses of all keyframes in the window
        to minimize photometric reprojection error between all overlapping pairs.
        Keyframe 0 is held fixed as the gauge origin.
        """
        if len(keyframes) < 2:
            return

        # 1. Setup variables: We optimize the poses of KF 1 to N-1
        n_opt = len(keyframes) - 1
        x0 = np.zeros(n_opt * 6)
        
        # Cache original poses
        original_poses = [kf.T_W_C.copy() for kf in keyframes]

        # Pre-compute valid point-to-camera pairs using initial poses 
        # so the residual array size stays perfectly constant for scipy finite differences.
        active_pairs = []
        for i, kf_host in enumerate(keyframes):
            if len(kf_host.pts_3d_w) == 0:
                continue
                
            step = max(1, len(kf_host.pts_3d_c) // 100)
            pts_c_host = kf_host.pts_3d_c[::step]
            pts_w_host = kf_host.pts_3d_w[::step]
            ints_host = kf_host.intensities[::step]
            
            for j, kf_target in enumerate(keyframes):
                if i == j:
                    continue
                    
                T_C_W_target = np.linalg.inv(original_poses[j])
                pts_c = pts_w_host @ T_C_W_target[:3, :3].T + T_C_W_target[:3, 3]
                
                Z = pts_c[:, 2]
                
                # Filter to only points initially valid and inside the target frustum
                valid = Z > 0.01
                if not np.any(valid): continue
                
                X = pts_c[valid, 0]
                Y = pts_c[valid, 1]
                u = self.fx * X / Z[valid] + self.cx
                v = self.fy * Y / Z[valid] + self.cy
                
                h, w = kf_target.img.shape
                in_bounds = (u >= 1) & (u < w - 2) & (v >= 1) & (v < h - 2)
                
                final_mask = np.zeros(len(pts_w_host), dtype=bool)
                idx_valid = np.where(valid)[0]
                final_mask[idx_valid[in_bounds]] = True
                
                if np.sum(final_mask) > 10:
                    active_pairs.append({
                        'i': i,
                        'j': j,
                        'pts_c': pts_c_host[final_mask],
                        'ref_ints': ints_host[final_mask]
                    })

        if len(active_pairs) == 0:
            return

        def residual_fn(x):
            current_poses = [original_poses[0]]
            for i in range(n_opt):
                delta = x[i*6:(i+1)*6]
                # CRITICAL FIX: Perturbations MUST be applied in the Local Camera frame (Right-Multiplication).
                T_W_C_new = original_poses[i+1] @ sp.SE3.exp(delta).matrix()
                current_poses.append(T_W_C_new)

            residuals = []
            
            for pair in active_pairs:
                i = pair['i']
                j = pair['j']
                pts_c_host = pair['pts_c']
                ref_ints = pair['ref_ints']
                
                # 1. Map from Host Camera to World
                T_W_C_host = current_poses[i]
                pts_w = pts_c_host @ T_W_C_host[:3, :3].T + T_W_C_host[:3, 3]
                
                # 2. Map from World to Target Camera
                T_C_W_target = np.linalg.inv(current_poses[j])
                pts_c = pts_w @ T_C_W_target[:3, :3].T + T_C_W_target[:3, 3]
                
                Z = pts_c[:, 2]
                
                # Clip Z to avoid div by zero if perturbation pushes point behind camera
                Z_clipped = np.clip(Z, 0.01, None)
                
                u = self.fx * pts_c[:, 0] / Z_clipped + self.cx
                v = self.fy * pts_c[:, 1] / Z_clipped + self.cy
                
                h, w = keyframes[j].img.shape
                
                # Generate 5-point pattern coordinates
                dx = np.array([0, 1, -1, 0, 0])
                dy = np.array([0, 0, 0, 1, -1])
                u_pat = (u[:, None] + dx).flatten()
                v_pat = (v[:, None] + dy).flatten()
                
                u_clipped = np.clip(u_pat, 0, w - 1)
                v_clipped = np.clip(v_pat, 0, h - 1)
                
                target_ints = nd.map_coordinates(keyframes[j].img, [v_clipped, u_clipped], order=1).reshape(-1, 5)
                
                # DSO Relative Exposure Mapping
                exp_diff = np.exp(keyframes[j].a - keyframes[i].a)
                bias_diff = keyframes[j].b - keyframes[i].b
                predicted_ints = exp_diff * ref_ints + bias_diff
                
                res = predicted_ints - target_ints
                
                # Zero out residuals for points that actually went out of bounds or behind camera
                invalid = (Z < 0.01) | (u < 1) | (u > w-2) | (v < 1) | (v > h-2)
                res[invalid] = 0.0
                
                residuals.append(res.flatten())
                
            return np.concatenate(residuals)

        # 2. Run Least Squares Optimization with STRICT Trust-Region Bounds
        # This physically prevents the optimizer from taking massive multi-meter steps 
        # if the Jacobian becomes degenerate in textureless regions!

        # Precompute image gradients for Analytical Jacobian
        grad_xs = []
        grad_ys = []
        for kf in keyframes:
            grad_xs.append(cv2.Scharr(kf.img, cv2.CV_32F, 1, 0) / 16.0)
            grad_ys.append(cv2.Scharr(kf.img, cv2.CV_32F, 0, 1) / 16.0)

        def jacobian_fn(x):
            current_poses = [original_poses[0]]
            for i in range(n_opt):
                delta = x[i*6:(i+1)*6]
                T_W_C_new = original_poses[i+1] @ sp.SE3.exp(delta).matrix()
                current_poses.append(T_W_C_new)

            J_full = []
            
            for pair in active_pairs:
                i = pair["i"]
                j = pair["j"]
                pts_c_host = pair["pts_c"]
                
                N_pts = len(pts_c_host)
                # J_pair shape: [N_pts * 5, n_opt * 6]
                J_pair = np.zeros((N_pts * 5, n_opt * 6))
                
                T_W_C_host = current_poses[i]
                T_C_W_target = np.linalg.inv(current_poses[j])
                
                pts_w = pts_c_host @ T_W_C_host[:3, :3].T + T_W_C_host[:3, 3]
                pts_c = pts_w @ T_C_W_target[:3, :3].T + T_C_W_target[:3, 3]
                
                Z = pts_c[:, 2]
                Z_clipped = np.clip(Z, 0.01, None)
                Z_inv = 1.0 / Z_clipped
                
                X, Y = pts_c[:, 0], pts_c[:, 1]
                
                u = self.fx * X * Z_inv + self.cx
                v = self.fy * Y * Z_inv + self.cy
                
                h, w = keyframes[j].img.shape
                
                dx = np.array([0, 1, -1, 0, 0])
                dy = np.array([0, 0, 0, 1, -1])
                u_pat = (u[:, None] + dx).flatten()
                v_pat = (v[:, None] + dy).flatten()
                
                u_clipped = np.clip(u_pat, 0, w - 1)
                v_clipped = np.clip(v_pat, 0, h - 1)
                
                ix = nd.map_coordinates(grad_xs[j], [v_clipped, u_clipped], order=1).reshape(-1, 5)
                iy = nd.map_coordinates(grad_ys[j], [v_clipped, u_clipped], order=1).reshape(-1, 5)
                
                # Chain rule: d(res) = - d(target)
                J_px = -ix * (self.fx * Z_inv)[:, None]
                J_py = -iy * (self.fy * Z_inv)[:, None]
                J_pz = -(J_px * X[:, None] + J_py * Y[:, None]) * Z_inv[:, None]
                
                # J_proj shape: [N, 5, 3]
                J_proj = np.stack([J_px, J_py, J_pz], axis=-1)
                
                invalid = (Z < 0.01) | (u < 1) | (u > w-2) | (v < 1) | (v > h-2)
                
                # If Host is optimized
                if i > 0:
                    host_idx = i - 1
                    X_h, Y_h, Z_h = pts_c_host[:, 0], pts_c_host[:, 1], pts_c_host[:, 2]
                    J_se3_h = np.zeros((N_pts, 3, 6))
                    J_se3_h[:, 0, 0] = 1; J_se3_h[:, 1, 1] = 1; J_se3_h[:, 2, 2] = 1
                    J_se3_h[:, 0, 4] = -Z_h; J_se3_h[:, 0, 5] = Y_h
                    J_se3_h[:, 1, 3] = Z_h;  J_se3_h[:, 1, 5] = -X_h
                    J_se3_h[:, 2, 3] = -Y_h; J_se3_h[:, 2, 4] = X_h
                    
                    R_t_h = T_C_W_target[:3, :3] @ T_W_C_host[:3, :3]
                    J_h_rot = np.einsum("ab,nbc->nac", R_t_h, J_se3_h)
                    
                    J_block_h = np.einsum("npc,ncv->npv", J_proj, J_h_rot)
                    J_block_h[invalid] = 0.0
                    J_pair[:, host_idx*6:(host_idx+1)*6] = J_block_h.reshape(-1, 6)
                    
                # If Target is optimized
                if j > 0:
                    target_idx = j - 1
                    J_se3_t = np.zeros((N_pts, 3, 6))
                    J_se3_t[:, 0, 0] = -1; J_se3_t[:, 1, 1] = -1; J_se3_t[:, 2, 2] = -1
                    J_se3_t[:, 0, 4] = Z;  J_se3_t[:, 0, 5] = -Y
                    J_se3_t[:, 1, 3] = -Z; J_se3_t[:, 1, 5] = X
                    J_se3_t[:, 2, 3] = Y;  J_se3_t[:, 2, 4] = -X
                    
                    J_block_t = np.einsum("npc,ncv->npv", J_proj, J_se3_t)
                    J_block_t[invalid] = 0.0
                    J_pair[:, target_idx*6:(target_idx+1)*6] = J_block_t.reshape(-1, 6)
                    
                J_full.append(J_pair)
                
            return np.vstack(J_full)

        # 2. Run Least Squares Optimization with STRICT Trust-Region Bounds
        bounds = (-0.1, 0.1)
        
        res = least_squares(
            residual_fn, 
            x0, 
            jac=jacobian_fn,
            bounds=bounds,
            method='trf', 
            loss='huber', 
            f_scale=10.0, 
            max_nfev=max_iters
        )
        
        # 3. Apply Optimized Poses
        optimized_x = res.x
        for i in range(n_opt):
            delta = optimized_x[i*6:(i+1)*6]
            # Apply locally in camera frame
            T_W_C_new = original_poses[i+1] @ sp.SE3.exp(delta).matrix()
            kf = keyframes[i+1]
            kf.T_W_C = T_W_C_new
            
            # Re-project the local stereo depth points into the optimized world pose
            if len(kf.pts_3d_c) > 0:
                R_W_C = kf.T_W_C[:3, :3]
                t_W_C = kf.T_W_C[:3, 3]
                kf.pts_3d_w = (R_W_C @ kf.pts_3d_c.T).T + t_W_C

    def optimize_mono_pose_window(self, mono_window, max_iters=3, max_points_per_host=120):
        """
        Pose-only photometric BA for mono keyframes.
        Landmarks are fixed world points from Sparse3D; the oldest keyframe is the gauge.
        """
        keyframes, mono_map = self._mono_window(mono_window)
        if len(keyframes) < 2:
            return {"ran": False, "window": len(keyframes), "residuals": 0}

        n_opt = len(keyframes) - 1
        x0 = np.zeros(n_opt * 6)
        original_poses = [kf.T_W_C.copy() for kf in keyframes]
        active_pairs = []

        for i, kf_host in enumerate(keyframes):
            refs = self._mono_reference_records(mono_map, kf_host)
            if len(refs) == 0:
                continue
            step = max(1, len(refs) // max_points_per_host)
            refs = refs[::step]
            pts_w = np.asarray([ref["point_w"] for ref in refs], dtype=np.float64)
            ref_ints = np.asarray([ref["intensity"] for ref in refs], dtype=np.float32)
            for j, kf_target in enumerate(keyframes):
                if i == j:
                    continue
                mask = self._visible_mask(pts_w, original_poses[j], kf_target.image.shape)
                if np.sum(mask) > 10:
                    active_pairs.append({
                        "i": i,
                        "j": j,
                        "pts_w": pts_w[mask],
                        "ref_ints": ref_ints[mask],
                    })

        if not active_pairs:
            return {"ran": False, "window": len(keyframes), "residuals": 0}

        def current_poses(x):
            poses = [original_poses[0]]
            for i in range(n_opt):
                delta = x[i * 6:(i + 1) * 6]
                poses.append(original_poses[i + 1] @ sp.SE3.exp(delta).matrix())
            return poses

        def residual_fn(x):
            poses = current_poses(x)
            residuals = []
            for pair in active_pairs:
                i = pair["i"]
                j = pair["j"]
                pts_w = pair["pts_w"]
                T_C_W = np.linalg.inv(poses[j])
                pts_c = pts_w @ T_C_W[:3, :3].T + T_C_W[:3, 3]
                z = np.clip(pts_c[:, 2], 0.01, None)
                u = self.fx * pts_c[:, 0] / z + self.cx
                v = self.fy * pts_c[:, 1] / z + self.cy
                h, w = keyframes[j].image.shape
                u_pat = (u[:, None] + PATTERN_DX).reshape(-1)
                v_pat = (v[:, None] + PATTERN_DY).reshape(-1)
                target = nd.map_coordinates(
                    keyframes[j].image,
                    [np.clip(v_pat, 0, h - 1), np.clip(u_pat, 0, w - 1)],
                    order=1,
                ).reshape(-1, PATTERN_SIZE)
                exp_diff = np.exp(keyframes[j].affine_a - keyframes[i].affine_a)
                bias_diff = keyframes[j].affine_b - keyframes[i].affine_b
                res = exp_diff * pair["ref_ints"] + bias_diff - target
                invalid = (pts_c[:, 2] < 0.01) | (u < 1) | (u > w - 2) | (v < 1) | (v > h - 2)
                res[invalid] = 0.0
                residuals.append(res.reshape(-1))
            return np.concatenate(residuals)

        res = least_squares(
            residual_fn,
            x0,
            bounds=(-0.05, 0.05),
            method="trf",
            loss="huber",
            f_scale=10.0,
            max_nfev=max_iters,
        )

        for i in range(n_opt):
            delta = res.x[i * 6:(i + 1) * 6]
            keyframes[i + 1].T_W_C = original_poses[i + 1] @ sp.SE3.exp(delta).matrix()

        return {"ran": True, "window": len(keyframes), "residuals": int(res.fun.size)}

    def optimize_mono_inverse_depth_window(
        self,
        mono_window,
        max_iters=5,
        max_landmarks=180,
        max_targets_per_landmark=4,
        max_initial_photometric_error=80.0,
        max_abs_log_depth_update=0.12,
    ):
        """
        Refine mono landmark inverse depths along their host keyframe rays.
        Poses are fixed; high-error target projections are treated as occluded and skipped.
        """
        keyframes, mono_map = self._mono_window(mono_window)
        if len(keyframes) < 2:
            return {
                "ran": False,
                "window": len(keyframes),
                "landmarks": 0,
                "edges": 0,
                "cost_before": 0.0,
                "cost_after": 0.0,
                "updated": 0,
                "median_abs_log_depth_update": 0.0,
                "max_abs_log_depth_update": 0.0,
            }

        original_poses = [kf.T_W_C.copy() for kf in keyframes]
        hosts = []
        seen = set()
        for host_idx, kf in enumerate(keyframes):
            refs = self._mono_reference_records(mono_map, kf)
            if len(refs) == 0:
                continue
            for ref in refs:
                fid_i = int(ref["landmark_id"])
                if fid_i in seen:
                    continue
                seen.add(fid_i)
                T_C_W_host = np.linalg.inv(original_poses[host_idx])
                p_host = T_C_W_host[:3, :3] @ np.asarray(ref["point_w"], dtype=np.float64) + T_C_W_host[:3, 3]
                if p_host[2] <= 0.05:
                    continue
                bearing = p_host / np.linalg.norm(p_host)
                depth_along_ray = np.linalg.norm(p_host)
                if depth_along_ray <= 0.05:
                    continue
                hosts.append({
                    "fid": fid_i,
                    "host_idx": host_idx,
                    "point_idx": int(ref["point_idx"]),
                    "bearing": bearing,
                    "rho0": 1.0 / depth_along_ray,
                    "ref_ints": np.asarray(ref["intensity"], dtype=np.float64),
                })

        if len(hosts) > max_landmarks:
            step = max(1, len(hosts) // max_landmarks)
            hosts = hosts[::step][:max_landmarks]

        edges = []
        for var_idx, host in enumerate(hosts):
            host_pose = original_poses[host["host_idx"]]
            p_host0 = host["bearing"] / host["rho0"]
            point_w0 = host_pose[:3, :3] @ p_host0 + host_pose[:3, 3]
            target_edges = []
            for target_idx, target_kf in enumerate(keyframes):
                if target_idx == host["host_idx"]:
                    continue
                error = self._mono_patch_error(
                    point_w0,
                    host["ref_ints"],
                    host["host_idx"],
                    target_idx,
                    original_poses[target_idx],
                    keyframes,
                )
                if error is None or error > max_initial_photometric_error:
                    continue
                target_edges.append((var_idx, target_idx))
                if len(target_edges) >= max_targets_per_landmark:
                    break
            edges.extend(target_edges)

        if len(edges) < 10:
            return {
                "ran": False,
                "window": len(keyframes),
                "landmarks": len(hosts),
                "edges": len(edges),
                "cost_before": 0.0,
                "cost_after": 0.0,
                "updated": 0,
                "median_abs_log_depth_update": 0.0,
                "max_abs_log_depth_update": 0.0,
            }

        def point_for_host(host, log_rho_delta):
            rho = host["rho0"] * np.exp(log_rho_delta)
            p_host = host["bearing"] / max(rho, 1e-9)
            host_pose = original_poses[host["host_idx"]]
            return host_pose[:3, :3] @ p_host + host_pose[:3, 3]

        edges_by_landmark = [[] for _ in hosts]
        for var_idx, target_idx in edges:
            edges_by_landmark[var_idx].append(target_idx)

        def landmark_residuals(var_idx, log_rho_delta):
            host = hosts[var_idx]
            point_w = point_for_host(host, log_rho_delta)
            residuals = []
            for target_idx in edges_by_landmark[var_idx]:
                res = self._mono_patch_residual(
                    point_w,
                    host["ref_ints"],
                    host["host_idx"],
                    target_idx,
                    original_poses[target_idx],
                    keyframes,
                )
                if res is None:
                    residuals.extend([0.0] * PATTERN_SIZE)
                else:
                    residuals.extend(res.tolist())
            return np.asarray(residuals, dtype=np.float64)

        def robust_abs_cost(residuals):
            if len(residuals) == 0:
                return 0.0
            abs_res = np.abs(residuals)
            delta = 10.0
            loss = np.where(abs_res <= delta, 0.5 * abs_res * abs_res, delta * (abs_res - 0.5 * delta))
            return float(np.mean(loss))

        r0 = np.concatenate([landmark_residuals(i, 0.0) for i in range(len(hosts)) if edges_by_landmark[i]])
        cost_before = float(np.mean(np.abs(r0))) if len(r0) else 0.0
        optimized_log_rho = np.zeros(len(hosts), dtype=np.float64)
        max_abs_log_depth_update = float(max(1e-3, max_abs_log_depth_update))
        bounds = (-max_abs_log_depth_update, max_abs_log_depth_update)
        for var_idx in range(len(hosts)):
            if not edges_by_landmark[var_idx]:
                continue

            def objective(log_rho_delta):
                return robust_abs_cost(landmark_residuals(var_idx, log_rho_delta))

            opt = minimize_scalar(
                objective,
                bounds=bounds,
                method="bounded",
                options={"maxiter": max(3, int(max_iters))},
            )
            if np.isfinite(opt.x) and np.isfinite(opt.fun) and opt.fun <= objective(0.0):
                optimized_log_rho[var_idx] = float(opt.x)

        r_after = np.concatenate([
            landmark_residuals(i, optimized_log_rho[i]) for i in range(len(hosts)) if edges_by_landmark[i]
        ])
        cost_after = float(np.mean(np.abs(r_after))) if len(r_after) else 0.0

        refined_points = {}
        for var_idx, host in enumerate(hosts):
            refined_points[host["fid"]] = point_for_host(host, optimized_log_rho[var_idx])

        abs_updates = np.abs(optimized_log_rho)
        updated = int(np.sum(abs_updates > 1e-4))

        for kf in keyframes:
            if len(kf.landmark_ids) > 0:
                for point_idx, fid in enumerate(kf.landmark_ids):
                    point_w = refined_points.get(int(fid))
                    if point_w is not None:
                        kf.points_w[point_idx] = point_w
            for obs in kf.observations.values():
                point_w = refined_points.get(int(obs.landmark_id))
                if point_w is not None:
                    obs.point_w = point_w.copy()

        if mono_map is not None:
            mono_map.sync_landmark_tracks_from_keyframes()

        return {
            "ran": True,
            "window": len(keyframes),
            "landmarks": len(hosts),
            "edges": len(edges),
            "cost_before": cost_before,
            "cost_after": cost_after,
            "updated": updated,
            "median_abs_log_depth_update": float(np.median(abs_updates)) if len(abs_updates) else 0.0,
            "max_abs_log_depth_update": float(np.max(abs_updates)) if len(abs_updates) else 0.0,
        }

    def optimize_mono_geometric_pose_window(self, mono_window, max_iters=5, max_obs_per_keyframe=300):
        """
        Pose-only geometric BA for mono keyframes with fixed Sparse3D world points.
        Observation residuals are reprojection errors in pixels; keyframe 0 is fixed.
        """
        keyframes, mono_map = self._mono_window(mono_window)
        if len(keyframes) < 2:
            return {"ran": False, "window": len(keyframes), "edges": 0}

        n_opt = len(keyframes) - 1
        original_poses = [kf.T_W_C.copy() for kf in keyframes]
        edges = []
        for kf_idx, kf in enumerate(keyframes):
            observations = list(kf.observations.values())
            if len(observations) > max_obs_per_keyframe:
                step = max(1, len(observations) // max_obs_per_keyframe)
                observations = observations[::step]
            for obs in observations:
                point_w = self._mono_observation_point(mono_map, obs)
                if not self._visible_mask(np.asarray([point_w]), original_poses[kf_idx], kf.image.shape)[0]:
                    continue
                sigma = np.sqrt(np.maximum(np.diag(obs.covariance), 1e-9))
                edges.append((kf_idx, point_w.copy(), obs.pixel.copy(), sigma))

        if len(edges) < 10:
            return {"ran": False, "window": len(keyframes), "edges": len(edges)}

        x0 = np.zeros(n_opt * 6)

        def current_poses(x):
            poses = [original_poses[0]]
            for i in range(n_opt):
                delta = x[i * 6:(i + 1) * 6]
                poses.append(original_poses[i + 1] @ sp.SE3.exp(delta).matrix())
            return poses

        def residual_fn(x):
            poses = current_poses(x)
            residuals = []
            for kf_idx, point_w, pixel, sigma in edges:
                T_C_W = np.linalg.inv(poses[kf_idx])
                p = T_C_W[:3, :3] @ point_w + T_C_W[:3, 3]
                if p[2] <= 0.01:
                    residuals.extend([0.0, 0.0])
                    continue
                pred = np.array([
                    self.fx * p[0] / p[2] + self.cx,
                    self.fy * p[1] / p[2] + self.cy,
                ])
                residuals.extend(((pred - pixel) / sigma).tolist())
            return np.asarray(residuals, dtype=np.float64)

        res = least_squares(
            residual_fn,
            x0,
            bounds=(-0.05, 0.05),
            method="trf",
            loss="huber",
            f_scale=3.0,
            max_nfev=max_iters,
        )

        for i in range(n_opt):
            delta = res.x[i * 6:(i + 1) * 6]
            keyframes[i + 1].T_W_C = original_poses[i + 1] @ sp.SE3.exp(delta).matrix()

        return {"ran": True, "window": len(keyframes), "edges": len(edges)}

    def optimize_mono_joint_window(self, mono_window, max_nfev=40, max_points=200, prior_reg=1e-6):
        """Joint local BA over window poses + covisible points (Phase 3A).

        Optimizes keyframe poses (oldest fixed as gauge) and the positions of
        landmarks observed in >=2 window keyframes, with reprojection likelihood +
        a per-point Gaussian prior from the covariance transferred at promotion
        (well-triangulated points -> tight prior, fresh points -> loose). scipy TRF
        with a sparse Jacobian pattern.
        """
        from collections import defaultdict
        from scipy.sparse import lil_matrix

        keyframes, mono_map = self._mono_window(mono_window)
        if len(keyframes) < 2:
            return {"ran": False, "window": len(keyframes), "points": 0, "edges": 0}

        lm_obs = defaultdict(list)
        for i, kf in enumerate(keyframes):
            for fid, obs in kf.observations.items():
                if int(fid) in mono_map.landmarks:
                    sigma = np.sqrt(np.maximum(np.diag(np.asarray(obs.covariance, dtype=np.float64)), 1e-9))
                    lm_obs[int(fid)].append((i, np.asarray(obs.pixel, dtype=np.float64), sigma))

        covis = [fid for fid, ob in lm_obs.items() if len(ob) >= 2]
        if len(covis) == 0:
            return {"ran": False, "window": len(keyframes), "points": 0, "edges": 0}
        if len(covis) > max_points:
            sel = np.linspace(0, len(covis) - 1, max_points).astype(int)
            covis = [covis[k] for k in sel]
        point_idx = {fid: j for j, fid in enumerate(covis)}
        n_pts = len(covis)
        n_opt = len(keyframes) - 1

        original_poses = [kf.T_W_C.copy() for kf in keyframes]
        point0 = np.asarray([mono_map.landmarks[fid].point_w for fid in covis], dtype=np.float64)
        priors = point0.copy()
        prior_L = np.zeros((n_pts, 3, 3))
        for j, fid in enumerate(covis):
            P = np.asarray(mono_map.landmarks[fid].point_covariance, dtype=np.float64)
            try:
                prior_L[j] = np.linalg.cholesky(np.linalg.inv(P + np.eye(3) * prior_reg))
            except np.linalg.LinAlgError:
                prior_L[j] = np.eye(3) * np.sqrt(prior_reg)

        edges = [(i, point_idx[fid], pixel, sigma) for fid in covis for (i, pixel, sigma) in lm_obs[fid]]
        n_repro = len(edges)
        n_res = 2 * n_repro + 3 * n_pts
        n_var = 6 * n_opt + 3 * n_pts

        def unpack(x):
            poses = [original_poses[0]]
            for i in range(n_opt):
                poses.append(original_poses[i + 1] @ sp.SE3.exp(x[i * 6:(i + 1) * 6]).matrix())
            return poses, x[6 * n_opt:].reshape(n_pts, 3)

        def residual_fn(x):
            poses, points = unpack(x)
            r = np.zeros(n_res)
            for e, (i, j, pixel, sigma) in enumerate(edges):
                T_C_W = np.linalg.inv(poses[i])
                p = T_C_W[:3, :3] @ points[j] + T_C_W[:3, 3]
                if p[2] > 0.01:
                    pred = np.array([self.fx * p[0] / p[2] + self.cx, self.fy * p[1] / p[2] + self.cy])
                    r[2 * e:2 * e + 2] = (pred - pixel) / sigma
            base = 2 * n_repro
            for j in range(n_pts):
                r[base + 3 * j:base + 3 * j + 3] = prior_L[j].T @ (points[j] - priors[j])
            return r

        S = lil_matrix((n_res, n_var), dtype=bool)
        for e, (i, j, _pixel, _sigma) in enumerate(edges):
            if i >= 1:
                S[2 * e:2 * e + 2, (i - 1) * 6:i * 6] = True
            S[2 * e:2 * e + 2, 6 * n_opt + 3 * j:6 * n_opt + 3 * j + 3] = True
        base = 2 * n_repro
        for j in range(n_pts):
            S[base + 3 * j:base + 3 * j + 3, 6 * n_opt + 3 * j:6 * n_opt + 3 * j + 3] = True

        x0 = np.zeros(n_var)
        x0[6 * n_opt:] = point0.reshape(-1)
        lb = np.full(n_var, -np.inf)
        ub = np.full(n_var, np.inf)
        lb[:6 * n_opt] = -0.2
        ub[:6 * n_opt] = 0.2
        cost_before = 0.5 * float(np.sum(residual_fn(x0) ** 2))
        res = least_squares(
            residual_fn, x0, jac_sparsity=S, bounds=(lb, ub),
            method="trf", loss="huber", f_scale=3.0, max_nfev=max_nfev,
        )

        poses, points = unpack(res.x)
        for i in range(n_opt):
            keyframes[i + 1].T_W_C = poses[i + 1]
        for j, fid in enumerate(covis):
            mono_map.update_landmark_point(fid, points[j])
        return {
            "ran": True, "window": len(keyframes), "points": n_pts, "edges": n_repro,
            "cost_before": cost_before, "cost_after": float(res.cost),
        }

    def _visible_mask(self, pts_w, T_W_C, image_shape):
        if len(pts_w) == 0:
            return np.zeros((0,), dtype=bool)
        T_C_W = np.linalg.inv(T_W_C)
        pts_c = pts_w @ T_C_W[:3, :3].T + T_C_W[:3, 3]
        z = pts_c[:, 2]
        valid = z > 0.01
        u = self.fx * pts_c[:, 0] / np.clip(z, 0.01, None) + self.cx
        v = self.fy * pts_c[:, 1] / np.clip(z, 0.01, None) + self.cy
        h, w = image_shape
        return valid & (u >= 1) & (u < w - 2) & (v >= 1) & (v < h - 2)

    def _mono_patch_error(self, point_w, ref_ints, host_idx, target_idx, T_W_C_target, keyframes):
        residual = self._mono_patch_residual(point_w, ref_ints, host_idx, target_idx, T_W_C_target, keyframes)
        if residual is None:
            return None
        return float(np.mean(np.abs(residual)))

    def _mono_patch_residual(self, point_w, ref_ints, host_idx, target_idx, T_W_C_target, keyframes):
        target_kf = keyframes[target_idx]
        T_C_W = np.linalg.inv(T_W_C_target)
        p = T_C_W[:3, :3] @ point_w + T_C_W[:3, 3]
        if p[2] <= 0.01:
            return None
        u = self.fx * p[0] / p[2] + self.cx
        v = self.fy * p[1] / p[2] + self.cy
        h, w = target_kf.image.shape
        if u < 1 or u > w - 2 or v < 1 or v > h - 2:
            return None
        u_pat = u + PATTERN_DX
        v_pat = v + PATTERN_DY
        target = nd.map_coordinates(target_kf.image, [v_pat, u_pat], order=1)
        exp_diff = np.exp(target_kf.affine_a - keyframes[host_idx].affine_a)
        bias_diff = target_kf.affine_b - keyframes[host_idx].affine_b
        predicted = exp_diff * ref_ints + bias_diff
        return predicted - target
