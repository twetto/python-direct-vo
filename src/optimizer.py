import numpy as np
import cv2
import scipy.ndimage as nd
import sophuspy as sp
from scipy.optimize import least_squares

class PhotometricBA:
    def __init__(self, K):
        self.K = K
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

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
