import numpy as np
import cv2

import scipy.ndimage as nd
import sophuspy as sp

class DirectTracker:
    def __init__(self, K):
        self.K = K
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        
    def track_map(self, img_cur, pts_3d_w, intensities_ref, a_ref, b_ref, T_C_W_guess, a_guess=0.0, b_guess=0.0, max_iters=15):
        """
        Run Gauss-Newton optimization to find the SE(3) pose mapping World to Camera (T_C_W)
        and affine brightness parameters (a, b).
        Returns the optimized T_C_W matrix, a boolean mask of visible points, and the new (a, b).
        """
        img_cur = img_cur.astype(np.float32)
        
        # Compute image gradients using Scharr for the current frame
        grad_x = cv2.Scharr(img_cur, cv2.CV_32F, 1, 0) / 16.0
        grad_y = cv2.Scharr(img_cur, cv2.CV_32F, 0, 1) / 16.0
        
        T = sp.SE3(T_C_W_guess)
        a = a_guess
        b = b_guess
        
        for iter_ in range(max_iters):
            R = T.rotationMatrix()
            t = T.translation()
            
            # Transform World points to Current Camera Frame
            pts_c = pts_3d_w @ R.T + t
            X, Y, Z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
            
            valid = Z > 0.01
            if not np.any(valid):
                break
                
            # Project to 2D
            u = self.fx * X[valid] / Z[valid] + self.cx
            v = self.fy * Y[valid] / Z[valid] + self.cy
            
            # Check bounds
            h, w = img_cur.shape
            in_bounds = (u >= 1) & (u < w - 2) & (v >= 1) & (v < h - 2)
            
            idx = np.where(valid)[0][in_bounds]
            if len(idx) < 50:
                break
                
            u, v = u[in_bounds], v[in_bounds]
            X, Y, Z = X[idx], Y[idx], Z[idx]
            pts_ref_i = intensities_ref[idx] # Shape: [N, 5]
            a_ref_i = a_ref[idx]
            b_ref_i = b_ref[idx]
            
            # Interpolate brightness and gradients for the 5-pixel pattern
            dx = np.array([0, 1, -1, 0, 0])
            dy = np.array([0, 0, 0, 1, -1])
            u_pat = (u[:, None] + dx).flatten()
            v_pat = (v[:, None] + dy).flatten()
            
            cur_intensities = nd.map_coordinates(img_cur, [v_pat, u_pat], order=1).reshape(-1, 5)
            ix = nd.map_coordinates(grad_x, [v_pat, u_pat], order=1).reshape(-1, 5)
            iy = nd.map_coordinates(grad_y, [v_pat, u_pat], order=1).reshape(-1, 5)
            
            # DSO Affine Brightness Residuals (Relative Exposure)
            exp_diff = np.exp(a - a_ref_i)[:, None]
            bias_diff = (b - b_ref_i)[:, None]
            residuals = cur_intensities - (exp_diff * pts_ref_i + bias_diff)
            
            # Jacobians (8-DOF) broadcast over the 5-pixel pattern
            Z_inv = 1.0 / Z
            J_geo_x = ix * (self.fx * Z_inv)[:, None]
            J_geo_y = iy * (self.fy * Z_inv)[:, None]
            J_geo_z = -(J_geo_x * X[:, None] + J_geo_y * Y[:, None]) * Z_inv[:, None]
            
            J = np.zeros((len(u) * 5, 8))
            J[:, 0] = J_geo_x.flatten()
            J[:, 1] = J_geo_y.flatten()
            J[:, 2] = J_geo_z.flatten()
            J[:, 3] = (-J_geo_y * Z[:, None] + J_geo_z * Y[:, None]).flatten()
            J[:, 4] = ( J_geo_x * Z[:, None] - J_geo_z * X[:, None]).flatten()
            J[:, 5] = (-J_geo_x * Y[:, None] + J_geo_y * X[:, None]).flatten()
            # Affine Photometric Jacobians (Relative)
            J[:, 6] = -(exp_diff * pts_ref_i).flatten()
            J[:, 7] = -1.0
            
            # --- Robust Loss (Huber Weighting) ---
            delta_thresh = 15.0
            residuals_flat = residuals.flatten()
            abs_res = np.abs(residuals_flat)
            weights = np.where(abs_res <= delta_thresh, 1.0, delta_thresh / (abs_res + 1e-6))
            
            # Weight Jacobians and Residuals
            sqrt_w = np.sqrt(weights)
            J_w = J * sqrt_w[:, None]
            res_w = residuals_flat * sqrt_w
            
            H = J_w.T @ J_w
            b_vec = -J_w.T @ res_w
            
            # --- Tikhonov Regularization for Affine Drift ---
            # Instead of heavily penalizing the absolute value of 'a' and 'b' towards 0 
            # (which ruins tracking if lighting actually changes), we only damp the UPDATE step
            lambda_pose = 1e-4
            lambda_affine = 1e-2
            H[np.diag_indices(6)] += lambda_pose
            H[6, 6] += lambda_affine
            H[7, 7] += lambda_affine
            
            try:
                update = np.linalg.solve(H, b_vec)
            except np.linalg.LinAlgError:
                break
                
            T = sp.SE3.exp(update[:6]) * T
            a += update[6]
            b += update[7]
            
            if np.linalg.norm(update[:6]) < 1e-4:
                break
                
        # Final pass to compute inliers based on final optimized pose
        R = T.rotationMatrix()
        t = T.translation()
        pts_c = pts_3d_w @ R.T + t
        X, Y, Z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
        
        valid = Z > 0.01
        if np.any(valid):
            u = self.fx * X[valid] / Z[valid] + self.cx
            v = self.fy * Y[valid] / Z[valid] + self.cy
            
            h, w = img_cur.shape
            in_bounds = (u >= 1) & (u < w - 2) & (v >= 1) & (v < h - 2)
            idx_in_frustum = np.where(valid)[0][in_bounds]
            
            u, v = u[in_bounds], v[in_bounds]
            
            dx = np.array([0, 1, -1, 0, 0])
            dy = np.array([0, 0, 0, 1, -1])
            u_pat = (u[:, None] + dx).flatten()
            v_pat = (v[:, None] + dy).flatten()
            
            cur_ints_flat = nd.map_coordinates(img_cur, [v_pat, u_pat], order=1)
            cur_ints = cur_ints_flat.reshape(-1, 5)
            
            exp_diff = np.exp(a - a_ref[idx_in_frustum])[:, None]
            bias_diff = (b - b_ref[idx_in_frustum])[:, None]
            predicted_ints = exp_diff * intensities_ref[idx_in_frustum] + bias_diff
            
            res = np.abs(predicted_ints - cur_ints)
            
            # Mean error across the 5-point pattern must be < 30 to reject partial occlusions
            good_res = np.mean(res, axis=1) < 40.0
            
            final_inliers = np.zeros_like(valid, dtype=bool)
            final_inliers[idx_in_frustum[good_res]] = True
        else:
            final_inliers = np.zeros_like(valid, dtype=bool)
        
        return T.matrix(), final_inliers, a, b

