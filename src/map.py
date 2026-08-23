import numpy as np
import cv2

class Keyframe:
    def __init__(self, kf_id, img, depth, T_W_C, K, a=0.0, b=0.0, mask=None, num_features=2000):
        self.kf_id = kf_id
        self.T_W_C = T_W_C.copy()
        self.a = a
        self.b = b
        
        # 1. Detect strong corners in the image
        corners = cv2.goodFeaturesToTrack(
            img, 
            maxCorners=num_features, 
            qualityLevel=0.1, 
            minDistance=10,
            mask=mask
        )
        
        if corners is not None:
            corners = corners.reshape(-1, 2)
            u = np.clip(np.round(corners[:, 0]).astype(int), 0, img.shape[1]-1)
            v = np.clip(np.round(corners[:, 1]).astype(int), 0, img.shape[0]-1)
            
            # 2. Filter corners by valid depth from the stereo map
            z = depth[v, u]
            valid = (z > 0.5) & (z < 10.0)
            
            # Filter out corners too close to the boundary to safely extract a 5-pixel pattern
            h, w = img.shape
            valid = valid & (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
            
            u = u[valid]
            v = v[valid]
            z = z[valid]
            
            # 3. Store RAW Intensities using a 5-pixel DSO-style pattern
            # Pattern: Center, Right, Left, Down, Up
            dx = np.array([0, 1, -1, 0, 0])
            dy = np.array([0, 0, 0, 1, -1])
            
            self.intensities = np.zeros((len(u), 5), dtype=np.float32)
            for i in range(5):
                self.intensities[:, i] = img[v + dy[i], u + dx[i]].astype(np.float32)
            
            # 3. Back-project to the Camera Frame
            cx, cy = K[0, 2], K[1, 2]
            fx, fy = K[0, 0], K[1, 1]
            
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            self.pts_3d_c = np.stack([x, y, z], axis=1) # (N, 3)
            
            # 4. Transform to the World Frame using the Keyframe's pose
            R_W_C = self.T_W_C[:3, :3]
            t_W_C = self.T_W_C[:3, 3]
            self.pts_3d_w = (R_W_C @ self.pts_3d_c.T).T + t_W_C
        else:
            self.pts_3d_w = np.empty((0, 3), dtype=np.float64)
            self.pts_3d_c = np.empty((0, 3), dtype=np.float64)
            self.intensities = np.empty((0, 5), dtype=np.float32)
            
        self.img = img.copy()

class Map:
    def __init__(self, K, window_size=7):
        self.K = K
        self.keyframes = []
        self.window_size = window_size
        self.kf_counter = 0
        
    def add_keyframe(self, img, depth, T_W_C, a=0.0, b=0.0):
        """Creates a new Keyframe, extracting NEW landmarks, and adds it to the active window."""
        
        # Prevent initialization of duplicate landmarks by masking out the existing points!
        mask = np.ones(img.shape, dtype=np.uint8) * 255
        
        pts_w, _, _, _, _ = self.get_window_points()
        if len(pts_w) > 0:
            T_C_W = np.linalg.inv(T_W_C)
            R, t = T_C_W[:3, :3], T_C_W[:3, 3]
            pts_c = pts_w @ R.T + t
            
            Z = pts_c[:, 2]
            valid = Z > 0.01
            
            if np.any(valid):
                u = (self.K[0,0] * pts_c[valid, 0] / Z[valid] + self.K[0,2]).astype(int)
                v = (self.K[1,1] * pts_c[valid, 1] / Z[valid] + self.K[1,2]).astype(int)
                
                h, w = img.shape
                in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                
                for ui, vi in zip(u[in_bounds], v[in_bounds]):
                    cv2.circle(mask, (ui, vi), radius=15, color=0, thickness=-1)
                    
        kf = Keyframe(self.kf_counter, img, depth, T_W_C, self.K, a, b, mask=mask)
        self.kf_counter += 1
        self.keyframes.append(kf)
        
        # Enforce sliding window marginalization (DSO-style Spatial Redundancy)
        if len(self.keyframes) > self.window_size:
            # Protect the newest 2 frames for tracking stability
            candidates = self.keyframes[:-2]
            
            if not candidates:
                self.keyframes.pop(0)
            else:
                latest_kf = self.keyframes[-1]
                best_drop_idx = -1
                max_redundancy_score = -float('inf')
                
                for i, c_kf in enumerate(candidates):
                    redundancy = 0.0
                    for other_kf in self.keyframes:
                        if other_kf is c_kf:
                            continue
                        dist = np.linalg.norm(c_kf.T_W_C[:3, 3] - other_kf.T_W_C[:3, 3])
                        redundancy += 1.0 / (dist + 1e-5)
                        
                    # Weight by distance to the latest frame. 
                    dist_to_latest = np.linalg.norm(c_kf.T_W_C[:3, 3] - latest_kf.T_W_C[:3, 3])
                    score = redundancy * np.sqrt(dist_to_latest)
                    
                    if score > max_redundancy_score:
                        max_redundancy_score = score
                        best_drop_idx = i
                
                # --- Point Transfer Logic ---
                drop_idx = best_drop_idx
                dropped_kf = self.keyframes[drop_idx]
                
                T_C_W_latest = np.linalg.inv(latest_kf.T_W_C)
                R, t = T_C_W_latest[:3, :3], T_C_W_latest[:3, 3]
                
                pts_c = dropped_kf.pts_3d_w @ R.T + t
                Z = pts_c[:, 2]
                valid = Z > 0.01
                
                if np.any(valid):
                    u = (self.K[0,0] * pts_c[valid, 0] / Z[valid] + self.K[0,2])
                    v = (self.K[1,1] * pts_c[valid, 1] / Z[valid] + self.K[1,2])
                    
                    h, w = latest_kf.img.shape
                    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                    
                    rescue_mask = np.zeros_like(valid, dtype=bool)
                    rescue_mask[np.where(valid)[0][in_bounds]] = True
                    
                    if np.sum(rescue_mask) > 0:
                        latest_kf.pts_3d_w = np.vstack([latest_kf.pts_3d_w, dropped_kf.pts_3d_w[rescue_mask]])
                        latest_kf.pts_3d_c = np.vstack([latest_kf.pts_3d_c, pts_c[rescue_mask]])
                        
                        # Apply relative affine mapping to transfer RAW intensities into the new host's exposure space!
                        # I_latest = exp(a_latest - a_dropped) * I_dropped + (b_latest - b_dropped)
                        exp_diff = np.exp(latest_kf.a - dropped_kf.a)
                        bias_diff = latest_kf.b - dropped_kf.b
                        transferred_ints = exp_diff * dropped_kf.intensities[rescue_mask] + bias_diff
                        
                        latest_kf.intensities = np.concatenate([latest_kf.intensities, transferred_ints])
                
                self.keyframes.pop(drop_idx)
            
        return len(kf.pts_3d_w)
        
    def get_window_points(self):
        """Returns the aggregated point cloud, intensities, exposure states, and source KF IDs."""
        if not self.keyframes:
            return np.empty((0, 3)), np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,), dtype=int)
            
        pts = np.vstack([kf.pts_3d_w for kf in self.keyframes])
        ints = np.concatenate([kf.intensities for kf in self.keyframes])
        a_vals = np.concatenate([np.full(len(kf.intensities), kf.a) for kf in self.keyframes])
        b_vals = np.concatenate([np.full(len(kf.intensities), kf.b) for kf in self.keyframes])
        kf_ids = np.concatenate([np.full(len(kf.intensities), kf.kf_id, dtype=int) for kf in self.keyframes])
        
        return pts, ints, a_vals, b_vals, kf_ids
