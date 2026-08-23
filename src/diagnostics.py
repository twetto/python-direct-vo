import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='[DEBUG-MONITOR] %(message)s')
logger = logging.getLogger('vo_monitor')

class VOMonitor:
    def __init__(self):
        self.last_T_W_C = None
        
    def check_pose_jump(self, frame_id, stage, T_W_C, threshold=0.5):
        if self.last_T_W_C is not None:
            dt = np.linalg.norm(T_W_C[:3, 3] - self.last_T_W_C[:3, 3])
            if dt > threshold:
                logger.error(f"FRAME {frame_id} | POSE JUMP DETECTED in {stage}! dt = {dt:.3f} meters")
                logger.error(f"Old translation: {self.last_T_W_C[:3, 3]}")
                logger.error(f"New translation: {T_W_C[:3, 3]}")
        
    def set_baseline(self, T_W_C):
        self.last_T_W_C = T_W_C.copy()

    def check_landmark_consistency(self, frame_id, stage, pts, ints, a, b):
        if len(pts) != len(ints) or len(pts) != len(a) or len(pts) != len(b):
            logger.error(f"FRAME {frame_id} | ARRAY SIZE MISMATCH in {stage}!")
            logger.error(f"pts: {len(pts)}, ints: {len(ints)}, a: {len(a)}, b: {len(b)}")
            
        if len(pts) > 0:
            nan_pts = np.sum(np.isnan(pts))
            if nan_pts > 0:
                logger.error(f"FRAME {frame_id} | NANs DETECTED in {stage}! {nan_pts} NaN values in points.")

    def check_landmark_jump(self, frame_id, pts_before, pts_after, threshold=0.2):
        if len(pts_before) == len(pts_after) and len(pts_before) > 0:
            diffs = np.linalg.norm(pts_after - pts_before, axis=1)
            mean_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            if mean_diff > threshold or max_diff > threshold * 5:
                logger.error(f"FRAME {frame_id} | LANDMARK JUMP DETECTED during BA! Mean shift: {mean_diff:.4f}m, Max shift: {max_diff:.4f}m")

    def check_alignment_jump(self, frame_id, s, R, t):
        if not hasattr(self, 'last_s'):
            self.last_s = s
            self.last_R = R
            self.last_t = t
        else:
            ds = abs(s - self.last_s)
            dt = np.linalg.norm(t - self.last_t)
            dR = np.linalg.norm(R - self.last_R)
            if ds > 0.05 or dt > 0.1 or dR > 0.05:
                logger.warning(f"FRAME {frame_id} | ALIGNMENT JUMP! ds: {ds:.4f}, dt: {dt:.4f}, dR: {dR:.4f}")
            self.last_s = s
            self.last_R = R
            self.last_t = t
