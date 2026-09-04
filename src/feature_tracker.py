from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FeatureTrackerConfig:
    max_features: int = 800
    quality_level: float = 0.01
    min_distance: int = 12
    lk_win_size: int = 21
    lk_max_level: int = 3
    ransac_reproj_thresh: float = 2.0
    min_ransac_points: int = 12


class FeatureTracker:
    """Persistent sparse 2D tracker for Sparse3D landmark observations."""

    def __init__(self, config: FeatureTrackerConfig | None = None):
        self.config = config or FeatureTrackerConfig()
        self.prev_img: np.ndarray | None = None
        self.prev_points = np.empty((0, 2), dtype=np.float32)
        self.ids = np.empty((0,), dtype=int)
        self.next_id = 0

    def update(self, image: np.ndarray, exclusion_points: np.ndarray | None = None) -> dict[int, np.ndarray]:
        # Histogram-equalize before KLT/detection: EuRoC auto-exposure swings otherwise
        # drift/drop tracks (the stereo path already does this for its LK, main.py).
        gray = cv2.equalizeHist(image.astype(np.uint8))

        if self.prev_img is None:
            self.prev_img = gray.copy()
            self._detect_new(gray, exclusion_points=exclusion_points)
            return self.observations()

        if len(self.prev_points) == 0:
            self.prev_points = np.empty((0, 2), dtype=np.float32)
            self.ids = np.empty((0,), dtype=int)
        else:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_img,
                gray,
                self.prev_points.reshape(-1, 1, 2),
                None,
                winSize=(self.config.lk_win_size, self.config.lk_win_size),
                maxLevel=self.config.lk_max_level,
            )

            if next_points is None or status is None:
                self.prev_points = np.empty((0, 2), dtype=np.float32)
                self.ids = np.empty((0,), dtype=int)
            else:
                next_points = next_points.reshape(-1, 2)
                good = status.reshape(-1).astype(bool)
                h, w = gray.shape
                good &= (next_points[:, 0] >= 1) & (next_points[:, 0] < w - 1)
                good &= (next_points[:, 1] >= 1) & (next_points[:, 1] < h - 1)

                if np.sum(good) >= self.config.min_ransac_points:
                    _, mask = cv2.findFundamentalMat(
                        self.prev_points[good],
                        next_points[good],
                        cv2.FM_RANSAC,
                        self.config.ransac_reproj_thresh,
                        0.99,
                    )
                    if mask is not None:
                        good_indices = np.where(good)[0]
                        ransac_good = mask.reshape(-1).astype(bool)
                        good[good_indices[~ransac_good]] = False

                self.prev_points = next_points[good].astype(np.float32)
                self.ids = self.ids[good]

        self._detect_new(gray, exclusion_points=exclusion_points)
        self.prev_img = gray.copy()
        return self.observations()

    def observations(self) -> dict[int, np.ndarray]:
        return {
            int(fid): pixel.astype(np.float64)
            for fid, pixel in zip(self.ids, self.prev_points)
        }

    def set_positions(self, positions: dict[int, np.ndarray]) -> int:
        """Overwrite the tracked pixel of given feature ids (used to re-seed KLT).

        Called after the pose refine to re-anchor map-point tracks to their
        pose-refined reprojection, so the next frame's KLT starts from the
        geometrically-correct location instead of its own accumulated edge-drift.
        """
        if len(self.ids) == 0 or not positions:
            return 0
        id_to_idx = {int(fid): i for i, fid in enumerate(self.ids)}
        updated = 0
        for fid, pixel in positions.items():
            idx = id_to_idx.get(int(fid))
            if idx is None:
                continue
            self.prev_points[idx] = np.asarray(pixel, dtype=np.float32)
            updated += 1
        return updated

    def remove_ids(self, feature_ids) -> None:
        if len(self.ids) == 0:
            return
        remove = set(int(fid) for fid in feature_ids)
        if not remove:
            return
        keep = np.array([int(fid) not in remove for fid in self.ids], dtype=bool)
        self.prev_points = self.prev_points[keep]
        self.ids = self.ids[keep]

    def _detect_new(self, image: np.ndarray, exclusion_points: np.ndarray | None = None) -> None:
        need = self.config.max_features - len(self.prev_points)
        if need <= 0:
            return

        mask = np.full(image.shape, 255, dtype=np.uint8)
        for p in self.prev_points:
            cv2.circle(mask, tuple(np.round(p).astype(int)), self.config.min_distance, 0, -1)
        if exclusion_points is not None:
            h, w = image.shape
            for p in np.asarray(exclusion_points, dtype=np.float64).reshape(-1, 2):
                x, y = np.round(p).astype(int)
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(mask, (x, y), self.config.min_distance, 0, -1)

        corners = cv2.goodFeaturesToTrack(
            image,
            maxCorners=need,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            mask=mask,
        )
        if corners is None:
            return

        new_points = corners.reshape(-1, 2).astype(np.float32)
        new_ids = np.arange(self.next_id, self.next_id + len(new_points), dtype=int)
        self.next_id += len(new_points)

        self.prev_points = np.vstack([self.prev_points, new_points]).astype(np.float32)
        self.ids = np.concatenate([self.ids, new_ids])
