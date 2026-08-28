from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class Frame:
    """Lightweight image frame with optional pose state and image pyramid."""

    frame_id: int
    timestamp: float
    image: np.ndarray
    T_W_C: np.ndarray = field(default_factory=lambda: np.eye(4))
    is_keyframe: bool = False
    pyramid: Optional[list[np.ndarray]] = None

    def build_pyramid(self, levels: int = 4) -> list[np.ndarray]:
        if levels < 1:
            raise ValueError("levels must be >= 1")

        pyr = [self.image.astype(np.float32)]
        for _ in range(1, levels):
            pyr.append(cv2.pyrDown(pyr[-1]))
        self.pyramid = pyr
        return pyr


@dataclass
class PosePrior:
    """Pose prediction boundary for IMU, constant-velocity, or future MSCKF priors."""

    T_W_C: np.ndarray
    covariance: Optional[np.ndarray] = None
    source: str = "unknown"

