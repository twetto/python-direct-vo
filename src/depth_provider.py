from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DepthEstimate:
    depth: np.ndarray
    confidence: Optional[np.ndarray] = None
    source: str = "unknown"


class DepthProvider(ABC):
    """Abstraction that keeps stereo depth out of the core monocular pipeline."""

    @abstractmethod
    def compute(self, left_img: np.ndarray, right_img: Optional[np.ndarray] = None) -> DepthEstimate:
        raise NotImplementedError


class FixedDepthProvider(DepthProvider):
    def __init__(self, depth: np.ndarray, confidence: Optional[np.ndarray] = None, source: str = "fixed"):
        self.depth = depth
        self.confidence = confidence
        self.source = source

    def compute(self, left_img: np.ndarray, right_img: Optional[np.ndarray] = None) -> DepthEstimate:
        return DepthEstimate(self.depth.copy(), None if self.confidence is None else self.confidence.copy(), self.source)


class StereoDepthProvider(DepthProvider):
    def __init__(self, focal_length: float, baseline: float, stereo_fn):
        self.focal_length = focal_length
        self.baseline = baseline
        self.stereo_fn = stereo_fn

    def compute(self, left_img: np.ndarray, right_img: Optional[np.ndarray] = None) -> DepthEstimate:
        if right_img is None:
            raise ValueError("right_img is required for stereo depth")
        disparity, depth = self.stereo_fn(left_img, right_img, self.focal_length, self.baseline)
        confidence = (disparity > 0).astype(np.float32)
        return DepthEstimate(depth, confidence, "stereo")

