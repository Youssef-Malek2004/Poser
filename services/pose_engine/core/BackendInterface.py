from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


class PoseBackend(ABC):
    """Abstract base for any pose model (MediaPipe, MoveNet, etc.)."""

    @abstractmethod
    def process(self, frame_rgb: np.ndarray) -> List[Landmark]:
        """Run pose estimation on one RGB frame and return a list of landmarks."""
        ...

    @abstractmethod
    def draw(self, frame_rgb: np.ndarray, landmarks: List[Landmark]) -> np.ndarray:
        """Return a BGR image with landmarks drawn (for debugging / saving)."""
        ...
