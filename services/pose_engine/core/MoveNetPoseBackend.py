import numpy as np
from services.pose_engine.exercises.ExerciseDetector import ExcerciseDetector
from services.pose_engine.core.BackendInterface import Landmark, PoseBackend
from typing import List

class MoveNetPoseBackend(PoseBackend):
    def __init__(self, excecise_detector: ExcerciseDetector | None = None):
        pass

    def process(self, frame_rgb: np.ndarray) -> List[Landmark]:
        pass

    def draw(self, frame_rgb: np.ndarray, landmarks: List[Landmark]) -> np.ndarray:
        pass