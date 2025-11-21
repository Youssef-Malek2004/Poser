from typing import Any, List

from services.pose_engine.core.BackendInterface import Landmark
from services.pose_engine.exercises.ExerciseDetector import ExerciseDetector

class LatPullDownDetector(ExerciseDetector):
    def __init__(self):
        super().__init__()

    def in_start_position(self, landmarks: List[Landmark]) -> Any:
        pass
    def update_reps(self, landmarks: List[Landmark]) -> int:
        pass