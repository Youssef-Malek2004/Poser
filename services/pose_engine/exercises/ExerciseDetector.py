from abc import ABC, abstractmethod
from typing import List, Any
from services.pose_engine.core.BackendInterface import Landmark


class ExerciseDetector(ABC):
    def __init__(self):
        # Internal state machine for rep counting
        # "unknown" -> no position yet
        # "up"      -> start/top position
        # "down"    -> end/bottom position
        self._rep_state: str = "unknown"
        self.reps: int = 0

    @abstractmethod
    def in_start_position(self, landmarks: List[Landmark]) -> Any:
        """
        Return either:
          - bool  (True if in start position)
          - or a dict with at least {"is_start": bool}
        """
        ...

    @abstractmethod
    def in_end_position(self, landmarks: List[Landmark]) -> Any:
        """
        Return either:
          - bool  (True if in end position)
          - or a dict with at least {"is_end": bool}
        """
        ...

    def update_reps(self, landmarks: List[Landmark]) -> int:
        """
        Generic rep counter.

        One rep = START (up) -> END (down) -> START (up) again.
        Call this once per frame with current landmarks.
        Stores count in self.reps and also returns it.
        """
        start_info = self.in_start_position(landmarks)
        end_info = self.in_end_position(landmarks)

        if isinstance(start_info, dict):
            is_start = bool(start_info.get("is_start"))
        else:
            is_start = bool(start_info)

        if isinstance(end_info, dict):
            is_end = bool(end_info.get("is_end"))
        else:
            is_end = bool(end_info)

        # --- State machine ---
        if self._rep_state in ("unknown", "up"):
            if is_end:
                self._rep_state = "down"
            elif is_start:
                self._rep_state = "up"

        elif self._rep_state == "down":
            if is_start:
                self.reps += 1
                self._rep_state = "up"
            elif is_end:
                self._rep_state = "down"

        return self.reps
