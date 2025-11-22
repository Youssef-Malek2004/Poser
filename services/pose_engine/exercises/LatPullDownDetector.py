from typing import List, Optional
import numpy as np

from services.pose_engine.core.BackendInterface import Landmark
from services.pose_engine.exercises.ExerciseDetector import ExerciseDetector
from services.pose_engine.core.joints import BodyJoint


class LatPullDownDetector(ExerciseDetector):
    def __init__(
        self,
        start_elbow_angle: float = 160.0,
        end_elbow_angle: float = 50.0,
        elbow_tolerance: float = 20.0,
        min_visibility: float = 0.6,
    ):
        """
        Lat Pull Down Detector
        
        Start position: Arms extended overhead (elbows straight)
        End position: Bar pulled down to chest (elbows bent ~90 degrees)
        
        Args:
            start_elbow_angle: Expected elbow angle when arms are extended overhead
            end_elbow_angle: Expected elbow angle when bar is at chest level
            elbow_tolerance: Tolerance for angle detection
            min_visibility: Minimum visibility threshold for landmarks
        """
        super().__init__()
        
        self.min_visibility = min_visibility
        self.start_elbow_min = start_elbow_angle - elbow_tolerance
        self.end_elbow_max = end_elbow_angle + elbow_tolerance
        self.end_elbow_min = end_elbow_angle - elbow_tolerance

    def _get_point(
        self,
        landmarks: List[Landmark],
        joint: BodyJoint,
    ) -> Optional[np.ndarray]:
        """Extract 2D point from landmarks for a given joint."""
        idx = int(joint)
        if idx >= len(landmarks):
            return None

        lm = landmarks[idx]
        if lm.visibility is not None and lm.visibility < self.min_visibility:
            return None

        return np.array([lm.x, lm.y], dtype=np.float32)

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
        """
        Calculate angle ABC (with B as the vertex) in degrees.
        """
        if a is None or b is None or c is None:
            return None

        ba = a - b
        bc = c - b

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return None

        cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_angle)))
    
    @staticmethod
    def _vertical_position(wrist: Optional[np.ndarray], shoulder: Optional[np.ndarray]) -> Optional[str]:
        """
        Check if wrist is above or below shoulder (y-axis check).
        Returns 'above', 'below', or None.
        """
        if wrist is None or shoulder is None:
            return None
        
        # In image coordinates, smaller y = higher position
        if wrist[1] < shoulder[1]:
            return 'above'
        return 'below'

    def in_start_position(self, landmarks: List[Landmark]):
        """
        Start position: Arms extended overhead
        - Elbows should be relatively straight (high angle)
        - Wrists should be above shoulders
        """
        # Get key joints
        l_shoulder = self._get_point(landmarks, BodyJoint.LEFT_SHOULDER)
        r_shoulder = self._get_point(landmarks, BodyJoint.RIGHT_SHOULDER)
        
        l_elbow = self._get_point(landmarks, BodyJoint.LEFT_ELBOW)
        r_elbow = self._get_point(landmarks, BodyJoint.RIGHT_ELBOW)
        
        l_wrist = self._get_point(landmarks, BodyJoint.LEFT_WRIST)
        r_wrist = self._get_point(landmarks, BodyJoint.RIGHT_WRIST)

        # Calculate elbow angles: shoulder - elbow - wrist
        left_elbow_angle = self._angle(l_shoulder, l_elbow, l_wrist) if l_elbow is not None else None
        right_elbow_angle = self._angle(r_shoulder, r_elbow, r_wrist) if r_elbow is not None else None

        # Check if wrists are above shoulders
        left_wrist_position = self._vertical_position(l_wrist, l_shoulder)
        right_wrist_position = self._vertical_position(r_wrist, r_shoulder)

        # Validation: at least one side must be valid
        def side_ok_high_angle(a_left, a_right, threshold):
            """Check if at least one arm has elbow angle above threshold."""
            vals = [v for v in (a_left, a_right) if v is not None]
            if not vals:
                return False
            median_angle = np.median(vals)
            return median_angle >= threshold

        elbows_straight = side_ok_high_angle(left_elbow_angle, right_elbow_angle, self.start_elbow_min)
        
        # Check if at least one wrist is above shoulder
        wrists_overhead = left_wrist_position == 'above' or right_wrist_position == 'above'

        result = {
            "is_start": elbows_straight and wrists_overhead,
            "angles": {
                "left_elbow": left_elbow_angle,
                "right_elbow": right_elbow_angle,
            },
            "positions": {
                "left_wrist": left_wrist_position,
                "right_wrist": right_wrist_position,
            },
            "points": {
                "l_shoulder": l_shoulder,
                "r_shoulder": r_shoulder,
                "l_elbow": l_elbow,
                "r_elbow": r_elbow,
                "l_wrist": l_wrist,
                "r_wrist": r_wrist,
            }
        }

        return result

    def in_end_position(self, landmarks: List[Landmark]):
        """
        End position: Bar pulled down to chest level
        - Elbows should be bent (~90 degrees)
        - Wrists should be near or below shoulders
        """
        # Get key joints
        l_shoulder = self._get_point(landmarks, BodyJoint.LEFT_SHOULDER)
        r_shoulder = self._get_point(landmarks, BodyJoint.RIGHT_SHOULDER)
        
        l_elbow = self._get_point(landmarks, BodyJoint.LEFT_ELBOW)
        r_elbow = self._get_point(landmarks, BodyJoint.RIGHT_ELBOW)
        
        l_wrist = self._get_point(landmarks, BodyJoint.LEFT_WRIST)
        r_wrist = self._get_point(landmarks, BodyJoint.RIGHT_WRIST)

        # Calculate elbow angles: shoulder - elbow - wrist
        left_elbow_angle = self._angle(l_shoulder, l_elbow, l_wrist) if l_elbow is not None else None
        right_elbow_angle = self._angle(r_shoulder, r_elbow, r_wrist) if r_elbow is not None else None

        # Check if wrists are below or at shoulder level
        left_wrist_position = self._vertical_position(l_wrist, l_shoulder)
        right_wrist_position = self._vertical_position(r_wrist, r_shoulder)

        # Validation: elbows should be bent (angle in range)
        def side_ok_bent(a_left, a_right, min_angle, max_angle):
            """Check if at least one arm has elbow angle in bent range."""
            vals = [v for v in (a_left, a_right) if v is not None]
            if not vals:
                return False
            median_angle = np.median(vals)
            return min_angle <= median_angle <= max_angle

        elbows_bent = side_ok_bent(left_elbow_angle, right_elbow_angle, 
                                   self.end_elbow_min, self.end_elbow_max)
        
        # Check if at least one wrist is at or below shoulder
        wrists_down = left_wrist_position == 'below' or right_wrist_position == 'below'

        result = {
            "is_end": elbows_bent and wrists_down,
            "angles": {
                "left_elbow": left_elbow_angle,
                "right_elbow": right_elbow_angle,
            },
            "positions": {
                "left_wrist": left_wrist_position,
                "right_wrist": right_wrist_position,
            },
            "points": {
                "l_shoulder": l_shoulder,
                "r_shoulder": r_shoulder,
                "l_elbow": l_elbow,
                "r_elbow": r_elbow,
                "l_wrist": l_wrist,
                "r_wrist": r_wrist,
            }
        }

        return result