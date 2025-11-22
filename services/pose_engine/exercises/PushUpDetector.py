from typing import List, Optional
import numpy as np

from services.pose_engine.core.BackendInterface import Landmark
from services.pose_engine.exercises.ExerciseDetector import ExerciseDetector
from services.pose_engine.core.joints import BodyJoint, BodyJointMoveNet


class PushUpStartDetector(ExerciseDetector):
    def __init__(
        self,
        base_angle: float = 160.0,
        tolerance: float = 40.0,
        elbow_tolerance: float = 15.0,
        min_visibility: float = 0.6,
    ):
        super().__init__()

        self.min_visibility = min_visibility
        self.min_elbow_angle = base_angle - elbow_tolerance
        self.min_hip_angle   = base_angle - tolerance
        self.min_knee_angle  = base_angle - tolerance

    def _get_point(
        self,
        landmarks: List[Landmark],
        joint: BodyJoint,
    ) -> Optional[np.ndarray]:
        idx = int(joint)
        if idx >= len(landmarks):
            return None

        lm = landmarks[idx]
        if lm.visibility is not None and lm.visibility < self.min_visibility:
            return None

        # 2D for angle purposes (normalized image coordinates)
        return np.array([lm.x, lm.y], dtype=np.float32)

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
        """
        Angle ABC (with B as the vertex) in degrees.
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

    def in_start_position(self, landmarks: List[Landmark]):
        # Grab key joints (using backend-agnostic joints)
        l_shoulder = self._get_point(landmarks, BodyJointMoveNet.LEFT_SHOULDER)
        r_shoulder = self._get_point(landmarks, BodyJointMoveNet.RIGHT_SHOULDER)

        l_elbow = self._get_point(landmarks, BodyJointMoveNet.LEFT_ELBOW)
        r_elbow = self._get_point(landmarks, BodyJointMoveNet.RIGHT_ELBOW)

        l_wrist = self._get_point(landmarks, BodyJointMoveNet.LEFT_WRIST)
        r_wrist = self._get_point(landmarks, BodyJointMoveNet.RIGHT_WRIST)

        l_hip = self._get_point(landmarks, BodyJointMoveNet.LEFT_HIP)
        r_hip = self._get_point(landmarks, BodyJointMoveNet.RIGHT_HIP)

        l_knee = self._get_point(landmarks, BodyJointMoveNet.LEFT_KNEE)
        r_knee = self._get_point(landmarks, BodyJointMoveNet.RIGHT_KNEE)

        l_ankle = self._get_point(landmarks, BodyJointMoveNet.LEFT_ANKLE)
        r_ankle = self._get_point(landmarks, BodyJointMoveNet.RIGHT_ANKLE)

        # Elbow angles: shoulder - elbow - wrist
        left_elbow_angle = self._angle(l_shoulder, l_elbow, l_wrist) if l_elbow is not None else None
        right_elbow_angle = self._angle(r_shoulder, r_elbow, r_wrist) if r_elbow is not None else None

        # Hip angles: shoulder - hip - knee
        left_hip_angle = self._angle(l_shoulder, l_hip, l_knee) if l_hip is not None else None
        right_hip_angle = self._angle(r_shoulder, r_hip, r_knee) if r_hip is not None else None

        # Knee angles: hip - knee - ankle
        left_knee_angle = self._angle(l_hip, l_knee, l_ankle) if l_knee is not None else None
        right_knee_angle = self._angle(r_hip, r_knee, r_ankle) if r_knee is not None else None

        # Require at least one side (left/right) to be valid for each joint
        def side_ok(a_left, a_right, threshold):
            vals = [v for v in (a_left, a_right) if v is not None]
            if not vals:
                return False
            median_angle = np.median(vals)
            return median_angle >= threshold

        elbows_straight = side_ok(left_elbow_angle, right_elbow_angle, self.min_elbow_angle)
        hips_straight = side_ok(left_hip_angle, right_hip_angle, self.min_hip_angle)
        knees_straight = side_ok(left_knee_angle, right_knee_angle, self.min_knee_angle)

        result = {
            "is_start": elbows_straight and hips_straight and knees_straight,
            "angles": {
                "left_elbow": left_elbow_angle,
                "right_elbow": right_elbow_angle,
                "left_hip": left_hip_angle,
                "right_hip": right_hip_angle,
                "left_knee": left_knee_angle,
                "right_knee": right_knee_angle,
            },
            "points": {
                "l_elbow": l_elbow,
                "r_elbow": r_elbow,
                "l_hip": l_hip,
                "r_hip": r_hip,
                "l_knee": l_knee,
                "r_knee": r_knee,
            }
        }

        return result

    def in_end_position(self, landmarks: List[Landmark]):
        # Grab key joints (backend-agnostic)
        l_shoulder = self._get_point(landmarks, BodyJointMoveNet.LEFT_SHOULDER)
        r_shoulder = self._get_point(landmarks, BodyJointMoveNet.RIGHT_SHOULDER)

        l_elbow = self._get_point(landmarks, BodyJointMoveNet.LEFT_ELBOW)
        r_elbow = self._get_point(landmarks, BodyJointMoveNet.RIGHT_ELBOW)

        l_wrist = self._get_point(landmarks, BodyJointMoveNet.LEFT_WRIST)
        r_wrist = self._get_point(landmarks, BodyJointMoveNet.RIGHT_WRIST)

        l_hip = self._get_point(landmarks, BodyJointMoveNet.LEFT_HIP)
        r_hip = self._get_point(landmarks, BodyJointMoveNet.RIGHT_HIP)

        l_knee = self._get_point(landmarks, BodyJointMoveNet.LEFT_KNEE)
        r_knee = self._get_point(landmarks, BodyJointMoveNet.RIGHT_KNEE)

        l_ankle = self._get_point(landmarks, BodyJointMoveNet.LEFT_ANKLE)
        r_ankle = self._get_point(landmarks, BodyJointMoveNet.RIGHT_ANKLE)

        # Elbow angles: shoulder - elbow - wrist
        left_elbow_angle = self._angle(l_shoulder, l_elbow, l_wrist) if l_elbow is not None else None
        right_elbow_angle = self._angle(r_shoulder, r_elbow, r_wrist) if r_elbow is not None else None

        # Hip angles: shoulder - hip - knee
        left_hip_angle = self._angle(l_shoulder, l_hip, l_knee) if l_hip is not None else None
        right_hip_angle = self._angle(r_shoulder, r_hip, r_knee) if r_hip is not None else None

        # Knee angles: hip - knee - ankle
        left_knee_angle = self._angle(l_hip, l_knee, l_ankle) if l_knee is not None else None
        right_knee_angle = self._angle(r_hip, r_knee, r_ankle) if r_knee is not None else None

        # Helper: at least one side valid & below/above threshold
        def side_ok_min(a_left, a_right, threshold):
            """Used for joints that should be straight-ish (angle >= threshold)."""
            vals = [v for v in (a_left, a_right) if v is not None]
            if not vals:
                return False
            median_angle = np.median(vals)
            return median_angle >= threshold

        def side_ok_max(a_left, a_right, max_angle):
            """Used for joints that should be bent (angle <= max_angle)."""
            vals = [v for v in (a_left, a_right) if v is not None]
            if not vals:
                return False
            median_angle = np.median(vals)
            return median_angle <= max_angle

        # --- End-position thresholds ---
        bottom_elbow_max = self.min_elbow_angle - 40.0

        elbows_bent = side_ok_max(left_elbow_angle, right_elbow_angle, bottom_elbow_max)
        hips_straight = side_ok_min(left_hip_angle, right_hip_angle, self.min_hip_angle)
        knees_straight = side_ok_min(left_knee_angle, right_knee_angle, self.min_knee_angle)

        result = {
            "is_end": elbows_bent and hips_straight and knees_straight,
            "angles": {
                "left_elbow": left_elbow_angle,
                "right_elbow": right_elbow_angle,
                "left_hip": left_hip_angle,
                "right_hip": right_hip_angle,
                "left_knee": left_knee_angle,
                "right_knee": right_knee_angle,
            },
            "points": {
                "l_elbow": l_elbow,
                "r_elbow": r_elbow,
                "l_hip": l_hip,
                "r_hip": r_hip,
                "l_knee": l_knee,
                "r_knee": r_knee,
            },
        }

        return result
