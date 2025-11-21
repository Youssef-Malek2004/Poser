import cv2
import mediapipe as mp
from typing import List
import numpy as np

from services.pose_engine.core.BackendInterface import PoseBackend, Landmark
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class MediaPipePoseBackend(PoseBackend):
    def __init__(self, exercise_detector: "ExerciseDetector | None" = None):
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.exercise_detector = exercise_detector

    def process(self, frame_rgb: np.ndarray) -> List[Landmark]:
        results = self._pose.process(frame_rgb)
        landmarks: List[Landmark] = []

        if hasattr(results, "pose_landmarks") and results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.append(
                    Landmark(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=lm.visibility,
                    )
                )

        return landmarks

    def _draw_angle(self, frame, point, angle, label):
        if angle is None:
            return
        x, y = int(point[0]), int(point[1])
        cv2.putText(
            frame,
            f"{label}: {int(angle)}°",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    def draw(self, frame_rgb: np.ndarray, landmarks: List[Landmark]) -> np.ndarray:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # --- Convert landmarks to Mediapipe protobuf ---
        lm_list = landmark_pb2.NormalizedLandmarkList(
            landmark=[
                landmark_pb2.NormalizedLandmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility
                )
                for lm in landmarks
            ]
        )

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame_bgr,
            lm_list,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        )

        # If there is an exercise detector, use it in a generic way
        if self.exercise_detector is not None and landmarks:
            # --- Start / end position info (detector-specific) ---
            det_start = self.exercise_detector.in_start_position(landmarks)
            det_end = self.exercise_detector.in_end_position(landmarks)

            # Support both dict-return and bool-return detectors
            if isinstance(det_start, dict):
                is_start = bool(det_start.get("is_start"))
            else:
                is_start = bool(det_start)

            if isinstance(det_end, dict):
                is_end = bool(det_end.get("is_end"))
            else:
                is_end = bool(det_end)

            # --- Status text (exercise-agnostic wording) ---
            if is_start:
                status_text = "Start position"
                status_color = (0, 255, 255)  # Yellow
            elif is_end:
                status_text = "End position"
                status_color = (0, 255, 255)
            else:
                status_text = "Not in start / end"
                status_color = (50, 50, 255)

            cv2.putText(
                frame_bgr,
                status_text,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                status_color,
                2,
                cv2.LINE_AA,
            )

            chosen_det = det_start if is_start else det_end

            angles = None
            points = None
            if isinstance(chosen_det, dict):
                angles = chosen_det.get("angles")
                points = chosen_det.get("points")

            if angles and points:
                joints = [
                    ("LE", "left_elbow", "l_elbow"),
                    ("RE", "right_elbow", "r_elbow"),
                    ("LH", "left_hip", "l_hip"),
                    ("RH", "right_hip", "r_hip"),
                    ("LK", "left_knee", "l_knee"),
                    ("RK", "right_knee", "r_knee"),
                ]

                for label, ang_key, pt_key in joints:
                    if ang_key not in angles or pt_key not in points:
                        continue

                    angle = angles[ang_key]
                    point = points[pt_key]

                    if point is not None and angle is not None:
                        px = int(point[0] * frame_bgr.shape[1])
                        py = int(point[1] * frame_bgr.shape[0])
                        self._draw_angle(frame_bgr, (px, py), angle, label)

        return frame_bgr
