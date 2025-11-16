import cv2
import mediapipe as mp
from typing import List
import numpy as np

from services.pose_engine.core.BackendInterface import PoseBackend, Landmark
from mediapipe.framework.formats import landmark_pb2


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class MediaPipePoseBackend(PoseBackend):
    def __init__(self):
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_rgb: np.ndarray) -> List[Landmark]:
        results = self._pose.process(frame_rgb)
        landmarks: List[Landmark] = []

        if hasattr(results, "pose_landmarks"):
            if results.pose_landmarks:
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

    def draw(self, frame_rgb: np.ndarray, landmarks: List[Landmark]) -> np.ndarray:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Build protobuf list
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

        mp_drawing.draw_landmarks(
            frame_bgr,
            lm_list,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        )

        return frame_bgr

