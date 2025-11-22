import numpy as np
import tensorflow as tf
import cv2
from services.pose_engine.exercises.ExerciseDetector import ExerciseDetector
from services.pose_engine.core.BackendInterface import Landmark, PoseBackend
from typing import List


# Most suitable for detecting the pose of a single person who is 3ft ~ 6ft away from a
# device’s webcam that captures the video stream.
class MoveNetPoseBackend(PoseBackend):
    def __init__(self, exercise_detector: ExerciseDetector | None = None):
        self.exercise_detector = exercise_detector
        self.interpreter = self.load_model()

    def load_model(self):
        interpreter = tf.lite.Interpreter(
            model_path="models/thunder.tflite"
        )
        interpreter.allocate_tensors()
        return interpreter

    def process(self, frame_rgb: np.ndarray) -> List[Landmark]:
        img = frame_rgb.copy()
        #  A frame of video or an image, represented as an int32 tensor of shape: 192x192x3(Lightning) /
        # 256x256x3(Thunder). Channels order: RGB with values in [0, 255].
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)

        input_image = tf.cast(img, dtype=tf.float32)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]["index"], input_image.numpy())
        # Invoke inference.
        self.interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = self.interpreter.get_tensor(output_details[0]["index"])

        MOVENET_TO_MEDIAPIPE = {
            0: 0,   # nose -> NOSE
            1: 2,   # left_eye -> LEFT_EYE
            2: 5,   # right_eye -> RIGHT_EYE
            3: 7,   # left_ear -> LEFT_EAR
            4: 8,   # right_ear -> RIGHT_EAR
            5: 11,  # left_shoulder -> LEFT_SHOULDER
            6: 12,  # right_shoulder -> RIGHT_SHOULDER
            7: 13,  # left_elbow -> LEFT_ELBOW
            8: 14,  # right_elbow -> RIGHT_ELBOW
            9: 15,  # left_wrist -> LEFT_WRIST
            10: 16, # right_wrist -> RIGHT_WRIST
            11: 23, # left_hip -> LEFT_HIP
            12: 24, # right_hip -> RIGHT_HIP
            13: 25, # left_knee -> LEFT_KNEE
            14: 26, # right_knee -> RIGHT_KNEE
            15: 27, # left_ankle -> LEFT_ANKLE
            16: 28, # right_ankle -> RIGHT_ANKLE
        }
        
        # Initialize with 33 landmarks (MediaPipe format), set missing ones to zero visibility
        movenet_landmarks = [
            Landmark(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(33)
        ]
        
        # Map MoveNet keypoints to MediaPipe indices
        for movenet_idx, kp in enumerate(keypoints_with_scores[0][0]):
            mediapipe_idx = MOVENET_TO_MEDIAPIPE[movenet_idx]
            movenet_landmarks[mediapipe_idx] = Landmark(
            x=float(kp[1]), 
            y=float(kp[0]), 
            z=0.0, 
            visibility=float(kp[2])
            )

        return movenet_landmarks

        # input_img = tf.cast(img, dtype=tf.int32)

        # model = self.interpreter.signatures['serving_default']
        # output_details = model(input_img)
        # keypoints_with_scores = output_details['output_0'].numpy()

        # return keypoints_with_scores[0][0]

    def _draw_angle(self, frame, point, angle, label):
        if angle is None:
            return
        x, y = int(point[0]), int(point[1])
        
        # Format angle text
        text = f"{label}: {int(angle)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw semi-transparent background for better readability
        cv2.rectangle(
            frame,
            (x + 5, y - text_height - 15),
            (x + text_width + 15, y - 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (x + 10, y - 10),
            font,
            font_scale,
            (0, 255, 255),  # Yellow color
            thickness,
            cv2.LINE_AA
        )

    def draw(self, frame_rgb: np.ndarray, landmarks: List[Landmark]) -> np.ndarray:
        # Lower threshold for better keypoint visibility
        confidence_threshold = 0.3
        self._draw_connections(frame_rgb, landmarks, confidence_threshold=confidence_threshold)
        self._draw_keypoints(frame_rgb, landmarks, confidence_threshold=confidence_threshold)
        
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
            frame_rgb,
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
                    px = int(point[0] * frame_rgb.shape[1])
                    py = int(point[1] * frame_rgb.shape[0])
                    self._draw_angle(frame_rgb, (px, py), angle, label)
        return frame_rgb

    def _draw_keypoints(
        self, frame_rgb: np.ndarray, landmarks: List[Landmark], confidence_threshold
    ) -> np.ndarray:
        height, width, _ = frame_rgb.shape

        for kp in landmarks:
            ky = int(kp.y * height)
            kx = int(kp.x * width)
            kp_conf = kp.visibility
            
            if kp_conf > confidence_threshold:
                # Color code by confidence: high confidence = green, medium = yellow
                if kp_conf > 0.6:
                    color = (0, 255, 0)  # Green for high confidence
                    radius = 6
                else:
                    color = (0, 255, 255)  # Yellow for medium confidence
                    radius = 5
                    
                cv2.circle(frame_rgb, (kx, ky), radius, color, -1)
                # Add white border for better visibility
                cv2.circle(frame_rgb, (kx, ky), radius + 1, (255, 255, 255), 1)

    def _draw_connections(
        self, frame_rgb: np.ndarray, landmarks: List[Landmark], confidence_threshold
    ) -> np.ndarray:
        # Use BodyJoint indices (0-32) since landmarks array is mapped to BodyJoint format
        # These connections match the skeleton structure for the 17 MoveNet keypoints
        KEYPOINT_EDGE_INDS_TO_COLOR = {
            # Face connections
            (0, 2): "m",
            (0, 5): "c",
            (2, 7): "m",
            (5, 8): "c",
            (0,11): "m",
            (0,12): "c",
            (11,13): "m",
            (13,15): "m",
            (12,14): "c",
            (14,16): "c",
            (11,12): "y",
            (11,23): "m",
            (12,24): "c",
            (23,24): "y",
            (23,25): "m",
            (25,27): "m",
            (24,26): "c",
            (26,28): "c"
        }

        height, width, _ = frame_rgb.shape

        for edge, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            p1 = landmarks[edge[0]]
            p2 = landmarks[edge[1]]
            y1 = int(p1.y * height)
            x1 = int(p1.x * width)
            y2 = int(p2.y * height)
            x2 = int(p2.x * width)

            if (p1.visibility > confidence_threshold) and (
                p2.visibility > confidence_threshold
            ):
                cv2.line(frame_rgb, (x1, y1), (x2, y2), (0, 255, 255), 2)
