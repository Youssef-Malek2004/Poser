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
        interpreter = tf.lite.Interpreter(model_path="models/movenet_single_pose_lightning.tflite")
        interpreter.allocate_tensors()
        return interpreter

    def process(self, frame_rgb: np.ndarray) -> List[Landmark]:
        img = frame_rgb.copy()
        #  A frame of video or an image, represented as an int32 tensor of shape: 192x192x3(Lightning) /
        # 256x256x3(Thunder). Channels order: RGB with values in [0, 255].
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)


        input_image = tf.cast(img, dtype=tf.float32)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        self.interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = self.interpreter.get_tensor(output_details[0]['index'])
        keypoints_with_scores = [
            Landmark(
                x=float(kp[1]), y=float(kp[0]), z=0.0,
                visibility=float(kp[2])
            )
            for kp in keypoints_with_scores[0][0]
        ]
        return keypoints_with_scores



        # input_img = tf.cast(img, dtype=tf.int32)

        # model = self.interpreter.signatures['serving_default']
        # output_details = model(input_img)
        # keypoints_with_scores = output_details['output_0'].numpy()
        
        # return keypoints_with_scores[0][0]

    def draw(self, frame_rgb: np.ndarray, keypoint: List[Landmark]) -> np.ndarray:
        self._draw_connections(frame_rgb, keypoint, confidence_threshold=0.4)
        self._draw_keypoints(frame_rgb, keypoint, confidence_threshold=0.4)
        return frame_rgb

    def _draw_keypoints(self, frame_rgb: np.ndarray, keypoint: List[Landmark], confidence_threshold) -> np.ndarray:
        height, width, _ = frame_rgb.shape

        for kp in keypoint:
            ky = int(kp.y * height)
            kx = int(kp.x * width)
            kp_conf = kp.visibility
            if kp_conf > confidence_threshold:
                # this is bgr
                cv2.circle(frame_rgb, (kx, ky), 5, (0,255,0), -1)
    
    def _draw_connections(self, frame_rgb: np.ndarray, keypoint: List[Landmark], confidence_threshold) -> np.ndarray:
        KEYPOINT_EDGE_INDS_TO_COLOR = {
            (0, 1): 'm',
            (0, 2): 'c',
            (1, 3): 'm',
            (2, 4): 'c',
            (0, 5): 'm',
            (0, 6): 'c',
            (5, 7): 'm',
            (7, 9): 'm',
            (6, 8): 'c',
            (8, 10): 'c',
            (5, 6): 'y',
            (5, 11): 'm',
            (6, 12): 'c',
            (11, 12): 'y',
            (11, 13): 'm',
            (13, 15): 'm',
            (12, 14): 'c',
            (14, 16): 'c'
        }
        
        height, width, _ = frame_rgb.shape

        for edge, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            p1 = keypoint[edge[0]]
            p2 = keypoint[edge[1]]
            y1 = int(p1.y * height)
            x1 = int(p1.x * width)
            y2 = int(p2.y * height)
            x2 = int(p2.x * width)

            if (p1.visibility > confidence_threshold) and (p2.visibility > confidence_threshold):
                cv2.line(frame_rgb, (x1, y1), (x2, y2), (0,255,255), 2)