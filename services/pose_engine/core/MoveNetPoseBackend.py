import numpy as np
import tensorflow as tf
import cv2
from services.pose_engine.exercises.ExerciseDetector import ExerciseDetector
from services.pose_engine.core.BackendInterface import Landmark, PoseBackend
from typing import List, Dict
from services.pose_engine.core.joints import BodyJointMoveNet
# Most suitable for detecting the pose of a single person who is 3ft ~ 6ft away from a
# device’s webcam that captures the video stream.

MIN_CROP_KEYPOINT_SCORE = 0.2
class MoveNetPoseBackend(PoseBackend):
    def __init__(self, exercise_detector: ExerciseDetector | None = None):
        self.exercise_detector = exercise_detector
        self.interpreter = self.load_model()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def load_model(self):
        interpreter = tf.lite.Interpreter(model_path="models/movenet_single_pose_lightning.tflite")
        interpreter.allocate_tensors()
        return interpreter
    

    # Confidence score to determine whether a keypoint prediction is reliable.

    def init_crop_region(self, image_height, image_width):
        """Defines the default crop region.

        The function provides the initial crop region (pads the full image from both
        sides to make it a square image) when the algorithm cannot reliably determine
        the crop region from the previous frame.
        """
        if image_width > image_height:
            box_height = image_width / image_height
            box_width = 1.0
            y_min = (image_height / 2 - image_width / 2) / image_height
            x_min = 0.0
        else:
            box_height = 1.0
            box_width = image_height / image_width
            y_min = 0.0
            x_min = (image_width / 2 - image_height / 2) / image_width

        return {
            'y_min': y_min,
            'x_min': x_min,
            'y_max': y_min + box_height,
            'x_max': x_min + box_width,
            'height': box_height,
            'width': box_width
        }

    def torso_visible(self, keypoints_with_scores):
        """Checks whether there are enough torso keypoints.

        This function checks whether the model is confident at predicting one of the
        shoulders/hips which is required to determine a good crop region.
        """
        return ((keypoints_with_scores[0, 0, BodyJointMoveNet.LEFT_HIP, 2] >
                MIN_CROP_KEYPOINT_SCORE or
                keypoints_with_scores[0, 0, BodyJointMoveNet.RIGHT_HIP, 2] >
                MIN_CROP_KEYPOINT_SCORE) and
                (keypoints_with_scores[0, 0, BodyJointMoveNet.LEFT_SHOULDER, 2] >
                MIN_CROP_KEYPOINT_SCORE or
                keypoints_with_scores[0, 0, BodyJointMoveNet.RIGHT_SHOULDER, 2] >
                MIN_CROP_KEYPOINT_SCORE))

    def determine_torso_and_body_range(self, keypoints_with_scores, target_keypoints, center_y, center_x):
        """Calculates the maximum distance from each keypoints to the center location.

        The function returns the maximum distances from the two sets of keypoints:
        full 17 keypoints and 4 torso keypoints. The returned information will be
        used to determine the crop size. See determine_crop_region for more detail.
        """
        torso_joints = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        for joint in torso_joints:
            dist_y = abs(center_y - target_keypoints[joint][0])
            dist_x = abs(center_x - target_keypoints[joint][1])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        for joint in BodyJointMoveNet:
            if keypoints_with_scores[0, 0, joint.value, 2] < MIN_CROP_KEYPOINT_SCORE:
                continue
            dist_y = abs(center_y - target_keypoints[joint.name][0])
            dist_x = abs(center_x - target_keypoints[joint.name][1])
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y
            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

    def determine_crop_region(self, landmarks: List[Landmark], image_height, image_width):
        """Determines the region to crop the image for the model to run inference on.

        The algorithm uses the detected joints from the previous frame to estimate
        the square region that encloses the full body of the target person and
        centers at the midpoint of two hip joints. The crop size is determined by
        the distances between each joints and the center point.
        When the model is not confident with the four torso joint predictions, the
        function returns a default crop which is the full image padded to square.
        """
        # Convert landmarks back to keypoints_with_scores format for cropping algorithm
        keypoints_with_scores = np.zeros((1, 1, 17, 3))
        for i, lm in enumerate(landmarks):
            keypoints_with_scores[0, 0, i, 0] = lm.y  # y coordinate (normalized)
            keypoints_with_scores[0, 0, i, 1] = lm.x  # x coordinate (normalized)
            keypoints_with_scores[0, 0, i, 2] = lm.visibility  # confidence score

        target_keypoints = {}
        for joint in BodyJointMoveNet:
            target_keypoints[joint.name] = [
                keypoints_with_scores[0, 0, joint.value, 0] * image_height,
                keypoints_with_scores[0, 0, joint.value, 1] * image_width
            ]

        if self.torso_visible(keypoints_with_scores):
            center_y = (target_keypoints['LEFT_HIP'][0] +
                        target_keypoints['RIGHT_HIP'][0]) / 2
            center_x = (target_keypoints['LEFT_HIP'][1] +
                        target_keypoints['RIGHT_HIP'][1]) / 2

            (max_torso_yrange, max_torso_xrange,
            max_body_yrange, max_body_xrange) = self.determine_torso_and_body_range(
                keypoints_with_scores, target_keypoints, center_y, center_x)

            crop_length_half = np.amax(
                [max_torso_xrange * 1.9, max_torso_yrange * 1.9,
                max_body_yrange * 1.2, max_body_xrange * 1.2])

            tmp = np.array(
                [center_x, image_width - center_x, center_y, image_height - center_y])
            crop_length_half = np.amin(
                [crop_length_half, np.amax(tmp)])

            crop_corner = [center_y - crop_length_half, center_x - crop_length_half]

            if crop_length_half > max(image_width, image_height) / 2:
                return self.init_crop_region(image_height, image_width)
            else:
                crop_length = crop_length_half * 2
            return {
                'y_min': crop_corner[0] / image_height,
                'x_min': crop_corner[1] / image_width,
                'y_max': (crop_corner[0] + crop_length) / image_height,
                'x_max': (crop_corner[1] + crop_length) / image_width,
                'height': (crop_corner[0] + crop_length) / image_height -
                    crop_corner[0] / image_height,
                'width': (crop_corner[1] + crop_length) / image_width -
                    crop_corner[1] / image_width
            }
        else:
            return self.init_crop_region(image_height, image_width)

    def crop_and_resize(self, image, crop_region, crop_size):
        """Crops and resize the image to prepare for the model input."""
        boxes = [[crop_region['y_min'], crop_region['x_min'],
                crop_region['y_max'], crop_region['x_max']]]
        output_image = tf.image.crop_and_resize(
            image, box_indices=[0], boxes=boxes, crop_size=crop_size)
        return output_image

    def run_model_inference(self, image, crop_region, crop_size=(192, 192)):
        """Runs model inference on the cropped region.

        The function runs the model inference on the cropped region and updates the
        model output to the original image coordinate system.
        """
        image_height, image_width, _ = image.shape
        
        # Crop and resize the image
        input_image = self.crop_and_resize(
            tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
        
        # Cast to float32 and prepare for TFLite model
        input_image = tf.cast(input_image, dtype=tf.float32)
        
        # Run TFLite inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image.numpy())
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Update the coordinates to original image space
        for idx in range(17):
            keypoints_with_scores[0, 0, idx, 0] = (
                crop_region['y_min'] * image_height +
                crop_region['height'] * image_height *
                keypoints_with_scores[0, 0, idx, 0]) / image_height
            keypoints_with_scores[0, 0, idx, 1] = (
                crop_region['x_min'] * image_width +
                crop_region['width'] * image_width *
                keypoints_with_scores[0, 0, idx, 1]) / image_width
        
        return keypoints_with_scores

    def process(self, frame_rgb: np.ndarray, crop_region: Dict) -> List[Landmark]:
        """Process a frame and return landmarks using the cropping algorithm."""
        keypoints_with_scores = self.run_model_inference(frame_rgb, crop_region)
        
        # Convert to List[Landmark]
        landmarks = [
            Landmark(
                x=float(keypoints_with_scores[0, 0, i, 1]),  # x coordinate
                y=float(keypoints_with_scores[0, 0, i, 0]),  # y coordinate
                z=0.0,  # MoveNet doesn't provide z
                visibility=float(keypoints_with_scores[0, 0, i, 2])  # confidence score
            )
            for i in range(17)
        ]
        
        return landmarks

    def draw(self, frame_rgb: np.ndarray, landmarks: List[Landmark]) -> np.ndarray:
        """Draw keypoints and connections on the frame."""
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self._draw_connections(frame_bgr, landmarks, confidence_threshold=0.2)
        self._draw_keypoints(frame_bgr, landmarks, confidence_threshold=0.2)
        
        # Draw exercise detector info if available
        if self.exercise_detector is not None and landmarks:
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

            # Status text
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

        return frame_bgr

    def _draw_keypoints(self, frame_bgr: np.ndarray, landmarks: List[Landmark], confidence_threshold):
        """Draw keypoints as circles."""
        height, width, _ = frame_bgr.shape

        for lm in landmarks:
            ky = int(lm.y * height)
            kx = int(lm.x * width)
            if lm.visibility > confidence_threshold:
                cv2.circle(frame_bgr, (kx, ky), 4, (0, 255, 0), -1)
    
    def _draw_connections(self, frame_bgr: np.ndarray, landmarks: List[Landmark], confidence_threshold):
        """Draw skeleton connections."""
        KEYPOINT_EDGE_INDS_TO_COLOR = {
            (0, 1): (255, 0, 255),    # magenta
            (0, 2): (0, 255, 255),    # cyan
            (1, 3): (255, 0, 255),
            (2, 4): (0, 255, 255),
            (0, 5): (255, 0, 255),
            (0, 6): (0, 255, 255),
            (5, 7): (255, 0, 255),
            (7, 9): (255, 0, 255),
            (6, 8): (0, 255, 255),
            (8, 10): (0, 255, 255),
            (5, 6): (0, 255, 0),      # yellow/green
            (5, 11): (255, 0, 255),
            (6, 12): (0, 255, 255),
            (11, 12): (0, 255, 0),
            (11, 13): (255, 0, 255),
            (13, 15): (255, 0, 255),
            (12, 14): (0, 255, 255),
            (14, 16): (0, 255, 255)
        }
        
        height, width, _ = frame_bgr.shape

        for edge, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            p1 = landmarks[edge[0]]
            p2 = landmarks[edge[1]]
            y1 = int(p1.y * height)
            x1 = int(p1.x * width)
            y2 = int(p2.y * height)
            x2 = int(p2.x * width)

            if (p1.visibility > confidence_threshold) and (p2.visibility > confidence_threshold):
                cv2.line(frame_bgr, (x1, y1), (x2, y2), color, 2)