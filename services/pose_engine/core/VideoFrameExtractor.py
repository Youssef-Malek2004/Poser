import cv2
import os

class VideoFrameExtractor:
    def __init__(self, target_fps: int = 15, resize_to=None, max_frames=None):
        """
        target_fps: desired sampling fps (e.g. 10–15 for exercise analysis)
        resize_to: (width, height) or None to keep original size
        max_frames: optional cap on number of frames to yield
        """
        self.target_fps = target_fps
        self.resize_to = resize_to
        self.max_frames = max_frames

    def iter_frames(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            # fallback if metadata is broken
            orig_fps = self.target_fps

        frame_step = max(1, round(orig_fps / self.target_fps))
        frame_idx = 0
        yielded = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # sample every frame_step
            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            if self.resize_to is not None:
                w, h = self.resize_to
                frame_bgr = cv2.resize(frame_bgr, (w, h))

            # convert BGR -> RGB for mediapipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            time_sec = frame_idx / orig_fps if orig_fps > 0 else 0.0
            yield frame_rgb, frame_idx, time_sec

            yielded += 1
            frame_idx += 1

            if self.max_frames is not None and yielded >= self.max_frames:
                break

        cap.release()
