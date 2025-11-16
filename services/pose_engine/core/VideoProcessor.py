import os
from typing import List, Dict, Any

from services.pose_engine.core.VideoFrameExtractor import VideoFrameExtractor
from services.pose_engine.core.BackendInterface import PoseBackend, Landmark


class VideoProcessor:
    def __init__(
        self,
        pose_backend: PoseBackend,
        target_fps: int = 15,
        resize_to=None,
        save_frames: bool = False,
        output_dir: str = "processed_frames",
    ):
        self.frame_extractor = VideoFrameExtractor(
            target_fps=target_fps,
            resize_to=resize_to,
        )
        self.pose_backend = pose_backend
        self.save_frames = save_frames
        self.output_dir = output_dir

        if self.save_frames:
            os.makedirs(self.output_dir, exist_ok=True)

    def process_video(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Process a saved video file and return a list of per-frame pose results.

        Returns:
            List[dict]: each item has frame index, timestamp, and landmarks.
        """
        results_list: List[Dict[str, Any]] = []

        for frame_rgb, frame_idx, t in self.frame_extractor.iter_frames(video_path):
            landmarks: List[Landmark] = self.pose_backend.process(frame_rgb)

            if self.save_frames and landmarks:
                drawn_bgr = self.pose_backend.draw(frame_rgb, landmarks)
                # Save frame with pose drawn
                filename = f"{self.output_dir}/frame_{frame_idx:04d}.jpg"
                import cv2

                cv2.imwrite(filename, drawn_bgr)

            results_list.append(
                {
                    "frame_idx": frame_idx,
                    "time_sec": t,
                    "landmarks": landmarks,
                }
            )

        return results_list
