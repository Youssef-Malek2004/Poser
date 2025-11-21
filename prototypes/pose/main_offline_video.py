from services.pose_engine.core.VideoProcessor import VideoProcessor
from services.pose_engine.exercises.PushUpDetector import PushUpStartDetector
from services.pose_engine.exercises.LatPullDownDetector import LatPullDownDetector
from services.pose_engine.core.MediaPipePoseBackend import MediaPipePoseBackend
from services.pose_engine.core.MoveNetPoseBackend import MoveNetPoseBackend

def main(*, detector: str, backend: str):
    video_path = "prototypes/test-videos/pushup2.mp4"

    if detector == "pushup":
        detector = PushUpStartDetector()
    elif detector == "latpulldown":
        detector = LatPullDownDetector()
    else:
        raise ValueError(f"Unknown detector: {detector}")
    
    if backend == "mediapipe":
        backend = MediaPipePoseBackend(exercise_detector=detector)
    elif backend == "movenet":
        backend = MoveNetPoseBackend(exercise_detector=detector)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    processor = VideoProcessor(
        pose_backend=backend,
        target_fps=15,
        # resize_to=(1280, 480),
        save_frames=True,
        output_dir="prototypes/pose/pose_frames",
    )

    pose_data = processor.process_video(video_path)

    print(f"Processed {len(pose_data)} frames")
    if pose_data:
        first = pose_data[0]
        print("First frame index:", first["frame_idx"])
        print("First frame time (sec):", first["time_sec"])
        print("Num landmarks:", len(first["landmarks"]))

        with_landmarks = [f for f in pose_data if len(f["landmarks"]) > 0]
        print(f"Frames with landmarks: {len(with_landmarks)} / {len(pose_data)}")

        if with_landmarks:
            first_non_empty = with_landmarks[0]
            print("First frame with landmarks:")
            print("  frame_idx:", first_non_empty["frame_idx"])
            print("  time_sec:", first_non_empty["time_sec"])
            print("  num_landmarks:", len(first_non_empty["landmarks"]))

        for frame in pose_data:
            landmarks = frame.get("landmarks", [])
            if not landmarks:
                continue
            detector.update_reps(landmarks)

        print(f"\nTotal push-up reps detected: {detector.reps}")

if __name__ == "__main__":
    main()
