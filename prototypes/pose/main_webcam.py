import cv2
from services.pose_engine.exercises.PushUpDetectorMoveNet import PushUpStartDetector
from services.pose_engine.exercises.LatPullDownDetectorMoveNet import LatPullDownDetector

def main(*, detector: str = "pushup", backend: str = "mediapipe"):
    if detector == "pushup":
        detector = PushUpStartDetector()
    elif detector == "latpulldown":
        detector = LatPullDownDetector()
    else:
        raise ValueError(f"Unknown detector: {detector}")
    
    if backend == "mediapipe":
        from services.pose_engine.core.MediaPipePoseBackend import MediaPipePoseBackend
        backend = MediaPipePoseBackend(exercise_detector=detector)
    elif backend == "movenet":
        from services.pose_engine.core.MoveNetPoseBackend import MoveNetPoseBackend
        backend = MoveNetPoseBackend(exercise_detector=detector)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks = backend.process(frame_rgb)

        if landmarks:
            current_reps = detector.update_reps(landmarks)

            drawn_bgr = backend.draw(frame_rgb, landmarks)

            cv2.putText(
                drawn_bgr,
                f"Reps: {current_reps}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            drawn_bgr = frame_bgr

        cv2.imshow("Poser - Webcam Pose", drawn_bgr)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
