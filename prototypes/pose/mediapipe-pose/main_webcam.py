import cv2
from services.pose_engine.core.MediaPipePoseBackend import MediaPipePoseBackend


def main():
    backend = MediaPipePoseBackend()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Optional: set resolution
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        landmarks = backend.process(frame_rgb)

        if landmarks:
            drawn_bgr = backend.draw(frame_rgb, landmarks)
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
