from prototypes.pose.main_webcam import main as run_webcam
from prototypes.pose.main_offline_video import main as run_offline_video
if __name__ == '__main__':
   run_offline_video(detector="latpulldown", backend="mediapipe")