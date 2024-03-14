import pathlib
import cv2
from NudeNet import nudenet
import os

# camera = cv2.VideoCapture(0) #webcam
detector = nudenet.NudeDetector()
model_fps = 14 # low ball
    
def video_detection(path, fps: int, output_dir = None, break_after_flag:bool = False):
    cap = cv2.VideoCapture(path)
    frame_count = 0
    assert fps >= 1, "fps cannot be lower than 1"
    skip_frames = -(-fps // model_fps) #ceiling
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % skip_frames == 0:
            frame_prediction = detector.detect(frame)
            print(f"Frame {frame_count}: Predicted class: {frame_prediction}")
            
            if break_after_flag and frame_prediction:
                break
            else:
                pass
            
            if output_dir is not None:
                output_dir = 'frames'
                frame_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
                cv2.imwrite(frame_path, frame)
    
        frame_count += 1
    cap.release()
    
if __name__ == '__main__':
    # video_detection('video_samples/dexter.mp4', 30)
    video_detection('image_samples/booba.jpg', 1)

