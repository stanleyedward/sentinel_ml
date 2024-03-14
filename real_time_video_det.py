import pathlib
import cv2
from NudeNet import nudenet
import os


videeo_detector = nudenet.NudeDetector()
video_model_fps = 14 # low ball
video_classes_to_detect = ['FACE_MALE']
    
def video_detection(path, fps: int, output_dir = None, break_after_flag:bool = False):
    assert fps >= 1, "fps cannot be lower than 1"
    
    cap = cv2.VideoCapture(path)
    frame_count = 0
    skip_frames = -(-fps // video_model_fps) #ceiling
    # skip_frames = fps // video_model_fps #floor
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % skip_frames == 0:
            frame_prediction = videeo_detector.detect(frame)
            print(f"Frame {frame_count}: Predicted class: {frame_prediction}")
            
            classes_detected = [detection for detection in frame_prediction if detection["class"] in video_classes_to_detect]
            if break_after_flag and classes_detected:
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
    video_detection('video_samples/me_in_toilet.mp4', fps = 30, output_dir=None, break_after_flag=True)
    # video_detection(0, fps = 30, output_dir=None, break_after_flag=True) for webcam 

