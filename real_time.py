import pathlib
import cv2
from NudeNet import nudenet
import os
# cascade_path=pathlib.Path(cv2.__file__).parent.absolute() / "data"/"haarcascade_frontalface_default.xml"
#print(cascade_path)
# clf = cv2.CascadeClassifier(str(cascade_path))
# camera = cv2.VideoCapture(0)            #webcam
# camera = cv2.VideoCapture("me_in_toilet.mp4")
# camera = cv2.VideoCapture("image_samples/sleep.jpeg")

# camera = cv2.VideoCapture("fnf.mp4")
#uncomment for testing!
detector = nudenet.NudeDetector()
cap = cv2.VideoCapture('video_samples/dexter.mp4')

# Create a directory to save frames
output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break        
    
    frame_prediction = detector.detect(frame)
    # Classify each frame
    # frame_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
    # predicted_class = NudeClassifier.classify(frame_path)
    print(frame_prediction)
    # 
    # cv2.imwrite(frame_path, frame)
    # 
    # Do something with the predicted class, like printing it
    # print(f"Frame {frame_count}: Predicted class: {predicted_class}")
    
    frame_count += 1

# Release the video capture object
cap.release()
