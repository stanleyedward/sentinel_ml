from NudeNet.nudenet import NudeDetector
nude_detector = NudeDetector()
import cv2

if __name__ == '__main__':
    img = cv2.imread('image_samples/cocka.jpg')
    result = nude_detector.detect(img)  # Returns list of detections
    print(result)