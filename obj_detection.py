import cv2
from HailoYOLODetector import HailoYOLODetector

def main():
  hef_path = 'models/yolov10x.hef'  # Update with the correct path to your HEF file
  detector = HailoYOLODetector(hef_path)
  # cap = cv2.VideoCapture(1)
  cap = cv2.VideoCapture('videos/854100-hd_1920_1080_25fps.mp4')

  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      
      bbox_image = detector.detect(frame)

      cv2.imshow('result', bbox_image)

      key = cv2.waitKey(1) & 0xff
      if key == ord('q'):
        break
      
  except Exception as e:
    print(e)
  finally:
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  main()