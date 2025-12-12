from ultralytics import YOLO
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True)
args = parser.parse_args()

model = YOLO("runs/detect/train/weights/best.pt")

results = model(args.image, conf=0.5)

# Show result
for r in results:
    img = r.plot()
    cv2.imshow("YOLO Detection", img)
    cv2.waitKey(0)
