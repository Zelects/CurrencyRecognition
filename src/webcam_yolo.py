from ultralytics import YOLO
import cv2

# Load trained YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Start webcam
cap = cv2.VideoCapture(0)  # Change to 1 if external webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # YOLO draws boxes automatically
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Currency Detection", annotated_frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
