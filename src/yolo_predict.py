from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("models/yolo_currency.pt")

def predict(path):
    results = model(path)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            print(f"Detected: {label}  ({conf:.2f})")

predict("test_images/sample.jpg")
