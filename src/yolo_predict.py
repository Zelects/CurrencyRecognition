from ultralytics import YOLO
import cv2
import numpy as np

# Load your trained YOLO model
model = YOLO("models/yolo_currency.pt")

def predict(path):
    # Read image
    frame = cv2.imread(path)
    if frame is None:
        raise ValueError(f"Could not read image: {path}")
    results_orig = model(frame)
    annotated_orig = results_orig[0].plot()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    edges = cv2.Canny(blurred, 50, 150)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    results_edges = model(frame)

    for result in results_edges:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Print detection info
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            print(f"Detected: {label} ({conf:.2f})")

            
            edgesss = edges[y1:y2, x1:x2]# Crop edges within YOLO box
            contours, _ = cv2.findContours(edgesss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# Find contours in the ROI
            if contours:
                # Merge all contours
                all_points = np.vstack(contours)
                x, y, w, h = cv2.boundingRect(all_points)
                # Adjust coordinates relative to full image
                cv2.rectangle(edges_color, (x1 + x, y1 + y), (x1 + x + w, y1 + y + h), (0, 255, 0), 2)

            # Add label on edge-detected image
            cv2.putText(edges_color, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    combined = cv2.hconcat([annotated_orig, edges_color])
    cv2.imshow("YOLO Original | Edge Detection + Contour", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
predict("test_images/sample.jpg")
