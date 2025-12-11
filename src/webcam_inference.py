import cv2
import torch
from torchvision import models, transforms
import numpy as np
from preprocess import preprocess_frame

# Load the datasets we are predicting
classes = ["1", "5", "10", "20", "50", "100"]

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(1280, 6)
model.load_state_dict(torch.load("models/currency_model.pth", map_location=device))
model.eval()
model.to(device)

# Transform for model
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Start Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cropped, frame_display, edges = preprocess_frame(frame)

    if cropped is not None:
        img_tensor = transform(cropped).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            _, pred = torch.max(output, 1)
            label = classes[pred.item()]

        cv2.putText(frame_display, f"Prediction: ${label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show main frame
    cv2.imshow("Currency Detection", frame_display)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
