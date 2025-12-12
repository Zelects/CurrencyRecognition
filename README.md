# Currency Detection YOLO Model

This project detects USD bills ($1, $5, $10, $20, $50, $100) using a custom-trained YOLO model.

## Running Inference

### Install Requirements
pip install -r requirements.txt

### Run Image Detection
python src/inference_yolo.py --image path/to/image.jpg

### Run Webcam Detection
python src/webcam_yolo.py
