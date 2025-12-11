# CurrencyDetection

A minimal starter for detecting and recognizing US currency (bills & coins) using OpenCV for preprocessing and a CNN for classification/detection.

Quick start
1. Create & activate a Python virtual environment:
   - python -m venv .venv
   - source .venv/bin/activate   (macOS / Linux)
   - .venv\Scripts\activate      (Windows PowerShell)

2. Install dependencies:
   pip install -r requirements.txt

3. Run demo:
   python examples/run_demo.py --image path/to/image.jpg --model models/checkpoint.pth

What this repo contains
- src/currency_detection/ — package with detector and utilities
- examples/run_demo.py — single-image inference script
- models/ — store model checkpoints (add to .gitignore)
- data/ — dataset notes and annotations (do not commit large files)

Usage
- Detect in an image:
  python examples/run_demo.py --image examples/test_bill.jpg --conf 0.4
- Start a webcam demo (if provided):
  python examples/webcam_demo.py --camera 0 --model models/checkpoint.pth

Notes
- Keep large files (datasets, model checkpoints, Gradle caches) out of Git; add them to .gitignore.
- Recommended detection architectures: Faster R-CNN, YOLO-family, or SSD. For edge: smaller YOLO or MobileNet-based detectors.
- For dataset format use COCO or Pascal VOC; keep labels consistent (e.g., USD_1, USD_5, USD_10, USD_25c).

Contributing
- Fork, create a feature branch, add tests, and open a PR against main.

License
- MIT (replace with your preferred license)

Contact
- Maintainer: darshan.neupane369@gmail.com