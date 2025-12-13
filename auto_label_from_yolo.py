from ultralytics import YOLO
import cv2
import os
import shutil

# Load your trained YOLO detector
model = YOLO("runs/detect/train/weights/best.pt")
SRC_ROOT = "/Users/jackzheng/CurrencyDetection1/dataset_processed"

OUT_IMG = "yolo_dataset/images/val"
OUT_LBL = "yolo_dataset/labels/val"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

SUPPORTED_EXT = (".jpg", ".jpeg", ".png")

total_images = 0
labeled_images = 0

for bill_folder in os.listdir(SRC_ROOT):
    bill_path = os.path.join(SRC_ROOT, bill_folder, "train")

    if not os.path.isdir(bill_path):
        continue

    print(f"\nProcessing {bill_folder}...")

    for img_name in os.listdir(bill_path):
        if not img_name.lower().endswith(SUPPORTED_EXT):
            continue

        img_path = os.path.join(bill_path, img_name)
        total_images += 1

        results = model(img_path, conf=0.4)
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            continue  # skip images with no detections

        # Copy image
        out_img_path = os.path.join(OUT_IMG, img_name)
        shutil.copy(img_path, out_img_path)

        # Write YOLO label file
        label_path = os.path.join(OUT_LBL, img_name.rsplit(".", 1)[0] + ".txt")
        with open(label_path, "w") as f:
            for box in r.boxes:
                cls = int(box.cls[0])
                x, y, w, h = box.xywhn[0]
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        labeled_images += 1

print("\nAUTO-LABELING DONE")
print(f"Total images seen: {total_images}")
print(f"Images labeled: {labeled_images}")
