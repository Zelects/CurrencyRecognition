import os
import xml.etree.ElementTree as ET
from shutil import copyfile

# PATHS
ANNOTATIONS_DIR = "dataset_raw/Annotations"
IMAGES_DIR = "dataset_raw/JPEGImages"

YOLO_IMAGES = "yolo_dataset/images/val"
YOLO_LABELS = "yolo_dataset/labels/val"

# Create folders if missing
os.makedirs(YOLO_IMAGES, exist_ok=True)
os.makedirs(YOLO_LABELS, exist_ok=True)

# BILL CLASSES (we accept only these)
BILL_CLASSES = {
    "OneBill": 0,
    "TwoBill": 1,
    "FiveBill": 2,
    "TenBill": 3,
    "TwentyBill": 4,
    "FiftyBill": 5,
    "HundredBill": 6
}

def voc_to_yolo(bbox, img_w, img_h):
    xmin, ymin, xmax, ymax = bbox
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height

print("Filtering dataset...")

for xml_file in os.listdir(ANNOTATIONS_DIR):

    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(IMAGES_DIR, image_name)

    if not os.path.exists(image_path):
        print(f"Missing image for {xml_file}, skipping.")
        continue

    img_w = int(root.find("size").find("width").text)
    img_h = int(root.find("size").find("height").text)

    yolo_label_lines = []
    keep_image = False

    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()

        # Skip coins — keep only bills
        if class_name not in BILL_CLASSES:
            continue

        keep_image = True
        class_id = BILL_CLASSES[class_name]

        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        x, y, w, h = voc_to_yolo((xmin, ymin, xmax, ymax), img_w, img_h)
        yolo_label_lines.append(f"{class_id} {x} {y} {w} {h}")

    # If no bill found — skip image 
    if not keep_image:
        continue

    # Save image
    copyfile(image_path, os.path.join(YOLO_IMAGES, image_name))

    # Save YOLO label file
    label_name = image_name.replace(".jpg", ".txt")
    label_path = os.path.join(YOLO_LABELS, label_name)

    with open(label_path, "w") as f:
        f.write("\n".join(yolo_label_lines))

print("DONE! Clean YOLO dataset created.")
