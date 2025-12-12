import os
import shutil
import random
import xml.etree.ElementTree as ET

# paths
RAW_IMAGES = "dataset_raw/JPEGImages"
RAW_ANN = "dataset_raw/Annotations"
OUT_IMAGES = "dataset/images"
OUT_LABELS = "dataset/labels"

# which labels to keep (only bills)
KEEP = ["OneBill", "TwoBill", "FiveBill", "TenBill", "TwentyBill", "FiftyBill", "HundredBill"]

# YOLO names mapping
NAME_MAP = {
    "OneBill": 0,
    "FiveBill": 1,
    "TenBill": 2,
    "TwentyBill": 3,
    "FiftyBill": 4,
    "HundredBill": 5
}

# Ensure folders exist
for split in ["train", "val"]:
    os.makedirs(f"{OUT_IMAGES}/{split}", exist_ok=True)
    os.makedirs(f"{OUT_LABELS}/{split}", exist_ok=True)

# split ratio
VAL_RATIO = 0.2

# get all XML annotation files
xml_files = [f for f in os.listdir(RAW_ANN) if f.endswith(".xml")]

for xml_file in xml_files:
    xml_path = os.path.join(RAW_ANN, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    img_path = os.path.join(RAW_IMAGES, filename)

    if not os.path.exists(img_path):
        print("Missing image:", filename)
        continue

    # decide train or val
    split = "val" if random.random() < VAL_RATIO else "train"

    # convert annotations
    label_out = f"dataset/labels/{split}/{filename.replace('.jpg', '.txt')}"
    img_out = f"dataset/images/{split}/{filename}"

    W = int(root.find("size").find("width").text)
    H = int(root.find("size").find("height").text)

    lines = []
    valid = False

    for obj in root.findall("object"):
        name = obj.find("name").text

        if name not in KEEP:
            continue  

        valid = True
        name = name.replace("", "")
        if name not in NAME_MAP:
            continue  # skip unknown labels 
        cls = NAME_MAP[name]

        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        # Convert to YOLO format
        x_center = ((xmin + xmax) / 2) / W
        y_center = ((ymin + ymax) / 2) / H
        width = (xmax - xmin) / W
        height = (ymax - ymin) / H

        lines.append(f"{cls} {x_center} {y_center} {width} {height}\n")

    if not valid:
        continue  # skip images with only coins

    # write label file
    with open(label_out, "w") as f:
        f.writelines(lines)

    # copy image
    shutil.copy(img_path, img_out)

print("DONE — YOLO dataset created!")
