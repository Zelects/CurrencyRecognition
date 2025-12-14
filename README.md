# group members 
Jack Zheng
Ayden Rodriguez
Darshan Neupane

### Link to the NEW google slides presentation
https://docs.google.com/presentation/d/1DG1HzDLkBzWNbqE8ndEDA3gpULOyDhGQQpASLTuVhmw/edit?slide=id.g3ae554e62a2_0_1353#slide=id.g3ae554e62a2_0_1353

# Currency Detection YOLO Model
This project detects USD bills ($1, $5, $10, $20, $50, $100) using a trained YOLO model.
 We decided to switch to this model because, with a standard CNN approach, 
 the system struggled to distinguish the bills from complex or cluttered backgrounds, 
 leading to multiple detection errors. YOLO, being a real time object detection model, 
 offers higher accuracy and faster inference by detecting both the object and its 
 location simultaneously, making it more robust in dynamic environments. 
 This improvement reduces false positives and ensures reliable recognition 
 even when bills are partially obscured or in varied lighting conditions.

### Failed Demo for CNN
### failed.mp4

### Sucessful demo for yolo VVVV
### MultiBills.mp4 showing result of multiple bills on an iphone
### SingleBills.mp4 showing reseult of single bills
### Install Requirements



pip3 install -r requirements.txt


### Run Webcam Detection
python3 src/webcam_yolo.py

### Run Image Detection
python3 src/single_image.py --image path/to/image.jpg


training data: Yolo_dataset images
12000 images trained
6000 with completion for yolo
