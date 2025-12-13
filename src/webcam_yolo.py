from ultralytics import YOLO
import cv2
import numpy as np
model = YOLO("runs/detect/train/weights/best.pt")#Load YOLO model from specified path

cap = cv2.VideoCapture(1)#Open video capture; 1 is external webcam, 0 is usually built-in

while True: 
    ret, frame = cap.read() 
    if not ret:#Check if the frame was not captured
        break 

    results_orig = model(frame) # Run YOLO on the current frame
    annotated_orig = results_orig[0].plot() # Get annotated frame with bounding boxes and labels

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#Convert frame to grayscale
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)#Apply Gaussian blur with 7x7 kernel
    edges = cv2.Canny(blurred, 50, 150)#Detect edges using Canny edge detector with thresholds 50 and 150
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)#Convert edge image to BGR for colored drawings

    results_edges = model(frame) # Run YOLO again on original frame to get bounding boxes for edge processing

    for result in results_edges:#Loop over YOLO detection results
        for box in result.boxes:#Loop over each detected bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])#Extract coordinates of bounding box as integers

            roi_edges = edges[y1:y2, x1:x2]#Crop the YOLO box region from edge image

            contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# Find external contours in cropped ROI
            if contours:# Check if any contours exist
                all_points = np.vstack(contours) #Merge all contour points into one array
                x, y, w, h = cv2.boundingRect(all_points) #Get bounding rectangle around merged contours
                cv2.rectangle(edges_color, (x1 + x, y1 + y), (x1 + x + w, y1 + y + h), (0, 255, 0), 2) #Draw rectangle on edge image

            cls = int(box.cls[0])#Get class index of detected object
            conf = float(box.conf[0])#Get confidence score of detection
            label = model.names[cls]#Get class name from YOLO model
            cv2.putText(edges_color, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)#Draw label above bounding box

    combined = cv2.hconcat([annotated_orig, edges_color])#Concatenate original and edge-detection frames horizontally
    cv2.imshow("YOLO Original and Edge Detection + Contour", combined)#Display combined frame

    if cv2.waitKey(1) & 0xFF == ord('q'): # Wait 1 ms for key press, exit if 'q' is pressed
        break 

cap.release() 
cv2.destroyAllWindows() 
