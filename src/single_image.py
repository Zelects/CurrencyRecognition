from ultralytics import YOLO
import cv2
import numpy as np  
import argparse  

parser = argparse.ArgumentParser()#Create an argument parser object
parser.add_argument("--image", type=str, required=True)#Add a required argument for the image path
args = parser.parse_args() # Parse the command-line arguments

model = YOLO("runs/detect/train/weights/best.pt")# Load a pre-trained YOLO model from a specified path

frame = cv2.imread(args.image)#Read the image file provided as argument
if frame is None:#Check if the image was not read correctly
    raise ValueError(f"Could not read image: {args.image}")#Raise an error if image reading failed

results_orig = model(frame)#Run YOLO detection on the original image
annotated_orig = results_orig[0].plot()# Get an annotated image with bounding boxes and labels

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# Convert the image to grayscale
blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)# Apply Gaussian blur to smooth the image
edges = cv2.Canny(blurred, 50, 150)  # Detect edges using the Canny edge detector
edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) # Convert edge image to BGR color for drawing colored boxes

results_edges = model(frame)#Run YOLO detection again (same as original, could reuse results_orig)

for result in results_edges:#Loop over YOLO results
    for box in result.boxes:#Loop over each detected box
        x1, y1, x2, y2 = map(int, box.xyxy[0])#Get box coordinates as integers

        roi_edges = edges[y1:y2, x1:x2]#Crop the edge-detected region of interest (ROI)

        # Find contours in the ROI
        contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# Find contours in the cropped edges
        if contours:  #Check if any contours were found
            # Merge all contours
            all_points = np.vstack(contours)  # Combine all contour points into one array
            x, y, w, h = cv2.boundingRect(all_points)  # Get bounding rectangle of all contours
            cv2.rectangle(edges_color, (x1 + x, y1 + y), (x1 + x + w, y1 + y + h), (0, 255, 0), 2)# Draw rectangle on edge image

        # Add label
        cls = int(box.cls[0]) # Get the class index of the detected object
        conf = float(box.conf[0]) # Get the confidence of the detection
        label = model.names[cls] # Get the class name from the YOLO model
        cv2.putText(edges_color, f"{label} {conf:.2f}", (x1, y1 - 5),# Draw label text above the bounding box
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

combined = cv2.hconcat([annotated_orig, edges_color]) #Horizontally concatenate original annotated image and edge+contour image
cv2.imshow("YOLO Original | Edge Detection + Contour", combined)  #Display the combined image in a window
cv2.waitKey(0)
cv2.destroyAllWindows()
