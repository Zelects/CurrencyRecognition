import cv2
import numpy as np

def preprocess_frame(frame):
    # Resize for easier processing
    resized = cv2.resize(frame, (600, 400))
    
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:  # threshold to remove noise
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # Look for rectangles (4 sides)
            if len(approx) == 4 and area > max_area:
                biggest = approx
                max_area = area

    # If a bill is detected
    if biggest is not None:
        pts = biggest.reshape(4, 2)
        # show the bounding box
        cv2.drawContours(resized, [biggest], -1, (0, 255, 0), 2)

        # Crop the detected bill region 
        x, y, w, h = cv2.boundingRect(biggest)
        cropped = resized[y:y+h, x:x+w]

        # Prepare for model input
        cropped_resized = cv2.resize(cropped, (224, 224))
        cropped_rgb = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2RGB)

        return cropped_rgb, resized, edges

    return None, resized, edges
