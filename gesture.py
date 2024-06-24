import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Function to process frames and detect gestures
def detect_gesture(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold the image
    _, thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contours were found
    if contours:
        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)
        # Calculate the convex hull
        hull = cv2.convexHull(max_contour)
        # Draw the contour and hull
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
        
        # Determine number of fingers (example: based on hull defects)
        defects = cv2.convexityDefects(max_contour, cv2.convexHull(max_contour, returnPoints=False))
        if defects is not None:
            count = 0
            for i in range(defects.shape[0]):
                s, e, f, _ = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])
                # Calculate triangle area around defects to filter out noise
                triangle_area = cv2.contourArea(np.array([start, far, end]))
                if triangle_area > 1000:  # Adjust threshold as needed
                    count += 1
            # Display number of fingers detected
            cv2.putText(frame, str(count+1), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    return frame

# Main loop to capture frames and detect gestures
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Process frame to detect gestures
    processed_frame = detect_gesture(frame)

    # Display processed frame
    cv2.imshow('Gesture Recognition', processed_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
