import cv2
import numpy as np

# Set up the canvas window and buttons as before

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True

# Define the color range for pen detection in HSV
lower_pen_color = np.array([35, 100, 100])  # Example values; adjust based on your pen color
upper_pen_color = np.array([85, 255, 255])

while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the pen color
    mask = cv2.inRange(hsv, lower_pen_color, upper_pen_color)
    
    # Find contours of the pen
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour, which is assumed to be the pen
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Get the center of the bounding box
        center = (x + w // 2, y + h // 2)
        
        # Draw a circle at the pen tip
        cv2.circle(frame, center, 10, (0, 255, 0), -1)

        # Add your drawing logic here using the center coordinates
        
    # Display the frame
    cv2.imshow("Pen Detection", frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
