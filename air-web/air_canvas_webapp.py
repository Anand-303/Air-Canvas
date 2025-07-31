import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import tensorflow as tf
import os
import time
from flask import Flask, render_template, Response

app = Flask(__name__)

tf.get_logger().setLevel('ERROR')  # Set log level to ERROR

def smooth_points(points, alpha=0.5):
    smoothed = [points[0]]   
    for i in range(1, len(points)):
        smoothed.append((alpha * points[i] + (1 - alpha) * smoothed[-1]).astype(int))
    return smoothed

bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

kernel = np.ones((5, 5), np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0   

brushSizes = [5, 10, 15, 20]
brushIndex = 1  # Default brush size index

isEraser = False

# Create a directory to save images if it doesn't exist
if not os.path.exists("saved_drawings"):
    os.mkdir("saved_drawings")

# Initialize the canvas size
paintWindow = np.zeros((471, 750, 3)) + 255  # Increase width to 750

# Draw buttons and labels on the paint window
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)  # Clear button
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)  # Blue button
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)  # Green button
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)  # Red button
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)  # Yellow button
paintWindow = cv2.rectangle(paintWindow, (620, 1), (710, 65), (255, 0, 255), 2)  # Save button

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "SAVE", (635, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils  

def generate_frames():
    global bpoints, gpoints, rpoints, ypoints, blue_index, green_index, red_index, yellow_index, paintWindow, colorIndex, isEraser
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to match the size of paintWindow
        frame = cv2.resize(frame, (750, 471))

        # Flip the frame horizontally for natural (mirror-like) interaction
        frame = cv2.flip(frame, 1)

        # Draw buttons and labels on the tracking window
        frame_with_buttons = frame.copy()
        frame_with_buttons = cv2.rectangle(frame_with_buttons, (40, 1), (140, 65), (0, 0, 0), 2)  # Clear button
        frame_with_buttons = cv2.rectangle(frame_with_buttons, (160, 1), (255, 65), (255, 0, 0), 2)  # Blue button
        frame_with_buttons = cv2.rectangle(frame_with_buttons, (275, 1), (370, 65), (0, 255, 0), 2)  # Green button
        frame_with_buttons = cv2.rectangle(frame_with_buttons, (390, 1), (485, 65), (0, 0, 255), 2)  # Red button
        frame_with_buttons = cv2.rectangle(frame_with_buttons, (505, 1), (600, 65), (0, 255, 255), 2)  # Yellow button
        frame_with_buttons = cv2.rectangle(frame_with_buttons, (620, 1), (710, 65), (255, 0, 255), 2)  # Save button

        cv2.putText(frame_with_buttons, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_with_buttons, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_with_buttons, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_with_buttons, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_with_buttons, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_with_buttons, "SAVE", (635, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        # Convert BGR to RGB
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(framergb)

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * 750)  # Adjust for resized frame
                    lmy = int(lm.y * 471)  # Adjust for resized frame
                    landmarks.append([lmx, lmy])

                # Draw hand landmarks on the frame
                mpDraw.draw_landmarks(frame_with_buttons, handslms, mpHands.HAND_CONNECTIONS)

            # Get coordinates of the forefinger and thumb
            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0], landmarks[4][1])
            cv2.circle(frame_with_buttons, center, brushSizes[brushIndex], (0, 255, 0), -1)  

            # Apply smoothing to the center point
            smoothed_center = smooth_points([np.array(center)])[0]
            
            if (thumb[1] - center[1] < 30):
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1

            elif center[1] <= 65:
                if 40 <= center[0] <= 140:  # Clear Button
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]

                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0
                    paintWindow[67:, :, :] = 255

                elif 160 <= center[0] <= 255:
                    colorIndex = 0  # Blue
                elif 275 <= center[0] <= 370:
                    colorIndex = 1  # Green
                elif 390 <= center[0] <= 485:
                    colorIndex = 2  # Red
                elif 505 <= center[0] <= 600:
                    colorIndex = 3  # Yellow
                elif 620 <= center[0] <= 710:
                    # Save Drawing button
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    cv2.imwrite(f"saved_drawings/drawing_{timestamp}.png", paintWindow)

            else:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(smoothed_center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(smoothed_center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(smoothed_center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(smoothed_center)

        points = [bpoints, gpoints, rpoints, ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame_with_buttons, points[i][j][k - 1], points[i][j][k], colors[i], brushSizes[brushIndex])

        ret, buffer = cv2.imencode('.jpg', frame_with_buttons)
        frame_with_buttons = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_with_buttons + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == "__main__":
    app.run(debug=True)
