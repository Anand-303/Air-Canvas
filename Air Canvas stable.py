import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Points for different colors
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Indices for color points
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# Brush sizes
brushSizes = [5, 10, 15, 20]
brushIndex = 1  # Default brush size index

# Shape options and selection
shapeIndex = -1  # -1 for freehand, 0 for square, 1 for circle, 2 for triangle, 3 for line

# Colors and default settings
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0  # Default color index
isEraser = False

# Create a white canvas and buttons
paintWindow = np.ones((471, 636, 3)) * 255

# Add buttons for colors and shapes
buttons = {
    "CLEAR": (40, 1, 140, 65, (0, 0, 0)),
    "BLUE": (160, 1, 255, 65, (255, 0, 0)),
    "GREEN": (275, 1, 370, 65, (0, 255, 0)),
    "RED": (390, 1, 485, 65, (0, 0, 255)),
    "YELLOW": (505, 1, 600, 65, (0, 255, 255)),
    "SQUARE": (5, 70, 55, 120, (0, 0, 0)),
    "CIRCLE": (5, 130, 55, 180, (0, 0, 0)),
    "TRIANGLE": (5, 190, 55, 240, (0, 0, 0)),
    "LINE": (5, 250, 55, 300, (0, 0, 0)),
    "BRUSH+": (5, 310, 55, 360, (0, 0, 0)),
    "BRUSH-": (5, 370, 55, 420, (0, 0, 0))
}

for label, (x1, y1, x2, y2, color) in buttons.items():
    paintWindow = cv2.rectangle(paintWindow, (x1, y1), (x2, y2), color, 2)
    cv2.putText(paintWindow, label, (x1 + 5, y1 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils  

# Initialize the webcam
cap = cv2.VideoCapture(0)

drawing = False
start_point = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x, y, c = frame.shape

    # Flip the frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw color and shape buttons on the frame
    for label, (x1, y1, x2, y2, color) in buttons.items():
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + 5, y1 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, brushSizes[brushIndex], (0, 255, 0), -1)  

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
                isEraser = False
            elif 275 <= center[0] <= 370:
                colorIndex = 1  # Green
                isEraser = False
            elif 390 <= center[0] <= 485:
                colorIndex = 2  # Red
                isEraser = False
            elif 505 <= center[0] <= 600:
                colorIndex = 3  # Yellow
                isEraser = False
            elif 5 <= center[0] <= 55:
                if 70 <= center[1] <= 120:
                    shapeIndex = 0  # Square
                elif 130 <= center[1] <= 180:
                    shapeIndex = 1  # Circle
                elif 190 <= center[1] <= 240:
                    shapeIndex = 2  # Triangle
                elif 250 <= center[1] <= 300:
                    shapeIndex = 3  # Line
                elif 310 <= center[1] <= 360:
                    brushIndex = (brushIndex + 1) % len(brushSizes)  # Brush size increment
                elif 370 <= center[1] <= 420:
                    brushIndex = (brushIndex - 1) % len(brushSizes)  # Brush size decrement

        else:
            if isEraser:
                color = (255, 255, 255)
            else:
                color = colors[colorIndex]

            if shapeIndex == -1:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)
            else:
                if not drawing:
                    start_point = center
                    drawing = True
                else:
                    if shapeIndex == 0:  # Square
                        top_left = (min(start_point[0], center[0]), min(start_point[1], center[1]))
                        bottom_right = (max(start_point[0], center[0]), max(start_point[1], center[1]))
                        cv2.rectangle(paintWindow, top_left, bottom_right, color, -1)
                    elif shapeIndex == 1:  # Circle
                        radius = int(np.sqrt((center[0] - start_point[0])**2 + (center[1] - start_point[1])**2))
                        cv2.circle(paintWindow, start_point, radius, color, -1)
                    elif shapeIndex == 2:  # Triangle
                        pts = np.array([start_point, (center[0], start_point[1]), (center[0], center[1])], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.fillPoly(paintWindow, [pts], color)
                    elif shapeIndex == 3:  # Line
                        cv2.line(paintWindow, start_point, center, color, brushSizes[brushIndex])
                    drawing = False

    # Draw color and shape buttons on the canvas
    for label, (x1, y1, x2, y2, color) in buttons.items():
        paintWindow = cv2.rectangle(paintWindow, (x1, y1), (x2, y2), color, 2)
        cv2.putText(paintWindow, label, (x1 + 5, y1 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Draw on the frame
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], brushSizes[brushIndex])

    # Show the images
    cv2.imshow("Paint", paintWindow)
    cv2.imshow("Frame", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
