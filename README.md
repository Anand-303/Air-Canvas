# Air Canvas Application

An interactive application that lets you draw in the air using hand gestures captured by your webcam. This project uses computer vision to track your finger movements and translates them into digital drawings.

## Features

- Real-time hand tracking using MediaPipe
- Multiple color options (Blue, Green, Red, Yellow)
- Adjustable brush sizes
- Clear canvas functionality
- Smooth drawing experience

## Prerequisites

- Python 3.7 or higher
- Webcam
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository or download the source code
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python "Air Canvas stable.py"
   ```

2. Position your hand in front of the webcam
3. Use your index finger to draw in the air
4. Use the on-screen buttons to change colors:
   - Blue
   - Green
   - Red
   - Yellow
5. Press 'c' to clear the canvas
6. Press 'q' to quit the application

## Controls

- **Draw**: Move your index finger in front of the camera
- **Change Color**: Click on the color buttons on the right side of the window
- **Clear Canvas**: Press 'c' key
- **Quit**: Press 'q' key

## Project Structure

- `Air Canvas stable.py` - Main application file (most stable version)
- `2nd approach.py` - Alternative implementation
- `requirements.txt` - List of required Python packages
- `PEN/` - Directory containing pen-related resources
- `saved_drawings/` - Directory where drawings can be saved

## Dependencies

- OpenCV
- NumPy
- MediaPipe

## Troubleshooting

- If you encounter issues with the webcam, ensure no other application is using it
- Make sure you have proper lighting conditions for better hand tracking
- If you get dependency errors, try reinstalling the requirements:
  ```
  pip uninstall -r requirements.txt -y
  pip install -r requirements.txt
  ```

## Acknowledgments

- Built with MediaPipe and OpenCV
- Inspired by various computer vision projects

## Contact

For any questions or feedback, please open an issue in the repository.
