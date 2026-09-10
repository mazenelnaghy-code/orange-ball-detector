"""Live orange-ball detection from a webcam using a custom-trained YOLO model.

Press 'q' to quit.
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# Paths are relative to the project, so the code works wherever the folder is copied
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "mazen.pt"

# The model rarely scores above 0.9 even on clear detections, so a 0.9 threshold
# hides almost everything. 0.5 is a workable starting point - raise it if you get
# too many false boxes, lower it if the ball is missed.
CONFIDENCE = 0.5

# Load the trained model
model = YOLO(MODEL_PATH)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    sys.exit("Could not open the webcam. Is another app using it, or is the index wrong?")

cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model.predict(frame, conf=CONFIDENCE, verbose=False)

    # Draw results
    for result in results:
        annotated_frame = result.plot()

        # Show frame
        cv2.imshow("Mazen Orange Ball", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
