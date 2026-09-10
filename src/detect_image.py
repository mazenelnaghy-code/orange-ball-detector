"""Run the orange-ball model on image files instead of the webcam.

Useful for checking the model works without plugging in a camera.

    python src/detect_image.py photo.jpg
    python src/detect_image.py photos/*.jpg --conf 0.4

Annotated copies are written to examples/.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "mazen.pt"
OUTPUT_DIR = PROJECT_DIR / "examples"

parser = argparse.ArgumentParser(description="Detect orange balls in image files.")
parser.add_argument("images", nargs="+", help="image files to run the model on")
parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold (default: 0.5)")
args = parser.parse_args()

model = YOLO(MODEL_PATH)
OUTPUT_DIR.mkdir(exist_ok=True)

for image in args.images:
    path = Path(image)
    if not path.exists():
        print(f"{path.name}: file not found")
        continue

    result = model.predict(path, conf=args.conf, verbose=False)[0]
    scores = [round(float(c), 3) for c in result.boxes.conf]

    print(f"{path.name}: {len(scores)} detection(s) {scores}")
    result.save(filename=OUTPUT_DIR / f"detected_{path.stem}.jpg")

print(f"\nAnnotated images saved to {OUTPUT_DIR}")
