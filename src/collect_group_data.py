
from pathlib import Path
import csv
import re

import cv2
import mediapipe as mp


# Project folders
PROJECT = Path(__file__).resolve().parent.parent
MODEL = PROJECT / "models" / "mediapipe" / "hand_landmarker.task"
DATA_FOLDER = PROJECT / "landmarks_data"

DATA_FOLDER.mkdir(exist_ok=True)

# Same 7 gestures for all group members
GESTURES = {
    ord("1"): "palm",
    ord("2"): "fist",
    ord("3"): "thumb",
    ord("4"): "index",
    ord("5"): "ok",
    ord("6"): "l",
    ord("7"): "down"
}

# Ask for member name
person = input("Enter your name (example: zoya): ").strip().lower()
person = re.sub(r"[^a-z0-9_-]", "", person)

if not person:
    raise ValueError("Please enter a valid name.")

if not MODEL.exists():
    raise FileNotFoundError(
        f"Hand Landmarker model not found: {MODEL}"
    )

csv_path = DATA_FOLDER / f"landmarks_{person}.csv"

# Create CSV with the format expected by train_landmarks.py
columns = [f"f{i}" for i in range(63)] + ["label", "person"]

if not csv_path.exists():
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)

# Count existing samples so collection can continue later
counts = {gesture: 0 for gesture in GESTURES.values()}

with open(csv_path, "r", newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row["label"] in counts:
            counts[row["label"]] += 1

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1
)

selected_gesture = None

print("\nGesture keys:")
for key, gesture in GESTURES.items():
    print(f"{chr(key)} = {gesture}")

print("\nPress SPACE to save one sample.")
print("Press Q to quit.\n")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror the webcam preview
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(mp_image)

            hand_detected = len(result.hand_landmarks) > 0

            # Draw detected hand landmarks
            if hand_detected:
                hand = result.hand_landmarks[0]

                for point in hand:
                    x = int(point.x * frame.shape[1])
                    y = int(point.y * frame.shape[0])
                    cv2.circle(
                        frame, (x, y), 4, (0, 255, 0), -1
                    )

            cv2.putText(
                frame,
                f"Person: {person}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            status = selected_gesture or "Select gesture: 1-7"

            cv2.putText(
                frame,
                f"Gesture: {status}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Hand detected: {hand_detected}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if hand_detected else (0, 0, 255),
                2
            )

            if selected_gesture:
                cv2.putText(
                    frame,
                    f"Saved: {counts[selected_gesture]}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            cv2.imshow("GestureX Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key in GESTURES:
                selected_gesture = GESTURES[key]
                print("Selected:", selected_gesture)

            elif key == ord(" ") and selected_gesture:
                if not hand_detected:
                    print("No hand detected. Sample not saved.")
                    continue

                # Extract 21 landmarks, each with x, y, z
                features = []

                for point in result.hand_landmarks[0]:
                    features.extend([
                        point.x,
                        point.y,
                        point.z
                    ])

                if len(features) != 63:
                    print("Invalid landmark count. Sample skipped.")
                    continue

                row = features + [selected_gesture, person]

                with open(csv_path, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(row)

                counts[selected_gesture] += 1

                print(
                    f"Saved {selected_gesture}: "
                    f"{counts[selected_gesture]}"
                )

finally:
    camera.release()
    cv2.destroyAllWindows()

print("\nCollection finished.")
print("CSV saved at:", csv_path)
print("Final sample counts:")

for gesture, count in counts.items():
    print(f"{gesture}: {count}")
