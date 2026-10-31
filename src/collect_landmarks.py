import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

LABELS = {"1": "palm", "2": "fist", "3": "ok", "4": "down",
          "5": "thumb", "6": "index", "7": "l"}
TARGET = 300  # har gesture ke itne samples

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "landmarks_data"
MODEL = DATA_DIR / "hand_landmarker.task"
OUT = DATA_DIR / "landmarks.csv"
DATA_DIR.mkdir(exist_ok=True)

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
               (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

person = input("Apna naam likho (e.g. zoya): ").strip().lower()


def normalize(landmarks):
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    pts -= pts[0]                      # wrist ko origin banao
    scale = np.abs(pts).max()
    if scale > 0:
        pts /= scale                   # size ka farq khatam
    return pts.flatten()               # 63 numbers


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    running_mode=vision.RunningMode.IMAGE,
)
detector = vision.HandLandmarker.create_from_options(options)

new_file = not OUT.exists()
f = open(OUT, "a", newline="")
writer = csv.writer(f)
if new_file:
    writer.writerow([f"f{i}" for i in range(63)] + ["label", "person"])

counts = {name: 0 for name in LABELS.values()}
current = None
recording = False

cap = cv2.VideoCapture(0)
print("1-7 = gesture chuno | R = recording start/stop | Q = quit")
print(LABELS)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            draw_hand(frame, hand)
            if recording and current:
                writer.writerow(list(normalize(hand)) + [current, person])
                counts[current] += 1
                if counts[current] >= TARGET:
                    recording = False
                    print(f"{current} complete!")

        info = f"Gesture: {current} ({counts.get(current, 0)}/{TARGET}) REC: {recording}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.imshow("Collect landmarks", frame)

        k = cv2.waitKey(1) & 0xFF
        if chr(k) in LABELS:
            current = LABELS[chr(k)]
            recording = False
        elif k == ord("r") and current:
            recording = not recording
        elif k == ord("q"):
            break
finally:
    f.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Saved:", OUT, counts)