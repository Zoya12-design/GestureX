import time
from collections import Counter, deque
from pathlib import Path

import cv2  
import joblib
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent.parent
model = joblib.load(ROOT / "models" / "gesture_landmark_model.pkl")
print("Classes:", model.classes_)

pyautogui.FAILSAFE = True  # mouse top-left corner mein le jao to ruk jayega

gesture_actions = {
    "palm": "space",
    "fist": "enter",
    "ok": "esc",
    "down": "down",
    "thumb": "up",
    "index": "right",
    "l": "left",
}

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
               (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def normalize(landmarks):
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.abs(pts).max()
    if scale > 0:
        pts /= scale
    return pts.flatten()


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(
        model_asset_path=str(ROOT / "landmarks_data" / "hand_landmarker.task")),
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    running_mode=vision.RunningMode.IMAGE,
)
detector = vision.HandLandmarker.create_from_options(options)

CONF = 0.80
history = deque(maxlen=5)
enabled = False
last_gesture = None
last_time = 0
cooldown = 1.0

cap = cv2.VideoCapture(0)
WIN = "GestureX (MediaPipe)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 320, 240)
cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
print("E = ENABLE, D = DISABLE, Q = QUIT")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # collect script jaisa hi
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        stable = None
        shown = "No hand"
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            draw_hand(frame, hand)
            x = normalize(hand).reshape(1, -1)
            probs = model.predict_proba(x)[0]
            i = int(np.argmax(probs))
            pred, conf = str(model.classes_[i]), probs[i]
            shown = f"{pred} ({conf:.2f})"

            if conf >= CONF:
                history.append(pred)
            else:
                history.clear()
                last_gesture = None

            if len(history) == 5:
                g, c = Counter(history).most_common(1)[0]
                if c >= 4:
                    stable = g
        else:
            history.clear()
            last_gesture = None

        now = time.time()
        if (enabled and stable in gesture_actions
                and stable != last_gesture and now - last_time >= cooldown):
            key = gesture_actions[stable]
            pyautogui.press(key)
            print(f"Gesture: {stable} -> Key: {key}")
            last_gesture, last_time = stable, now

        status = "ENABLED" if enabled else "DISABLED"
        cv2.putText(frame, f"Control: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {shown}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(WIN, frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("e"):
            enabled = True
            history.clear()
            last_gesture = None
        elif k == ord("d"):
            enabled = False
        elif k == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()