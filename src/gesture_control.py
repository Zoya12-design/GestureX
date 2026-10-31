import time
from collections import deque, Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
import pyautogui

# Model path (project folder ke relative)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "knn_gesture_model.pkl"
knn = joblib.load(MODEL_PATH)
print("Model loaded. Classes:", knn.classes_)

pyautogui.FAILSAFE = True  # mouse ko top-left corner mein le jao to ruk jayega

gesture_actions = {
    "palm": "space",
    "fist": "enter",
    "ok": "esc",
    "down": "down",
    "thumb": "up",
    "index": "right",
    "l": "left",
}

history = deque(maxlen=5)
enabled = False
last_action_gesture = None
last_action_time = 0
cooldown = 1.0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Webcam open nahi ho rahi.")

print("E = ENABLE controls, D = DISABLE, Q = QUIT")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        size = min(400, h, w)
        x1, y1 = (w - size) // 2, (h - size) // 2
        x2, y2 = x1 + size, y1 + size

        region = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        image = (resized.astype("float32") / 255.0).reshape(1, -1)

        probs = knn.predict_proba(image)[0]
        best = int(np.argmax(probs))
        confidence = probs[best]
        prediction = str(knn.classes_[best])

        if confidence >= 0.99:
            history.append(prediction)
        else:
            history.clear()
            last_action_gesture = None

        stable = None
        if len(history) == 5:
            g, c = Counter(history).most_common(1)[0]
            if c >= 4:
                stable = g

        now = time.time()
        if (enabled and stable in gesture_actions
                and stable != last_action_gesture
                and now - last_action_time >= cooldown):
            key = gesture_actions[stable]
            pyautogui.press(key)
            print(f"Gesture: {stable} -> Key: {key}")
            last_action_gesture = stable
            last_action_time = now

        status = "ENABLED" if enabled else "DISABLED"
        label = stable if stable else "Detecting..."
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Control: {status}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {label}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("PC Gesture Control", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("e"):
            enabled = True
            history.clear()
            last_action_gesture = None
        elif k == ord("d"):
            enabled = False
        elif k == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()