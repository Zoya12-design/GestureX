import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True   # mouse ko physically screen ke corner mein le jao to ruk jayega
SW, SH = pyautogui.size()

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "landmarks_data" / "hand_landmarker.task"

# ---- Tuning (zaroorat par badlo) ----
X0, X1 = 0.20, 0.80        # camera frame ka wo hissa jo poori screen banta hai
Y0, Y1 = 0.15, 0.70
SMOOTH = 0.35              # 0 se 1: chota = zyada smooth magar sust
PINCH_CLICK = 0.25         # isse chota faasla = click
PINCH_FREEZE = 0.40        # isse chota faasla = cursor ruk jata hai
SCROLL_GAIN = 3500
SCROLL_DEADZONE = 0.008
CLICK_COOLDOWN = 0.4

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
               (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def dist(a, b):
    return float(np.hypot(a.x - b.x, a.y - b.y))


def extended(lm, tip, pip):
    # ungli upar hai agar uska sira apne beech wale jor se upar (chhota y) ho
    return lm[tip].y < lm[pip].y


def draw_hand(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
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

WIN = "GestureX Cursor"
cap = cv2.VideoCapture(0)
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 320, 240)
cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)

enabled = False
sx, sy = SW / 2, SH / 2
pinching = False
last_click = 0
prev_scroll_y = None

print("E = ENABLE, D = DISABLE, Q = QUIT")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        mode = "idle"
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            draw_hand(frame, lm)

            idx = extended(lm, 8, 6)
            mid = extended(lm, 12, 10)
            ring = extended(lm, 16, 14)
            pinky = extended(lm, 20, 18)

            palm_size = max(dist(lm[0], lm[9]), 1e-6)
            pinch = dist(lm[4], lm[8]) / palm_size

            if not ring and not pinky and idx and mid:
                mode = "scroll"
            elif not mid and not ring and not pinky and (idx or pinch < PINCH_FREEZE):
                mode = "move"

            if enabled and mode == "move":
                prev_scroll_y = None
                if pinch >= PINCH_FREEZE:
                    tx = np.interp(lm[8].x, [X0, X1], [0, SW])
                    ty = np.interp(lm[8].y, [Y0, Y1], [0, SH])
                    sx += (tx - sx) * SMOOTH
                    sy += (ty - sy) * SMOOTH
                    pyautogui.moveTo(int(np.clip(sx, 5, SW - 5)),
                                     int(np.clip(sy, 5, SH - 5)))
                now = time.time()
                if pinch < PINCH_CLICK and not pinching and now - last_click > CLICK_COOLDOWN:
                    pyautogui.click()
                    pinching = True
                    last_click = now
                    print("CLICK")
                elif pinch > PINCH_FREEZE + 0.1:
                    pinching = False
            elif enabled and mode == "scroll":
                y = lm[9].y
                if prev_scroll_y is not None:
                    dy = prev_scroll_y - y        # haath upar = positive = scroll up
                    if abs(dy) > SCROLL_DEADZONE:
                        pyautogui.scroll(int(dy * SCROLL_GAIN))
                prev_scroll_y = y
            else:
                prev_scroll_y = None
                if mode == "idle":
                    pinching = False
        else:
            prev_scroll_y = None
            pinching = False

        # active area ka dabba dikhao
        cv2.rectangle(frame, (int(X0 * w), int(Y0 * h)), (int(X1 * w), int(Y1 * h)),
                      (255, 200, 0), 1)
        status = "ENABLED" if enabled else "DISABLED"
        cv2.putText(frame, f"{status} | {mode}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(WIN, frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("e"):
            enabled = True
        elif k == ord("d"):
            enabled = False
        elif k == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()