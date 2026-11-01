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

# ---- Tuning ----
X0, X1 = 0.20, 0.80        # camera frame ka wo hissa jo poori screen banta hai
Y0, Y1 = 0.15, 0.70
DEADZONE_PX = 8
PINCH_CLICK = 0.25
PINCH_FREEZE = 0.40
SCROLL_GAIN = 3500
SCROLL_DEADZONE = 0.008
CLICK_COOLDOWN = 0.4
PLAY_KEY = "playpause"     # ya "space" (space sirf tab chalta hai jab browser par focus ho)
HOLD_S = 0.6               # khula haath itni der rakho to pause/resume
ACTION_COOLDOWN = 1.5
VOL_STEP_S = 0.25          # volume har itne second mein ek step
THUMB_DY = 0.6             # thumbs up/down kitna saaf ho

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
               (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def dist(a, b):
    return float(np.hypot(a.x - b.x, a.y - b.y))


def extended(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def draw_hand(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


def classify(lm):
    idx = extended(lm, 8, 6)
    mid = extended(lm, 12, 10)
    ring = extended(lm, 16, 14)
    pinky = extended(lm, 20, 18)
    palm_size = max(dist(lm[0], lm[9]), 1e-6)
    pinch = dist(lm[4], lm[8]) / palm_size
    thumb_dy = (lm[2].y - lm[4].y) / palm_size   # + = angutha apne jor se upar

    if idx and mid and ring and pinky:
        return "open", pinch
    if idx and mid and not ring and not pinky:
        return "scroll", pinch
    if not mid and not ring and not pinky and (idx or pinch < PINCH_FREEZE):
        return "move", pinch
    if not (idx or mid or ring or pinky):
        if thumb_dy > THUMB_DY:
            return "vol_up", pinch
        if thumb_dy < -THUMB_DY:
            return "vol_down", pinch
    return "idle", pinch


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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 320, 240)
cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)

enabled = False
sx, sy = SW / 2, SH / 2
pinching = False
last_click = 0
prev_scroll_y = None
open_since = None
open_fired = False
last_action = 0
last_vol = 0

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
        now = time.time()
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            draw_hand(frame, lm)
            mode, pinch = classify(lm)

            if enabled:
                # --- cursor aur click ---
                if mode == "move":
                    if pinch >= PINCH_FREEZE:
                        tx = np.interp(lm[8].x, [X0, X1], [0, SW])
                        ty = np.interp(lm[8].y, [Y0, Y1], [0, SH])
                        d = float(np.hypot(tx - sx, ty - sy))
                        if d > DEADZONE_PX:
                            alpha = min(0.6, max(0.08, d / 400))
                            sx += (tx - sx) * alpha
                            sy += (ty - sy) * alpha
                            pyautogui.moveTo(int(np.clip(sx, 5, SW - 5)),
                                             int(np.clip(sy, 5, SH - 5)))
                    if pinch < PINCH_CLICK and not pinching and now - last_click > CLICK_COOLDOWN:
                        pyautogui.click()
                        pinching = True
                        last_click = now
                        print("CLICK")
                    elif pinch > PINCH_FREEZE + 0.1:
                        pinching = False

                # --- scroll ---
                if mode == "scroll":
                    y = lm[9].y
                    if prev_scroll_y is not None:
                        dy = prev_scroll_y - y
                        if abs(dy) > SCROLL_DEADZONE:
                            pyautogui.scroll(int(dy * SCROLL_GAIN))
                    prev_scroll_y = y

                # --- pause / resume (khula haath, kuch der ruk kar) ---
                if mode == "open":
                    if open_since is None:
                        open_since = now
                    elif (not open_fired and now - open_since >= HOLD_S
                          and now - last_action >= ACTION_COOLDOWN):
                        pyautogui.press(PLAY_KEY)
                        open_fired = True
                        last_action = now
                        print("PAUSE/RESUME")

                # --- volume ---
                if mode in ("vol_up", "vol_down") and now - last_vol >= VOL_STEP_S:
                    pyautogui.press("volumeup" if mode == "vol_up" else "volumedown")
                    last_vol = now
                    print(mode.upper())

        # state reset
        if mode != "scroll":
            prev_scroll_y = None
        if mode != "move":
            pinching = False
        if mode != "open":
            open_since = None
            open_fired = False

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