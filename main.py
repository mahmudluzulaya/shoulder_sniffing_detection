import cv2
import numpy as np
import pyautogui
import winsound
import time
from ultralytics import YOLO

# YOLO model
model = YOLO("yolov8n.pt")

# Kamera
cap = cv2.VideoCapture(0)

# Flag-lar
screen_blurred = False
alert_playing = False
danger_start = None

def blur_screen():
    global screen_blurred
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.GaussianBlur(frame, (51, 51), 0)
    cv2.imshow("SCREEN PROTECTED", frame)
    screen_blurred = True

def play_alert():
    winsound.Beep(1200, 500)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    person_count = 0

    frame_area = frame.shape[0] * frame.shape[1]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if model.names[cls] != "person":
            continue

        if conf < 0.6:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        # Kiçik obyektləri (barmaq, qol və s.) rədd et
        if area < frame_area * 0.05:
            continue

        person_count += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Shoulder surfing məntiqi
    if person_count > 1:
        if danger_start is None:
            danger_start = time.time()

        if time.time() - danger_start > 1.5:
            status = "WARNING: SHOULDER SURFING DETECTED!"
            color = (0, 0, 255)

            blur_screen()

            if not alert_playing:
                play_alert()
                alert_playing = True
        else:
            status = "Monitoring..."
            color = (0, 255, 255)
    else:
        danger_start = None
        alert_playing = False
        status = "STATUS: SAFE"
        color = (0, 255, 0)

        if screen_blurred:
            cv2.destroyWindow("SCREEN PROTECTED")
            screen_blurred = False

    cv2.putText(frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, f"Persons detected: {person_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("YOLO Shoulder Surfing Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
