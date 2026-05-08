import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import threading
import time
from datetime import datetime
import os
from picamera2 import Picamera2

# ================= CƠ CHẾ GIAO TIẾP VỚI SENSOR =================
STATUS_FILE = "/dev/shm/fire_status.txt"

def update_status(val):
    try:
        with open(STATUS_FILE, "w") as f:
            f.write(str(val))
    except: pass

update_status(0) 

# ================= CẤU HÌNH EMAIL =================
with open('emailpass.txt', 'r') as f:
    lines = f.readlines()
    sender_email = lines[0].strip()
    app_password = lines[1].strip()
    receiver_email = lines[2].strip() if len(lines) > 2 else ''

def send_email(frame, max_retries=3, delay=5):
    attempt = 0
    while attempt < max_retries:
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)  
            server.starttls()  
            server.login(sender_email, app_password)  
            msg = MIMEMultipart()
            msg['From'], msg['To'] = sender_email, receiver_email
            msg['Subject'] = '⚠️ CẢNH BÁO KHẨN CẤP: PHÁT HIỆN HỎA HOẠN! ⚠️'

            now = datetime.now()
            time_str = now.strftime("%H:%M:%S, ngày %d/%m/%Y")
            body = f"Hệ thống phát hiện cháy lúc: {time_str}.\nVui lòng gọi PCCC: 114.\nHình ảnh đính kèm."
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            _, buffer = cv2.imencode('.jpg', frame)
            img = MIMEImage(buffer.tobytes(), name="Fire_Alert.jpg")
            msg.attach(img)
            server.send_message(msg)
            print(f"✅ Đã gửi Email cảnh báo lúc ({time_str})!")
            break  
        except Exception as e:
            attempt += 1
            time.sleep(delay)  
        finally:
            try: server.quit()  
            except: pass  

# ================= KHỞI TẠO CAMERA & YOLO =================
model = YOLO("best.pt")
names = model.model.names
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1020, 500), "format": "BGR888"})
picam2.configure(config)
picam2.start()

CONFIDENCE_THRESHOLD, MIN_AREA, MAX_AREA = 0.6, 2000, 150000  
REQUIRED_FIRE_TIME, TOLERANCE_TIME = 10.0, 1.5       
last_email_time, first_fire_time, last_fire_time, count = 0, 0.0, 0.0, 0

try:
    while True:
        frame = picam2.capture_array()
        count += 1
        if count % 2 != 0: continue

        results = model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
        fire_in_current_frame = False

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes) 
            masks = results[0].masks

            if masks is not None:
                masks = masks.xy
                overlay = frame.copy()
                for box, track_id, class_id, mask in zip(boxes, track_ids, class_ids, masks):
                    c = names[class_id]
                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1)
                    if MIN_AREA <= area <= MAX_AREA and mask.size > 0:
                        mask_arr = np.array(mask, dtype=np.int32).reshape((-1, 1, 2)) 
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.fillPoly(overlay, [mask_arr], color=(0, 0, 255))
                        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                        if c in ['fire', 'smoke']: fire_in_current_frame = True
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # LOGIC 10 GIÂY
        current_time = time.time()
        if fire_in_current_frame:
            if first_fire_time == 0.0: first_fire_time = current_time
            last_fire_time = current_time 
            elapsed_time = current_time - first_fire_time
            cv2.putText(frame, f"Xac minh: {int(elapsed_time)}/10s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            if elapsed_time >= REQUIRED_FIRE_TIME:
                update_status(1) # BÁO CHÁY QUA RAM
                if current_time - last_email_time > 60:
                    threading.Thread(target=send_email, args=(frame.copy(),), daemon=True).start()
                    last_email_time = current_time
                    first_fire_time = 0.0 
        else:
            if first_fire_time != 0.0 and (current_time - last_fire_time > TOLERANCE_TIME):
                first_fire_time = 0.0
                update_status(0)
            elif first_fire_time == 0.0:
                update_status(0)

        cv2.imshow("YOLO Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    update_status(0)