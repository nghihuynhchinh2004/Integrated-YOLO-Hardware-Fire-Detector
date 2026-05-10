import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
import cvzone
from picamera2 import Picamera2

# ================= HIỂN THỊ X11 & ĐỒNG BỘ RAM =================
os.environ["QT_QPA_PLATFORM"] = "xcb"
STATUS_FILE = "/dev/shm/fire_status.txt"
IMAGE_FILE = "/dev/shm/latest_frame.jpg" # BIẾN MỚI: Đường dẫn lưu ảnh nháp trên RAM

def update_status(val_str):
    try:
        with open(STATUS_FILE, "w") as f:
            f.write(val_str)
    except: pass

update_status("SAFE")

# ================= KHỞI TẠO PICAMERA2 =================
print("📸 Đang khởi động Picamera2...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1020, 500), "format": "BGR888"})
picam2.configure(config)
picam2.start()
picam2.set_controls({
    "AwbEnable": False,
    "ColourGains": (1.5, 1.2) 
})

# ================= KHỞI TẠO YOLO =================
YOLO_MODEL_PATH = "/home/pi/Desktop/Main_Project_Code_Python/best.pt"
print("⏳ Đang tải mô hình YOLOv11 Detection...")
model = YOLO(YOLO_MODEL_PATH)
names = model.model.names

# ================= BIẾN LOGIC =================
CONFIDENCE_THRESHOLD = 0.4  
MIN_AREA = 1000            
MAX_AREA = 250000  

REQUIRED_TIME = 10.0       
TOLERANCE_TIME = 3.0       

hazard_start_time = None
last_seen_time = 0         
cancel_msg_time = 0   
count = 0
current_hazard_type = "SAFE" 

print("\n▶️ Hệ thống YOLO Detection bắt đầu hoạt động. Nhấn 'q' để tắt.\n")

try:
    while True:
        frame = picam2.capture_array()
        count += 1
        if count % 2 != 0: continue

        results = model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        highest_hazard_in_frame = "SAFE"
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist() 
            
            for box, class_id, conf in zip(boxes, class_ids, confidences):
                c = names[class_id]
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                
                if MIN_AREA <= area <= MAX_AREA:
                    color = (0, 0, 255) if c == 'fire' else (200, 200, 200)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{c} {conf:.2f}'
                    cvzone.putTextRect(frame, label, (x1, max(30, y1)), scale=1, thickness=1, colorR=color)
                    
                    if c == 'fire':
                        highest_hazard_in_frame = "FIRE"
                    elif c == 'smoke' and highest_hazard_in_frame != "FIRE":
                        highest_hazard_in_frame = "SMOKE"

        # ================= LOGIC XÁC MINH =================
        current_time = time.time()
        
        if highest_hazard_in_frame != "SAFE":
            last_seen_time = current_time 
            current_hazard_type = highest_hazard_in_frame 
            
            if hazard_start_time is None:
                hazard_start_time = current_time 
                print(f"Phát hiện {current_hazard_type}... Bắt đầu đếm 10 giây.")

            elapsed = current_time - hazard_start_time

            if elapsed >= REQUIRED_TIME:
                update_status(current_hazard_type) 
        else:
            if hazard_start_time is not None:
                time_since_last_seen = current_time - last_seen_time
                
                if time_since_last_seen <= TOLERANCE_TIME:
                    hold_time_left = TOLERANCE_TIME - time_since_last_seen
                    cv2.putText(frame, f"Signal lost... Holding: {hold_time_left:.1f}s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                else:
                    print("Mất dấu hoàn toàn -> Hủy bộ đếm.")
                    hazard_start_time = None
                    cancel_msg_time = current_time
                    update_status("SAFE") 

        # ================= HIỂN THỊ VÀ LƯU ẢNH TRỰC TIẾP =================
        if hazard_start_time is not None:
            time_left = max(0, REQUIRED_TIME - (current_time - hazard_start_time))
            cv2.putText(frame, f"Xac minh {current_hazard_type}: {time_left:.1f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        elif current_time - cancel_msg_time < 3.0:
            cv2.putText(frame, "Khong thay dau hieu -> Huy bo dem", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        cv2.imshow("He thong PCCC Thong minh - YOLO Detection", frame)
        
        # --- LƯU ẢNH HIỆN TẠI VÀO RAM ĐỂ GỬI MAIL ---
        try:
            cv2.imwrite(IMAGE_FILE, frame)
        except: pass

        if cv2.waitKey(1) & 0xFF == ord('q'): break

except Exception as e:
    print(f" ⚠️ Lỗi hệ thống Camera: {e}")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    update_status("SAFE")
    print("\n🛑 Đã tắt hệ thống Camera an toàn.")