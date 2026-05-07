import time, json, os, csv
from datetime import datetime
import threading

# Thư viện cho Camera & AI
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

# Thư viện cho Cảm biến phần cứng (Từ source của bạn)
from ml_predictor import predict_hazard
from DS18B20_Temperature_Sensor import get_temperature_data
from ADS_1115_Air_Sensor import get_air_sensor_data
from fusion_decision import fusion_decision
from severity_estimator import estimate_severity
from temporal_confirmation import TemporalConfirmation
from relay_control import setup_relays, write_actuators, cleanup_relays
# Nếu file era_mqtt_client.py không có sẵn trong thư mục, bạn hãy comment 3 hàm dưới đây lại nhé:
from era_mqtt_client import setup_mqtt, publish_telemetry, stop_mqtt

# ==========================================
# BIẾN TOÀN CỤC CHIA SẺ GIỮA 2 LUỒNG (SHARED STATE)
# ==========================================
shared_state = {
    "yolo_fire_confirmed": False,   # AI Camera đã xác nhận cháy (đủ 10s) chưa?
    "sensor_fire_confirmed": False, # Phần cứng có xác nhận cháy không?
    "current_temp": 0.0,
    "current_voc": 0
}

# ==========================================
# CẤU HÌNH GMAIL (GIỮ NGUYÊN)
# ==========================================
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
            msg['From'] = sender_email  
            msg['To'] = receiver_email
            msg['Subject'] = '⚠️ CẢNH BÁO KHẨN CẤP: PHÁT HIỆN HỎA HOẠN! ⚠️'

            now = datetime.now()
            time_str = now.strftime("%H:%M:%S, ngày %d/%m/%Y")

            body = f"""THÔNG BÁO: HỆ THỐNG PHÁT HIỆN SỰ CỐ!
Hệ thống báo cháy đã phát hiện dấu hiệu hỏa hoạn/khói vào thời điểm: {time_str}.

Bạn vui lòng gọi ngay cho Cảnh sát PCCC Thành phố Hồ Chí Minh theo thông báo sau:
- Số điện thoại báo cháy: 114
- Số điện thoại đường dây nóng: (028) 39 200 996
Hình ảnh camera ghi nhận tại hiện trường được đính kèm trong email này."""

            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            _, buffer = cv2.imencode('.jpg', frame)
            img = MIMEImage(buffer.tobytes(), name="Fire_Alert.jpg")
            img.add_header('Content-Disposition', 'attachment', filename="Fire_Alert.jpg")
            msg.attach(img)

            server.send_message(msg)
            print(f"✅ Đã gửi Email cảnh báo thành công!")
            break  
        except smtplib.SMTPException as e:
            attempt += 1
            print(f"❌ Lỗi gửi email: {e}. Thử lại lần {attempt}...")
            time.sleep(delay)  
        finally:
            try: server.quit()  
            except: pass  

# ==========================================
# LUỒNG SENSOR (CHẠY NGẦM SONG SONG VỚI CAMERA)
# ==========================================
def sensor_worker():
    temporal_filter = TemporalConfirmation(safe_confirm_count=5, hazard_confirm_count=3)
    LOG_DIR = "logs"
    JSONL_FILE = f"{LOG_DIR}/fire_realtime_ml.jsonl"
    CSV_FILE = f"{LOG_DIR}/fire_realtime_ml.csv"
    LOOP_INTERVAL = 1.0

    os.makedirs(LOG_DIR, exist_ok=True)
    setup_relays()
    try: setup_mqtt() 
    except: pass

    csv_exists = os.path.isfile(CSV_FILE)
    csv_file = open(CSV_FILE, mode="a", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["date", "time", "temperature_C", "temp_status", "temp_trend", "heat_rise", "mq2_HI", "mq135_HI", "voc_ppm", "ml_hazard", "fusion_hazard", "fusion_reason", "YOLO_STATUS"])

    print("\n✅ LUỒNG CẢM BIẾN & PHẦN CỨNG ĐÃ SẴN SÀNG...\n")

    try:
        while True:
            temp = get_temperature_data()
            air = get_air_sensor_data()
            ts_epoch = time.time()
            now = datetime.now()

            temp_c = round(temp["temperature_c"], 2)
            mq2_hi = round(air["mq2"]["HI"], 3)
            mq135_hi = round(air["mq135"]["HI"], 3)
            voc_ppm = int(air["voc"]["ppm"])
            
            # Cập nhật thông số ra ngoài cho Camera hiển thị
            shared_state["current_temp"] = temp_c
            shared_state["current_voc"] = voc_ppm

            # Tính toán AI và Mờ hóa
            severity_result = estimate_severity(temp_c, temp["heat_rise_status"], mq2_hi, mq135_hi, voc_ppm)
            ml_label, ml_hazard = predict_hazard(temp_c, mq2_hi, mq135_hi, voc_ppm)
            
            fusion_result = fusion_decision(
                temp_c=temp_c, temp_status=temp["status"], temp_rise=temp["heat_rise_status"],
                mq2_hi=mq2_hi, mq135_hi=mq135_hi, voc_ppm=voc_ppm, ml_label=ml_label,
                severity_score=severity_result["severity_score"], severity_level=severity_result["severity_level"],
                action_level=severity_result["action_level"]
            )
            
            temporal_result = temporal_filter.update(fusion_result)

            # ========================================================
            # LOGIC SO SÁNH GIÁ TRỊ (CROSS-VERIFICATION: YOLO & SENSORS)
            # ========================================================
            actuator = temporal_result["actuator"]
            
            # Nếu YOLO xác nhận có cháy, ÉP BUỘC rơ-le kích hoạt dù cảm biến chưa đủ ngưỡng
            if shared_state["yolo_fire_confirmed"]:
                temporal_result["confirmed_hazard"] = "FIRE (YOLO + SENSORS)"
                actuator["buzzer"] = True
                actuator["mist"] = True
                actuator["emergency"] = True
                temporal_result["relay_active"] = True

            # Cập nhật lại cho YOLO biết cảm biến có báo cháy không
            shared_state["sensor_fire_confirmed"] = (temporal_result["confirmed_hazard"] == "FIRE")

            write_actuators(actuator)

            # Ghi Log và Đẩy MQTT
            csv_writer.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), temp_c, temp["status"], temp["trend"], temp["heat_rise_status"], mq2_hi, mq135_hi, voc_ppm, severity_result["severity_score"], severity_result["severity_level"], severity_result["action_level"], ml_hazard, fusion_result["hazard"], fusion_result["reason"], shared_state["yolo_fire_confirmed"]])
            csv_file.flush()

            era_payload = {
                "temp_c": temp_c, "mq2_hi": mq2_hi, "voc_ppm": voc_ppm,
                "fusion_hazard": temporal_result["confirmed_hazard"],
                "yolo_fire_detected": shared_state["yolo_fire_confirmed"]
            }
            try: publish_telemetry(era_payload)
            except: pass

            time.sleep(LOOP_INTERVAL)

    except Exception as e:
        print(f"⚠️ Lỗi luồng cảm biến: {e}")
    finally:
        cleanup_relays()
        csv_file.close()
        try: stop_mqtt()
        except: pass

# ==========================================
# LUỒNG CHÍNH (CAMERA + YOLOv11)
# ==========================================
if __name__ == "__main__":
    # 1. Bật luồng cảm biến chạy ngầm
    sensor_thread = threading.Thread(target=sensor_worker, daemon=True)
    sensor_thread.start()

    # 2. Khởi tạo AI
    model = YOLO("best.pt")
    names = model.model.names
    cap = cv2.VideoCapture(0)
    count = 0

    CONFIDENCE_THRESHOLD = 0.6  
    MIN_AREA = 2000    
    MAX_AREA = 150000  

    last_email_time = 0
    EMAIL_COOLDOWN = 60  
    REQUIRED_FIRE_TIME = 10.0  
    TOLERANCE_TIME = 1.5       

    first_fire_time = 0.0
    last_fire_time = 0.0

    print("🚀 HỆ THỐNG TÍCH HỢP AI & HARDWARE BẮT ĐẦU HOẠT ĐỘNG...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        count += 1
        if count % 2 != 0: continue
        frame = cv2.resize(frame, (1020, 500))

        results = model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
        fire_in_current_frame = False

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)
            masks = results[0].masks.xy if results[0].masks is not None else []
            overlay = frame.copy()

            for box, track_id, class_id, mask in zip(boxes, track_ids, class_ids, masks):
                c = names[class_id]
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                
                if area < MIN_AREA or area > MAX_AREA: continue  

                if len(mask) > 0:
                    mask_arr = np.array(mask, dtype=np.int32).reshape((-1, 1, 2)) 
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.fillPoly(overlay, [mask_arr], color=(0, 0, 255))
                    cvzone.putTextRect(frame, f'{c} {track_id}', (x1, y1), 1, 1)

                    if c in ['fire', 'smoke']:
                        fire_in_current_frame = True

            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # LOGIC THỜI GIAN 10 GIÂY
        current_time = time.time()
        if fire_in_current_frame:
            if first_fire_time == 0.0: first_fire_time = current_time
            last_fire_time = current_time 
            elapsed_time = current_time - first_fire_time
            
            cv2.putText(frame, f"Dang xac minh: {int(elapsed_time)}/10s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            if elapsed_time >= REQUIRED_FIRE_TIME:
                shared_state["yolo_fire_confirmed"] = True # Bật cờ cho Phần cứng kích Relay
                
                if current_time - last_email_time > EMAIL_COOLDOWN:
                    threading.Thread(target=send_email, args=(frame.copy(),), daemon=True).start()
                    last_email_time = current_time
                    first_fire_time = 0.0 
        else:
            if first_fire_time != 0.0:
                if current_time - last_fire_time > TOLERANCE_TIME:
                    first_fire_time = 0.0
                    shared_state["yolo_fire_confirmed"] = False # Tắt cờ khi lửa tắt

        # HIỂN THỊ THÔNG SỐ CẢM BIẾN LÊN MÀN HÌNH CAMERA (GIAO TIẾP 2 CHIỀU)
        cv2.putText(frame, f"Temp: {shared_state['current_temp']} C | VOC: {shared_state['current_voc']} ppm", (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if shared_state["sensor_fire_confirmed"]:
            cv2.putText(frame, "SENSORS DETECTED HAZARD!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("HETHONG PCCC - YOLOV11 + HARDWARE", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()