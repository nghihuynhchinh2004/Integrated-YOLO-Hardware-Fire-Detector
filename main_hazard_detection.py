import time, json, os, csv
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

from ml_predictor import predict_hazard
from DS18B20_Temperature_Sensor import get_temperature_data
from ADS_1115_Air_Sensor import get_air_sensor_data
from fusion_decision import fusion_decision
from severity_estimator import estimate_severity
from temporal_confirmation import TemporalConfirmation
from relay_control import setup_relays, write_actuators, cleanup_relays
from era_mqtt_client import setup_mqtt, publish_telemetry, stop_mqtt

# ================= ĐỌC TRẠNG THÁI TỪ YOLO =================
STATUS_FILE = "/dev/shm/fire_status.txt"
IMAGE_FILE = "/dev/shm/latest_frame.jpg"

def get_yolo_status():
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "r") as f: 
                val = f.read().strip()
                if val in ["FIRE", "SMOKE", "SAFE"]:
                    return val
    except: pass
    return "SAFE"

# ================= HÀM GỬI EMAIL KÈM ẢNH BẰNG MULTIPART =================
def send_alert_email(subject, body, attach_image=True):
    try:
        with open('/home/pi/Desktop/Main_Project_Code_Python/emailpass.txt', 'r') as f:
            lines = f.read().splitlines()
            sender_email = lines[0]
            password = lines[1]
            receiver_email = lines[2]

        # Khởi tạo form Email cho phép đính kèm file
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email

        # Gắn phần text (nội dung)
        msg.attach(MIMEText(body, 'plain'))

        # Gắn phần ảnh (chụp từ RAM)
        if attach_image and os.path.exists(IMAGE_FILE):
            with open(IMAGE_FILE, 'rb') as img_f:
                img_data = img_f.read()
                image = MIMEImage(img_data, name="Camera_Snapshot.jpg")
                msg.attach(image)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.send_message(msg)
        print(f" 📧 [THÀNH CÔNG] Đã gửi Email kèm ảnh báo cáo: {subject}")
    except Exception as e:
        print(f" ⚠️ [LỖI] Không thể gửi Email: {e}")

# ================= CONFIG =================
temporal_filter = TemporalConfirmation(safe_confirm_count=5, hazard_confirm_count=3)
LOG_DIR = "logs"
JSONL_FILE = f"{LOG_DIR}/fire_realtime_ml.jsonl"
CSV_FILE = f"{LOG_DIR}/fire_realtime_ml.csv"
LOOP_INTERVAL = 1.0

setup_relays()
setup_mqtt()
os.makedirs(LOG_DIR, exist_ok=True)

csv_exists = os.path.isfile(CSV_FILE)
csv_file = open(CSV_FILE, mode="a", newline="")
csv_writer = csv.writer(csv_file)
if not csv_exists:
    csv_writer.writerow([
        "date", "time", "temperature_C", "temp_status", "temp_trend", "heat_rise",
        "mq2_HI", "mq135_HI", "voc_ppm", "severity_score", "severity_level", "action_level",
        "ml_hazard", "fusion_hazard", "fusion_reason", "yolo_status", "final_cross_decision" 
    ])

email_cooldowns = {
    "FIRE_CONFIRM": 0, "SMOKE_CONFIRM": 0, 
    "SUSPECT_FIRE": 0, "SUSPECT_SMOKE": 0, "GAS_LEAK": 0
}
EMAIL_DELAY = 60 

print("\n=== FIRE DETECTION SYSTEM – STRICT CROSS-VERIFICATION MODE ===\n")

try:
    while True:
        temp = get_temperature_data()
        air = get_air_sensor_data()
        now = datetime.now()
        ts_epoch = time.time()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        temp_c = round(temp["temperature_c"], 2)
        mq2_hi = round(air["mq2"]["HI"], 3)
        mq135_hi = round(air["mq135"]["HI"], 3)
        voc_ppm = int(air["voc"]["ppm"])
        temp_status = temp["status"]                 
        temp_rise = temp["heat_rise_status"]          
        temp_trend = temp["trend"]                    
        
        severity_result = estimate_severity(
            temperature_c=temp_c, heat_rise=temp_rise, mq2_hi=mq2_hi,
            mq135_hi=mq135_hi, voc_ppm=voc_ppm
        )
        severity_score = severity_result["severity_score"]
        severity_level = severity_result["severity_level"]
        action_level = severity_result["action_level"]
        
        ml_label, ml_hazard = predict_hazard(temp_c, mq2_hi, mq135_hi, voc_ppm)
        
        fusion_result = fusion_decision(
            temp_c=temp_c, temp_status=temp_status, temp_rise=temp_rise,
            mq2_hi=mq2_hi, mq135_hi=mq135_hi, voc_ppm=voc_ppm,
            ml_label=ml_label, severity_score=severity_score,
            severity_level=severity_level, action_level=action_level,
        )
        
        temporal_result = temporal_filter.update(fusion_result)
        confirmed_hazard = temporal_result["confirmed_hazard"] 

        # =======================================================================
        # ================== MA TRẬN ĐỒNG THUẬN QUYẾT ĐỊNH ======================
        # =======================================================================
        yolo_str = get_yolo_status() 
        
        yolo_fire = (yolo_str == "FIRE")
        yolo_smoke = (yolo_str == "SMOKE")
        sensor_fire = (confirmed_hazard == "FIRE")
        sensor_smoke = (confirmed_hazard == "SMOKE_AIR")

        actuator = {"buzzer": False, "fan": False, "mist": False, "emergency": False}
        final_hazard = "SAFE"
        final_reason = "Bình thường"
        email_to_send = None

        if yolo_fire and sensor_fire:
            final_hazard = "FIRE"
            final_reason = "ĐỒNG THUẬN CÓ LỬA (Camera + Cảm biến)"
            actuator.update({"buzzer": True, "mist": True, "emergency": True}) 
            email_to_send = "FIRE_CONFIRM"

        elif yolo_smoke and sensor_smoke:
            final_hazard = "SMOKE"
            final_reason = "ĐỒNG THUẬN CÓ KHÓI (Camera + Cảm biến)"
            actuator.update({"buzzer": True, "fan": True, "emergency": True}) 
            email_to_send = "SMOKE_CONFIRM"

        elif yolo_fire and not sensor_fire:
            final_hazard = "SUSPECT_FIRE"
            final_reason = "NGHI NGỜ CÓ LỬA (Chỉ Camera phát hiện, Cảm biến an toàn)"
            email_to_send = "SUSPECT_FIRE"

        elif not yolo_fire and sensor_fire:
            final_hazard = "SUSPECT_FIRE"
            final_reason = "NGHI NGỜ CÓ LỬA (Chỉ Cảm biến thấy nhiệt cao, Camera bị che khuất)"
            email_to_send = "SUSPECT_FIRE"

        elif yolo_smoke and not sensor_smoke:
            final_hazard = "SUSPECT_SMOKE"
            final_reason = "NGHI NGỜ CÓ KHÓI (Chỉ Camera phát hiện)"
            email_to_send = "SUSPECT_SMOKE"

        elif not yolo_smoke and sensor_smoke:
            final_hazard = "SUSPECT_SMOKE"
            final_reason = "NGHI NGỜ CÓ KHÓI/BỤI MỊN (Chỉ Cảm biến ngửi thấy)"
            email_to_send = "SUSPECT_SMOKE"

        elif confirmed_hazard == "GAS_LEAK":
            final_hazard = "GAS_LEAK"
            final_reason = "RÒ RỈ KHÍ GAS ĐỘC HẠI (Từ Cảm biến MQ-2)"
            actuator.update({"buzzer": True, "fan": True}) 
            email_to_send = "GAS_LEAK"

        write_actuators(actuator)
        final_relay_active = any(actuator.values())

        # ================= GỬI EMAIL THÔNG MINH =================
        current_time = time.time()
        if email_to_send and (current_time - email_cooldowns[email_to_send] > EMAIL_DELAY):
            
            # --- FORM BÁO CÁO THÔNG SỐ CHUẨN XÁC ---
            sensor_info = f"""
=== THÔNG SỐ VẬT LÝ GHI NHẬN ĐƯỢC ===
- Nhiệt độ hiện tại: {temp_c}°C ({temp_status})
- Khí dễ cháy (MQ-2): {mq2_hi} 
- Mức độ Khói/Bụi (MQ-135): {mq135_hi}
- Nồng độ VOC độc hại: {voc_ppm} ppm
=====================================
            """
            
            subject = ""
            body = ""
            
            if email_to_send == "FIRE_CONFIRM":
                subject = "🔥 BÁO ĐỘNG ĐỎ: ĐÃ XÁC NHẬN HỎA HOẠN 🔥"
                body = f"Hệ thống đã XÁC NHẬN CÓ HỎA HOẠN từ cả Camera AI và Cảm biến!\n\n- Đã tự động kích hoạt CÒI HÚ và BƠM SƯƠNG chữa cháy.\n- Vui lòng xem ảnh hiện trường đính kèm bên dưới và sơ tán ngay lập tức!\n\n{sensor_info}"
            
            elif email_to_send == "SMOKE_CONFIRM":
                subject = "🌫 BÁO ĐỘNG: ĐÃ XÁC NHẬN CÓ KHÓI DÀY ĐẶC 🌫"
                body = f"Hệ thống đã XÁC NHẬN CÓ KHÓI từ cả Camera AI và Cảm biến!\n\n- Đã tự động kích hoạt CÒI HÚ và QUẠT HÚT KHÓI.\n- Vui lòng xem ảnh chụp đính kèm và kiểm tra hiện trường.\n\n{sensor_info}"
            
            elif email_to_send == "SUSPECT_FIRE":
                subject = "⚠️ CẢNH BÁO NGHI NGỜ: PHÁT HIỆN DẤU HIỆU LỬA ⚠️"
                body = f"Hệ thống phát hiện dấu hiệu nghi ngờ có Lửa.\n\n- Tình trạng: {final_reason}\n- Để đảm bảo an toàn, hệ thống chữa cháy CHƯA TỰ ĐỘNG BẬT.\n- Người dùng vui lòng kiểm tra Camera xác minh. Nếu có cháy thật, hãy điều khiển bật bơm qua ứng dụng E-Ra!\n\n{sensor_info}"
            
            elif email_to_send == "SUSPECT_SMOKE":
                subject = "⚠️ CẢNH BÁO NGHI NGỜ: PHÁT HIỆN DẤU HIỆU KHÓI ⚠️"
                body = f"Hệ thống phát hiện dấu hiệu nghi ngờ có Khói.\n\n- Tình trạng: {final_reason}\n- Quạt hút và còi CHƯA TỰ ĐỘNG BẬT để tránh báo giả.\n- Vui lòng tự kiểm tra và điều khiển thủ công qua ứng dụng E-Ra nếu cần.\n\n{sensor_info}"
            
            elif email_to_send == "GAS_LEAK":
                subject = "🚨 CẢNH BÁO KHẨN: RÒ RỈ KHÍ GAS 🚨"
                body = f"Phát hiện rò rỉ khí Gas nguy hiểm! Đã bật Quạt tản khí. Vui lòng kiểm tra khóa Gas ngay!\n\n{sensor_info}"

            # Gọi hàm gửi Email (đính kèm ảnh)
            send_alert_email(subject, body, attach_image=True)
            email_cooldowns[email_to_send] = current_time

        # ---------- LOGGING VÀ MQTT ----------
        ml_record = {
            "ts": round(ts_epoch, 3), "temp_c": float(temp_c), "mq2_hi": float(mq2_hi),
            "mq135_hi": float(mq135_hi), "voc_ppm": int(voc_ppm), "ml_hazard_label": int(ml_label)
        }
        with open(JSONL_FILE, "a") as jf:
            jf.write(json.dumps(ml_record, ensure_ascii=False) + "\n")

        csv_writer.writerow([
            date_str, time_str, temp_c, temp_status, temp_trend, temp_rise,
            mq2_hi, mq135_hi, voc_ppm, severity_score, severity_level, action_level,
            ml_hazard, fusion_result["hazard"], fusion_result["reason"], yolo_str, final_reason
        ])
        csv_file.flush()

        era_payload = {
            "system_status": final_hazard,
            "temp_c": temp_c, "temp_status": temp_status, "temp_trend": temp_trend, "heat_rise": temp_rise,
            "mq2_hi": mq2_hi, "mq135_hi": mq135_hi, "voc_ppm": voc_ppm,
            "severity_score": severity_score, "severity_level": severity_level,
            "yolo_status": yolo_str,
            "relay_active": final_relay_active,
            "buzzer": actuator.get("buzzer"), "fan": actuator.get("fan"), "mist": actuator.get("mist")
        }
        publish_telemetry(era_payload)
        
        # ---------- IN RA MÀN HÌNH THEO DÕI ----------
        print("================================================")
        print(f" 🌡 Nhiệt độ: {temp_c} °C | Khí Gas MQ2: {mq2_hi} | Khói MQ135: {mq135_hi}")
        print(f" 🧠 Đánh giá Cảm biến → {confirmed_hazard}")
        print(f" 📷 Tín hiệu YOLO AI  → {yolo_str}")
        print("------------------------------------------------")
        print(f" 🎯 KẾT LUẬN CUỐI CÙNG → {final_hazard}")
        print(f" 📌 {final_reason}")
        print(f" 🔌 THIẾT BỊ (Quạt: {'ON' if actuator.get('fan') else 'OFF'}, "
              f"Bơm: {'ON' if actuator.get('mist') else 'OFF'}, Còi: {'ON' if actuator.get('buzzer') else 'OFF'})")
        print("================================================\n")

        time.sleep(LOOP_INTERVAL)

except KeyboardInterrupt:
    print("\n⛔ System stopped")

finally:
    cleanup_relays()
    csv_file.close()
    stop_mqtt()