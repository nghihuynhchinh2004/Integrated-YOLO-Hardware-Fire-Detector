import time, json, os, csv
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

from ml_predictor import predict_hazard
from DS18B20_Temperature_Sensor import get_temperature_data
from ADS_1115_Air_Sensor import get_air_sensor_data
from fusion_decision import fusion_decision
from severity_estimator import estimate_severity
from temporal_confirmation import TemporalConfirmation
from relay_control import setup_relays, write_actuators, cleanup_relays
from era_mqtt_client import setup_mqtt, publish_telemetry, stop_mqtt

# ================= KẾT NỐI YOLO =================
STATUS_FILE = "/dev/shm/fire_status.txt"
def get_yolo_status():
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "r") as f: return f.read().strip() == "1"
    except: return False
    return False

# ================= HÀM GỬI EMAIL CẢNH BÁO =================
def send_alert_email(subject, body):
    try:
        # Nhớ trỏ đường dẫn tuyệt đối để chạy ngầm không bị lỗi
        with open('/home/pi/Desktop/Main_Project_Code_Python/emailpass.txt', 'r') as f:
            lines = f.read().splitlines()
            sender_email = lines[0]
            password = lines[1]
            receiver_email = lines[2]

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.send_message(msg)
        print(" 📧 [THÀNH CÔNG] Đã gửi Email cảnh báo đến chủ nhà!")
    except Exception as e:
        print(f" ⚠️ [LỖI] Không thể gửi Email: {e}")

# ================= TEMPORAL FILTER =================
temporal_filter = TemporalConfirmation(
    safe_confirm_count=5,
    hazard_confirm_count=3
)

# ================= CONFIG =================
LOG_DIR = "logs"
JSONL_FILE = f"{LOG_DIR}/fire_realtime_ml.jsonl"
CSV_FILE = f"{LOG_DIR}/fire_realtime_ml.csv"
LOOP_INTERVAL = 1.0

setup_relays()
setup_mqtt()

os.makedirs(LOG_DIR, exist_ok=True)

# ================= HAZARD NAME =================
def hazard_name(label: int) -> str:
    return ["SAFE", "GAS_LEAK", "VOC_CHEMICAL", "SMOKE_AIR", "FIRE"][label]

# ================= CSV SETUP =================
csv_exists = os.path.isfile(CSV_FILE)
csv_file = open(CSV_FILE, mode="a", newline="")
csv_writer = csv.writer(csv_file)

if not csv_exists:
    csv_writer.writerow([
        "date", "time", "temperature_C", "temp_status", "temp_trend", "heat_rise",
        "mq2_HI", "mq135_HI", "voc_ppm", "severity_score", "severity_level", "action_level",
        "ml_hazard", "fusion_hazard", "fusion_reason", 
        "yolo_status", "final_cross_decision" # 2 Cột bổ sung cho AI Camera
    ])

# ================= EMAIL COOLDOWN SETUP =================
last_gas_email_time = 0
last_fire_email_time = 0
EMAIL_COOLDOWN = 60 # Giới hạn: 60 giây mới gửi 1 mail cho cùng 1 sự kiện

print("\n=== FIRE DETECTION SYSTEM – REALTIME CROSS-VERIFICATION MODE ===\n")

try:
    while True:
        # ---------- READ SENSORS ----------
        temp = get_temperature_data()
        air = get_air_sensor_data()

        now = datetime.now()
        ts_epoch = time.time()

        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # ---------- FEATURES ----------
        temp_c = round(temp["temperature_c"], 2)
        mq2_hi = round(air["mq2"]["HI"], 3)
        mq135_hi = round(air["mq135"]["HI"], 3)
        voc_ppm = int(air["voc"]["ppm"])
        temp_status = temp["status"]                 # safe / warning / danger
        temp_rise = temp["heat_rise_status"]          # normal / caution / high_risk
        temp_trend = temp["trend"]                    # rising / falling / stable
        
        severity_result = estimate_severity(
            temperature_c=temp_c, heat_rise=temp_rise, mq2_hi=mq2_hi,
            mq135_hi=mq135_hi, voc_ppm=voc_ppm
        )
        
        severity_score = severity_result["severity_score"]
        severity_level = severity_result["severity_level"]
        action_level = severity_result["action_level"]
        severity_reason = severity_result["reason"]

        # ---------- ML PREDICTION ----------
        ml_label, ml_hazard = predict_hazard(
            temp_c=temp_c, mq2_hi=mq2_hi, mq135_hi=mq135_hi, voc_ppm=voc_ppm
        )
        
        # ---------- FUSION DECISION ----------
        fusion_result = fusion_decision(
            temp_c=temp_c, temp_status=temp_status, temp_rise=temp_rise,
            mq2_hi=mq2_hi, mq135_hi=mq135_hi, voc_ppm=voc_ppm,
            ml_label=ml_label, severity_score=severity_score,
            severity_level=severity_level, action_level=action_level,
        )
        fusion_label = fusion_result["label"]
        fusion_hazard = fusion_result["hazard"]
        fusion_reason = fusion_result["reason"]
        fusion_source = fusion_result["source"]
        fusion_urgency = fusion_result["urgency"]
        
        # ---------- TEMPORAL RESULT ----------
        temporal_result = temporal_filter.update(fusion_result)

        confirmed_label = temporal_result["confirmed_label"]
        confirmed_hazard = temporal_result["confirmed_hazard"]
        confirmed_reason = temporal_result["confirmed_reason"]
        relay_active_sensor = temporal_result["relay_active"]
        actuator = temporal_result["actuator"].copy()

        # ==================================================
        # ---------- CHỐT KẾT QUẢ VỚI YOLO CAMERA ----------
        # ==================================================
        yolo_fire = get_yolo_status()
        yolo_str = "FIRE" if yolo_fire else "SAFE"
        
        final_hazard = confirmed_hazard
        final_reason = confirmed_reason

        # Ma trận bù trừ chéo
        if yolo_fire and confirmed_hazard == "FIRE":
            final_reason = "ĐỒNG THUẬN TUYỆT ĐỐI (AI Camera + Cảm biến)"
            final_hazard = "FIRE"
            actuator.update({"buzzer": True, "mist": True, "emergency": True})
        elif yolo_fire and confirmed_hazard != "FIRE":
            final_reason = "CAMERA BÁO TRƯỚC (Cảm biến chưa tới ngưỡng)"
            final_hazard = "FIRE"
            actuator.update({"buzzer": True, "mist": True, "emergency": True})
        elif not yolo_fire and confirmed_hazard == "FIRE":
            final_reason = "CẢM BIẾN BÁO CHÁY (Camera bị che khuất/điểm mù)"
            final_hazard = "FIRE"
            actuator.update({"buzzer": True, "mist": True, "emergency": True})

        # Thực thi Rơ-le với kết quả đã gộp
        write_actuators(actuator)
        final_relay_active = any(actuator.values())

        # ==================================================
        # ---------- KỊCH BẢN GỬI EMAIL CẢNH BÁO ----------
        # ==================================================
        current_time = time.time()

        # 1. Kịch bản Rò rỉ khí GAS
        if final_hazard == "GAS_LEAK" or confirmed_hazard == "GAS_LEAK":
            if current_time - last_gas_email_time > EMAIL_COOLDOWN:
                subject = "🚨 CẢNH BÁO KHẨN: RÒ RỈ KHÍ GAS 🚨"
                body = f"Hệ thống phát hiện nồng độ khí Gas nguy hiểm trong khu vực!\n\nChi tiết:\n- Mức độ Gas (MQ-2): {mq2_hi}\n- Cảnh báo từ: Cảm biến phần cứng\n- Đã tự động kích hoạt Rơ-le an toàn.\n\nVui lòng kiểm tra hiện trường ngay lập tức!"
                print(" ✉️ Phát hiện GAS - Đang gửi Email...")
                send_alert_email(subject, body)
                last_gas_email_time = current_time

        # 2. Kịch bản Hỏa hoạn (Có so sánh chéo)
        if final_hazard == "FIRE":
            if current_time - last_fire_email_time > EMAIL_COOLDOWN:
                subject = "🔥 CẢNH BÁO KHẨN CẤP: PHÁT HIỆN HỎA HOẠN 🔥"
                body = f"Hệ thống giám sát vừa phát hiện sự cố HỎA HOẠN!\n\n=== ĐÁNH GIÁ CHÉO (CROSS-VERIFICATION) ===\n- KẾT LUẬN: {final_reason}\n- Tín hiệu Camera YOLO: {yolo_str}\n- Tín hiệu Cảm biến: {confirmed_hazard}\n\n=== THÔNG SỐ VẬT LÝ ===\n- Nhiệt độ: {temp_c}°C ({temp_status})\n- Khói (MQ-135): {mq135_hi}\n\nToàn bộ hệ thống chữa cháy đã được kích hoạt!"
                print(" ✉️ Phát hiện LỬA - Đang gửi Email...")
                send_alert_email(subject, body)
                last_fire_email_time = current_time

        # ---------- JSONL LOG (ML INFERENCE) ----------
        ml_record = {
            "ts": round(ts_epoch, 3), "temp_c": float(temp_c), "mq2_hi": float(mq2_hi),
            "mq135_hi": float(mq135_hi), "voc_ppm": int(voc_ppm), "ml_hazard_label": int(ml_label)
        }
        with open(JSONL_FILE, "a") as jf:
            jf.write(json.dumps(ml_record, ensure_ascii=False) + "\n")

        # ---------- CSV LOG (HUMAN READABLE) ----------
        csv_writer.writerow([
            date_str, time_str, temp_c, temp_status, temp_trend, temp_rise,
            mq2_hi, mq135_hi, voc_ppm, severity_score, severity_level, action_level,
            ml_hazard, fusion_hazard, fusion_reason, yolo_str, final_reason
        ])
        csv_file.flush()

        # ---------- ERA PAYLOAD ----------
        era_payload = {
            "system_status": final_hazard,
            "temp_c": temp_c, "temp_status": temp_status, "temp_trend": temp_trend, "heat_rise": temp_rise,
            "mq2_hi": mq2_hi, "mq135_hi": mq135_hi, "voc_ppm": voc_ppm,
            "severity_score": severity_score, "severity_level": severity_level, "action_level": action_level,
            "ml_hazard": ml_hazard, "fusion_hazard": fusion_hazard, "fusion_source": fusion_source, "fusion_urgency": fusion_urgency,
            "temporal_hazard": confirmed_hazard, "streak": temporal_result["streak"], "required_count": temporal_result["required_count"],
            "yolo_status": yolo_str,
            "relay_active": final_relay_active,
            "buzzer": actuator.get("buzzer"), "fan": actuator.get("fan"), "mist": actuator.get("mist"), "emergency": actuator.get("emergency")
        }
        mqtt_ok = publish_telemetry(era_payload)
        print(f" ☁️ E-Ra MQTT → {'SENT' if mqtt_ok else 'NOT CONNECTED'}")
        
        # ---------- TERMINAL OUTPUT ----------
        print("------------------------------------------------")
        print(f" 🌡 Temp : {temp_c} °C | {temp_status} | {temp_trend} | {temp_rise}")
        print(f" 🫁 MQ2  : {mq2_hi} | MQ135 : {mq135_hi}")
        print(f" 💨 VOC  : {voc_ppm} ppm")
        print(f" SEVERITY → {severity_score}/100 | {severity_level} | {action_level}")
        print(f" Severity reason → {severity_reason}")
        print(f" 🤖 ML     → {ml_hazard}\n")
        print(f" 🧠 Result (Fusion based) → {fusion_hazard} ({fusion_source}, {fusion_urgency})")
        print(f" Fusion Reason : {fusion_reason}\n")
        print(f" ⏱ TEMPORAL    → {confirmed_hazard} | streak={temporal_result['streak']}/{temporal_result['required_count']}")
        print(f" 🔌 RELAY (SENSOR) → {'ON' if relay_active_sensor else 'OFF'}")
        
        # In thêm bảng chốt hạ của cả hệ thống
        print("================================================")
        print(f" 📷 YOLO STATUS    → {yolo_str}")
        print(f" 🎯 CROSS-DECISION → {final_hazard}")
        print(f" 📌 Reason         : {final_reason}")
        print("================================================")
        
        print(f" Buzzer   → {'ON' if actuator.get('buzzer') else 'OFF'}")
        print(f" Fan      → {'ON' if actuator.get('fan') else 'OFF'}")
        print(f" Mist     → {'ON' if actuator.get('mist') else 'OFF'}")
        print(f" 🚨 Emergency → {'ON' if actuator.get('emergency') else 'OFF'}")
        print("------------------------------------------------")

        time.sleep(LOOP_INTERVAL)

except KeyboardInterrupt:
    print("\n⛔ System stopped")

finally:
    cleanup_relays()
    csv_file.close()
    stop_mqtt()