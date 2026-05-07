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

# Đọc thông tin email
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

            # Lấy thời gian thực tế của hệ thống
            now = datetime.now()
            time_str = now.strftime("%H:%M:%S, ngày %d/%m/%Y")

            # NỘI DUNG EMAIL
            body = f"""THÔNG BÁO: HỆ THỐNG PHÁT HIỆN SỰ CỐ!

Hệ thống báo cháy đã phát hiện dấu hiệu hỏa hoạn/khói vào thời điểm: {time_str}.

Bạn vui lòng gọi ngay cho Cảnh sát PCCC Thành phố Hồ Chí Minh theo thông báo sau:
- Số điện thoại báo cháy: 114
- Số điện thoại đường dây nóng: (028) 39 200 996

Hình ảnh camera ghi nhận tại hiện trường được đính kèm trong email này.

--------------------------------------------------
* Lưu ý: Đây là hệ thống cảnh báo tự động, vui lòng không reply (trả lời) mail này."""

            # Đính kèm phần văn bản tiếng Việt vào mail
            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            # Đính kèm hình ảnh
            _, buffer = cv2.imencode('.jpg', frame)
            img_data = buffer.tobytes()  

            img = MIMEImage(img_data, name="Fire_Alert.jpg")
            img.add_header('Content-Disposition', 'attachment', filename="Fire_Alert.jpg")
            msg.attach(img)

            server.send_message(msg)
            print(f"✅ Đã gửi Email cảnh báo kèm thời gian ({time_str}) thành công!")
            break  

        except smtplib.SMTPException as e:
            attempt += 1
            print(f"❌ Lỗi gửi email: {e}. Đang thử lại lần {attempt}/{max_retries}...")
            time.sleep(delay)  

        finally:
            try:
                server.quit()  
            except:
                pass  

    if attempt == max_retries:
        print("🚨 Đã hết số lần thử. Không thể gửi email.")

# Gọi "bộ não" YOLO đã train xong
model = YOLO("best.pt")
names = model.model.names

# Chạy file video
cap = cv2.VideoCapture("vid.mp4")
count = 0

# ==========================================
# CÁC THÔNG SỐ TỐI ƯU HÓA CHO VIDEO
# ==========================================
CONFIDENCE_THRESHOLD = 0.6  
MIN_AREA = 1500    
MAX_AREA = 150000  

last_email_real_time = 0
EMAIL_COOLDOWN = 60  # Đợi 60s (thời gian thực) mới gửi mail tiếp

# LOGIC ĐẾM THỜI GIAN THEO "ĐỒNG HỒ CỦA VIDEO"
REQUIRED_FIRE_TIME = 10.0  # Yêu cầu phát hiện cháy liên tục 10 giây trong video
TOLERANCE_TIME = 1.5       # Châm chước 1.5 giây

first_fire_video_time = 0.0
last_fire_video_time = 0.0
# ==========================================

print("🚀 HỆ THỐNG TEST VIDEO BẮT ĐẦU HOẠT ĐỘNG...")

while True:
    ret, frame = cap.read()
    
    # Khi tua lại video, BẮT BUỘC phải reset lại bộ đếm thời gian
    if not ret:
        print("🎬 Đã phát hết, đang tự động tua lại video...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
        first_fire_video_time = 0.0
        last_fire_video_time = 0.0
        continue

    count += 1
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    # TẮT terminal spam bằng verbose=False
    results = model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, verbose=False)

    fire_in_current_frame = False

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = [-1] * len(boxes) 

        masks = results[0].masks
        if masks is not None:
            masks = masks.xy
            overlay = frame.copy()

            for box, track_id, class_id, mask in zip(boxes, track_ids, class_ids, masks):
                c = names[class_id]
                x1, y1, x2, y2 = box

                area = (x2 - x1) * (y2 - y1)
                
                if area < MIN_AREA or area > MAX_AREA:
                    continue  

                if mask.size > 0:
                    mask_arr = np.array(mask, dtype=np.int32).reshape((-1, 1, 2)) 

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.fillPoly(overlay, [mask_arr], color=(0, 0, 255))
                    cvzone.putTextRect(frame, f'{track_id}', (x2, y2), 1, 1)
                    cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

                    if c in ['fire', 'smoke']:
                        fire_in_current_frame = True

            alpha = 0.5  
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # ==========================================
    # LOGIC XỬ LÝ THỜI GIAN DỰA VÀO TIẾN ĐỘ VIDEO
    # ==========================================
    # Lấy thời gian hiện tại của video (tính bằng mili-giây -> đổi ra giây)
    current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    if fire_in_current_frame:
        if first_fire_video_time == 0.0:
            first_fire_video_time = current_video_time
            print("⏳ Phát hiện dấu hiệu cháy, bắt đầu đếm ngược 10 giây (theo video)...")

        last_fire_video_time = current_video_time 

        elapsed_time = current_video_time - first_fire_video_time
        
        cv2.putText(frame, f"Dang xac minh: {int(elapsed_time)}/10s", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        if elapsed_time >= REQUIRED_FIRE_TIME:
            # Gửi mail dựa trên thời gian thực của máy tính để chống spam
            current_real_time = time.time()
            if current_real_time - last_email_real_time > EMAIL_COOLDOWN:
                print(f"\n🚨 XÁC NHẬN CHÁY LIÊN TỤC {REQUIRED_FIRE_TIME}s! ĐANG GỬI MAIL...")
                email_thread = threading.Thread(target=send_email, args=(frame.copy(),))
                email_thread.daemon = True 
                email_thread.start()
                
                last_email_real_time = current_real_time
                first_fire_video_time = 0.0 
    else:
        if first_fire_video_time != 0.0:
            time_since_last_seen = current_video_time - last_fire_video_time
            
            if time_since_last_seen > TOLERANCE_TIME:
                print("✅ Dấu hiệu cháy đã biến mất. Hủy bộ đếm.")
                first_fire_video_time = 0.0

    cv2.imshow("FRAME", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()