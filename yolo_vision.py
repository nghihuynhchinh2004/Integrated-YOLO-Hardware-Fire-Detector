import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import time
import os
from picamera2 import Picamera2

# ================= CẤU HÌNH HIỂN THỊ TRÊN RASPBERRY PI 5 =================
# Ép buộc Qt dùng X11 để cửa sổ cam hiện lên trên VNC (Fix lỗi màn hình đen)
os.environ["QT_QPA_PLATFORM"] = "xcb"
print("🎥 Đang kích hoạt chế độ hiển thị X11 cho Camera...")

# ================= CƠ CHẾ GIAO TIẾP VỚI SENSOR (GHI XUỐNG RAM) =================
# Đây là file "nháp" nằm trên RAM để báo cáo kết quả sang file hardware
STATUS_FILE = "/dev/shm/fire_status.txt"

def update_status(val):
    try:
        # Ghi đè số 0 (Không cháy) hoặc 1 (Có cháy đủ 10s) vào file RAM
        with open(STATUS_FILE, "w") as f:
            f.write(str(val))
    except: pass

# Lúc mới bật, chốt hạ trạng thái là an toàn (0)
update_status(0) 

# ================= KHỞI TẠO CAMERA & YOLO =================
# Dùng đường dẫn tuyệt đối để đảm bảo an toàn khi chạy nền bằng file tổng
YOLO_MODEL_PATH = "/home/pi/Desktop/Main_Project_Code_Python/best.pt"

print("⏳ Đang tải mô hình YOLOv11 Segmentation...")
model = YOLO(YOLO_MODEL_PATH)
names = model.model.names

print("📸 Đang khởi động Picamera2...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1020, 500), "format": "BGR888"})
picam2.configure(config)
picam2.start()

# ================= KHỬ ÁM TÍM TỪ GỐC (PHẦN CỨNG) =================
print("🎨 Đang ép Cân bằng trắng (AWB) để khử lỗi ám tím/hồng...")
picam2.set_controls({
    "AwbEnable": True,
    # Điều chỉnh Gain (Khuếch đại) của kênh Đỏ và Xanh dương.
    # Thông số (1.5, 1.2) là mức cơ bản, nếu vẫn thấy tím, bạn có thể hạ số 1.5 xuống 1.2 hoặc 1.0.
    "ColourGains": (1.5, 1.2) 
})

# ================= CẤU HÌNH LOGIC AI =================
# Cấu hình ngưỡng AI và bộ lọc diện tích (Area Filter)
CONFIDENCE_THRESHOLD = 0.6  
MIN_AREA = 2000 
MAX_AREA = 150000  

# Cấu hình thời gian xác minh (Logic 10 giây)
REQUIRED_FIRE_TIME = 10.0   # Cần thấy lửa liên tục 10 giây
TOLERANCE_TIME = 1.5        # Cho phép mất dấu 1.5 giây (bù trừ nhiễu)

first_fire_time = 0.0
last_fire_time = 0.0
count = 0

print("\n▶️ Hệ thống YOLO Vision bắt đầu hoạt động. Nhấn 'q' để tắt.\n")

try:
    while True:
        # 1. Chụp ảnh từ camera (Đã được lọc tím từ phần cứng)
        frame = picam2.capture_array()
        
        # Bỏ qua khung hình để tăng tốc FPS (Chỉ xử lý khung hình chẵn)
        count += 1
        if count % 2 != 0: continue

        # 2. Chạy AI dò tìm và theo dõi (Track) khói/lửa
        results = model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
        fire_in_current_frame = False

        # 3. Xử lý kết quả AI
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes) 
            masks = results[0].masks

            # 4. Nếu thấy vật thể và có Mask (Vùng vẽ viền)
            if masks is not None:
                masks = masks.xy
                overlay = frame.copy() # Tạo màng phủ để tô màu Mask bán trong suốt
                
                for box, track_id, class_id, mask in zip(boxes, track_ids, class_ids, masks):
                    c = names[class_id]
                    x1, y1, x2, y2 = box
                    
                    # Bộ lọc diện tích (Area Filter) để loại bỏ nhiễu nhỏ
                    area = (x2 - x1) * (y2 - y1)
                    
                    if MIN_AREA <= area <= MAX_AREA and mask.size > 0:
                        # Bắt đầu vẽ
                        mask_arr = np.array(mask, dtype=np.int32).reshape((-1, 1, 2)) 
                        
                        # Vẽ khung vuông xanh (Bounding Box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Tô màng đỏ (Segmentation Mask) lên vùng khói lửa
                        cv2.fillPoly(overlay, [mask_arr], color=(0, 0, 255))
                        
                        # Ghi tên Lớp (Fire/Smoke)
                        cvzone.putTextRect(frame, f'{c}', (x1, y1), scale=1, thickness=1)
                        
                        # Xác nhận khung hình này có dấu hiệu cháy
                        if c in ['fire', 'smoke']:
                            fire_in_current_frame = True
                
                # Trộn màng màu đỏ (overlay) vào khung hình gốc (frame)
                alpha = 0.5 # Độ trong suốt 50%
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # ==================================================
        # ========== LOGIC XÁC MINH CHÁY 10 GIÂY ==========
        # ==================================================
        current_time = time.time()
        
        if fire_in_current_frame:
            # Lần đầu tiên thấy lửa
            if first_fire_time == 0.0:
                first_fire_time = current_time
            
            last_fire_time = current_time 
            elapsed_time = current_time - first_fire_time
            
            # Hiển thị thanh đếm ngược lên màn hình camera
            cv2.putText(frame, f"Xac minh: {int(elapsed_time)}/10s", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            # --- ĐÁNH GIÁ CHỐT HẠ ---
            if elapsed_time >= REQUIRED_FIRE_TIME:
                update_status(1) # Ghi số 1 xuống RAM cho file hardware biết
        else:
            # Khung hình hiện tại không thấy lửa
            if first_fire_time != 0.0:
                # Kiểm tra xem có phải chỉ bị mất dấu tạm thời do nhiễu không
                if (current_time - last_fire_time > TOLERANCE_TIME):
                    # Đã mất dấu quá 1.5 giây -> Reset bộ đếm, chốt trạng thái SAFE
                    first_fire_time = 0.0
                    update_status(0)
            elif first_fire_time == 0.0:
                # Không thấy gì cả -> Đảm bảo trạng thái luôn là SAFE (0)
                update_status(0)

        # 5. Hiển thị cửa sổ quan sát lên màn hình (Hội đồng xem ở đây)
        cv2.imshow("Hệ thống PCCC Thong minh - YOLO Vision", frame)
        
        # Nhấn 'q' để tắt cửa sổ cam
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f" ⚠️ Lỗi hệ thống Camera: {e}")

finally:
    # Giải phóng tài nguyên khi tắt máy
    picam2.stop()
    cv2.destroyAllWindows()
    update_status(0)
    print("\n🛑 Đã tắt hệ thống Camera an toàn.")