import subprocess
import time
import sys
import os

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
PYTHON_PATH = "/home/pi/env_pccc/bin/python"
PROJECT_DIR = "/home/pi/Desktop/Main_Project_Code_Python"

FILE_VISION = f"{PROJECT_DIR}/yolo_vision.py"
FILE_HARDWARE = f"{PROJECT_DIR}/sensor_hardware.py"

def start_system():
    print("="*65)
    print("🚀 ĐANG KHỞI CHẠY HỆ THỐNG PCCC ĐA TIẾN TRÌNH (MULTI-PROCESS)")
    print("="*65)

    # 1. Cấp quyền hiển thị màn hình (Fix triệt để lỗi OpenCV trên Pi 5)
    os.environ["DISPLAY"] = ":0"
    os.environ["QT_QPA_PLATFORM"] = "xcb" # Bảo vệ kép cho cửa sổ VNC

    # 2. Khởi tạo/Reset file trạng thái trên RAM về 0 (An toàn)
    try:
        with open("/dev/shm/fire_status.txt", "w") as f:
            f.write("0")
    except Exception as e:
        print(f"⚠️ Không thể tạo file trạng thái RAM: {e}")

    # 3. Kích hoạt tiến trình Camera & AI (YOLO)
    print("⏳ Đang bật Camera và tải mô hình YOLO (Cần khoảng 8s)...")
    proc_vision = subprocess.Popen([PYTHON_PATH, FILE_VISION])
    
    # Đợi 8 giây để AI Load xong bộ não vào RAM và hiện cửa sổ Camera lên trước
    time.sleep(8) 

    # 4. Kích hoạt tiến trình Cảm biến & E-Ra
    print("⏳ Đang bật Cảm biến phần cứng và Kết nối MQTT...")
    proc_hardware = subprocess.Popen([PYTHON_PATH, FILE_HARDWARE])

    print("\n✅ HỆ THỐNG ĐÃ SẴN SÀNG HOẠT ĐỘNG! Nhấn Ctrl+C để dừng tất cả.\n")

    try:
        # Vòng lặp chính để file Master giám sát 2 tiến trình con
        while True:
            time.sleep(1)
            
            # Kiểm tra xem có file nào bị crash/văng lỗi đột ngột không
            if proc_vision.poll() is not None:
                print("\n⚠️ CẢNH BÁO: Tiến trình YOLO Vision đã dừng đột ngột!")
                break 
            
            if proc_hardware.poll() is not None:
                print("\n⚠️ CẢNH BÁO: Tiến trình Sensor Hardware đã dừng đột ngột!")
                break
                
    except KeyboardInterrupt:
        print("\n\n🛑 NHẬN LỆNH TỪ NGƯỜI DÙNG: Đang ngắt hệ thống...")
    finally:
        # Đảm bảo tắt sạch sẽ các tiến trình ngầm khi file Master bị tắt
        print("🧹 Đang dọn dẹp bộ nhớ và ngắt kết nối Camera/Rơ-le...")
        proc_vision.terminate()
        proc_hardware.terminate()
        
        # Reset file trạng thái về 0 lần cuối
        try:
            with open("/dev/shm/fire_status.txt", "w") as f:
                f.write("0")
        except:
            pass
            
        print("👋 Đã tắt toàn bộ hệ thống an toàn!")

if __name__ == "__main__":
    start_system()