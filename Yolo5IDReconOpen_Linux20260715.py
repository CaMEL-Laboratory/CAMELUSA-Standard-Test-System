import os
import sys
import cv2
import time
import torch
import serial  # 👈 引入串口通信库
import face_recognition

# --- 在代码顶部定义变量 ---
last_trigger_time = 0  # 记录上次触发的时间
COOLDOWN_TIME = 30     # 冷却时间 30 秒
JOG_TIME = 1.5         # 点动时间 0.5 秒

# --- 运行环境设置 ---
yolov5_path = r"/home/nvidia/yolov5"
sys.path.insert(0, yolov5_path)
os.chdir(yolov5_path)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator

# --- ⚙️ 核心参数调节区 ---
model_path = r"/home/nvidia/yolov5/best.pt"
FACE_DB_DIR = r"/home/nvidia/yolov5/ID"

# 🛠️ 串口配置区（使用你的 USB 转串口设备）
SERIAL_PORT = '/dev/ttyUSB0'  
BAUDRATE = 9600  # 默认波特率 9600

# 调节此值以控制 YOLO 检测的灵敏度
YOLO_CONF_THRESHOLD = 0.1
# 调节此值以控制人脸比对严格度
STRICT_THRESHOLD = 0.35
# 调节开门后的持续时间（单位：秒）
OPEN_DURATION = 5
# 调节文字边框颜色
BLUE_COLOR = (255, 0, 0)

name_map = {0: "Amanda", 1: "Ines", 2: "Eric", 3: "Ethan"}

# --- 🔌 串口初始化与防护 ---
ser = None
try:
    # 开启串口，使用默认参数
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1
    )
    # 【核心优化】启动时，立刻强行将 TXD 压死在高电平，确保程序一启动，继电器绝不误触发，处于安全锁定状态[cite: 1]
    ser.break_condition = False 
    print(f"✅ 串口 {SERIAL_PORT} 初始化成功，已强行锁定引脚，防止误触发！[cite: 1, 2]")
except Exception as e:
    print(f"⚠️ 串口初始化失败 (可能是权限不足或设备未插入): {e}")
    print("💡 提示：如果报权限错误，请在终端运行: sudo chmod 777 " + SERIAL_PORT)

# --- 初始化 YOLO ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(model_path, map_location=device)
model.eval()

# --- 💡 自动初始化已知人脸库 ---
print("正在加载已知人脸库...")
known_face_encodings = []
if os.path.exists(FACE_DB_DIR):
    for root, dirs, files in os.walk(FACE_DB_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_p = os.path.join(root, file)
                try:
                    loaded_img = face_recognition.load_image_file(img_p)
                    encodings = face_recognition.face_encodings(loaded_img)
                    if encodings:
                        known_face_encodings.append(encodings[0])
                except Exception as e:
                    print(f"载入人脸图片失败: {img_p}, 错误: {e}")
print(f"成功录入 {len(known_face_encodings)} 张已知人脸特征！")

door_opened = False
last_open_time = 0


# --- 🚪 触发开门动作（利用强地电平 0V 触发继电器） ---
def trigger_door_lock():
    global last_trigger_time

    current_time = time.time()
    # 检查冷却时间：如果还没过 30 秒，则拒绝触发
    if current_time - last_trigger_time < COOLDOWN_TIME:
        print(f"⏳ 冷却中，请等待 {int(COOLDOWN_TIME - (current_time - last_trigger_time))} 秒后再次尝试。")
        return

    # 执行点动操作
    if ser and ser.is_open:
        try:
            print(f"🔓 [点动开门] 触发继电器 0.5 秒...")
            ser.break_condition = True  # 拉低电平吸合
            time.sleep(JOG_TIME)        # 点动保持 0.5 秒
            ser.break_condition = False # 恢复高电平断开
            
            last_trigger_time = time.time() # 更新触发时间戳
            print("🔒 [锁定] 继电器已释放，进入 30 秒冷却期。")
        except Exception as e:
            print(f"❌ 继电器驱动触发失败: {e}")


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    time_remains = 0
    if door_opened:
        elapsed = time.time() - last_open_time
        if elapsed > OPEN_DURATION:
            door_opened = False
            # 门禁时间到，恢复高电平，断开继电器（自动重新锁定）[cite: 1, 2]
            if ser and ser.is_open:
                ser.break_condition = False
            print("🔒 门禁时间到，引脚已恢复高电平，自动恢复锁定状态[cite: 1, 2]")
        else:
            time_remains = OPEN_DURATION - elapsed

    img = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device).permute(2, 0, 1).float() / 255.0
    if len(img.shape) == 3: img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False, visualize=False)[0]

    pred = non_max_suppression(pred, conf_thres=YOLO_CONF_THRESHOLD, iou_thres=0.45)

    annotator = Annotator(frame, line_width=2)
    label = "RUNNING"

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                raw_name = name_map.get(c, str(model.names[c]))

                if door_opened:
                    label = f'OPENED | {raw_name} ({conf * 100:.0f}%) | {time_remains:.1f}s'
                else:
                    x1, y1, x2, y2 = map(int, xyxy)
                    y1, y2, x1, x2 = max(0, y1), min(frame.shape[0], y2), max(0, x1), min(frame.shape[1], x2)
                    face_crop = frame[y1:y2, x1:x2]

                    try:
                        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        current_encodings = face_recognition.face_encodings(rgb_crop)

                        if current_encodings and known_face_encodings:
                            current_enc = current_encodings[0]
                            distances = face_recognition.face_distance(known_face_encodings, current_enc)
                            min_dist = float(min(distances))

                            if min_dist < STRICT_THRESHOLD:
                                label = f'MATCH | {raw_name} ({conf * 100:.0f}%) | Dist:{min_dist:.3f}'
                                trigger_door_lock()  # 👈 调用开门（触发继电器拉低）
                            else:
                                label = f'Insufficient similarity | {raw_name} ({conf * 100:.0f}%) | Dist:{min_dist:.3f}'
                        else:
                            label = f'UNKNOWN ({conf * 100:.0f}%)'
                    except Exception as e:
                        label = f'RUNNING ({conf * 100:.0f}%)'

                annotator.box_label(xyxy, label, color=BLUE_COLOR)

    # 🖥️ 显示窗口
    cv2.imshow("Face Recognition", annotator.result())
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]: break

    print(f"\r当前门禁状态: {label}      ", end="", flush=True)

# 退出时释放资源，确保关闭前恢复高电平，使继电器处于断开的安全状态[cite: 1, 2]
if ser and ser.is_open:
    try:
        ser.break_condition = False
    except:
        pass
    ser.close()
    print("\n🔌 串口引脚已安全复位并关闭。[cite: 1, 2]")

cap.release()
cv2.destroyAllWindows()