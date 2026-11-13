import os
import sys
import cv2
import time
import torch
import serial
import threading  # 👈 引入多线程库，防止 0.5 秒点动卡顿主画面
import face_recognition

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

# 🛠️ 串口配置（USB转TTL模块）
SERIAL_PORT = '/dev/ttyUSB0'  
BAUDRATE = 9600  

# 调节此值以控制 YOLO 检测的灵敏度
YOLO_CONF_THRESHOLD = 0.3
# 🚨 调节此值以控制人脸比对严格度（已设为 0.35）
STRICT_THRESHOLD = 0.35
# ⏱️ 门禁点动与冷却控制
COOLDOWN_TIME = 30     # 冷却时间：30 秒内禁止重复开门
JOG_TIME = 0.5         # 点动时间：继电器吸合保持 0.5 秒

BLUE_COLOR = (255, 0, 0)
name_map = {0: "Amanda", 1: "Ines", 2: "Eric", 3: "Ethan"}

# --- 🔌 串口初始化与防护 ---
ser = None
try:
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1
    )
    # 启动时强制拉高电平，确保绝对安全锁定状态
    ser.break_condition = False 
    print(f"✅ 串口 {SERIAL_PORT} 初始化成功，已强行锁定引脚，防止误触发！")
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

# 冷却控制与状态全局变量
last_trigger_time = 0 
lock_status = "LOCKED"
# 用于控制台日志去重的变量
last_logged_status = ""
last_logged_refusal = ""
last_refuse_time = 0


# --- 🚪 独立线程：执行 0.5秒 点动开门 ---
def run_door_jog():
    global lock_status
    if ser and ser.is_open:
        try:
            print(f"🔓 [点动开门] 触发继电器，TXD强行拉低至 0V...")
            ser.break_condition = True   # 拉低电平，继电器吸合
            time.sleep(JOG_TIME)         # 保持点动 0.5 秒
            ser.break_condition = False  # 恢复高电平，继电器释放
            print("🔒 [锁定] 继电器已释放，进入 30 秒冷却期。")
        except Exception as e:
            print(f"❌ 继电器硬件触发失败: {e}")
    lock_status = "LOCKED"


# --- 🔑 冷却验证与触发分流器 ---
def trigger_door_lock():
    global last_trigger_time, lock_status
    current_time = time.time()
    
    # 验证是否超出 30 秒冷却时间
    if current_time - last_trigger_time > COOLDOWN_TIME:
        last_trigger_time = current_time
        lock_status = "OPENING"
        # 启动后台线程点动，防止卡住 YOLO 检测主循环
        threading.Thread(target=run_door_jog).start()
    else:
        remaining = int(COOLDOWN_TIME - (current_time - last_trigger_time))
        print(f"⏳ [安全拦截] 人脸比对通过，但处于冷却期，还需等待 {remaining} 秒。")


# --- 打开摄像头 ---
cap = cv2.VideoCapture(0)

print("\n🚀 门禁监控系统已就绪，开始滚动打印日志：\n" + "-"*50)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    img = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device).permute(2, 0, 1).float() / 255.0
    if len(img.shape) == 3: img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False, visualize=False)[0]

    pred = non_max_suppression(pred, conf_thres=YOLO_CONF_THRESHOLD, iou_thres=0.45)

    annotator = Annotator(frame, line_width=2)
    label = f"STATUS: {lock_status}"
    
    # 如果状态发生改变，滚动打印当前的门禁状态
    if lock_status != last_logged_status:
        print(f"ℹ️ [系统状态变更] -> {lock_status}")
        last_logged_status = lock_status

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                raw_name = name_map.get(c, str(model.names[c]))

                # 裁剪人脸区域
                x1, y1, x2, y2 = map(int, xyxy)
                y1, y2, x1, x2 = max(0, y1), min(frame.shape[0], y2), max(0, x1), min(frame.shape[1], x2)
                face_crop = frame[y1:y2, x1:x2]

                min_dist = 999.0  # 默认不安全距离
                
                try:
                    rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    current_encodings = face_recognition.face_encodings(rgb_crop)

                    if current_encodings and known_face_encodings:
                        current_enc = current_encodings[0]
                        distances = face_recognition.face_distance(known_face_encodings, current_enc)
                        min_dist = float(distances.min())

                        # 🚨【关键严密判定】
                        if min_dist < STRICT_THRESHOLD:
                            label = f'MATCH | {raw_name} ({conf * 100:.0f}%) | Dist:{min_dist:.3f}'
                            print(f"🎯 [识别成功] 匹配到: {raw_name} | 置信度: {conf*100:.1f}% | 距离: {min_dist:.3f}")
                            trigger_door_lock()  # ✅ 严格符合条件，申请点动开门
                        else:
                            label = f'Insufficient similarity | {raw_name} ({conf * 100:.0f}%) | Dist:{min_dist:.3f}'
                            
                            # 限制“相似度不足”的滚动打印频率，防止同一张脸在一秒内刷屏几十条
                            current_time = time.time()
                            refusal_key = f"{raw_name}_{min_dist:.2f}"
                            if (refusal_key != last_logged_refusal) or (current_time - last_refuse_time > 1.5):
                                print(f"❌ [拒绝通行] 识别为 {raw_name} 但相似度不足 | 距离: {min_dist:.3f} (设定阈值: {STRICT_THRESHOLD})")
                                last_logged_refusal = refusal_key
                                last_refuse_time = current_time
                    else:
                        label = f'UNKNOWN ({conf * 100:.0f}%)'
                except Exception as e:
                    label = f'RUNNING ({conf * 100:.0f}%)'
                    print(f"⚠️ 人脸比对时发生异常: {e}")

                annotator.box_label(xyxy, label, color=BLUE_COLOR)

    # 显示画面
    cv2.imshow("Face Recognition", annotator.result())
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]: break

# 退出时彻底释放资源，并确保关闭串口前引脚恢复高电平
if ser and ser.is_open:
    try:
        ser.break_condition = False
    except:
        pass
    ser.close()
    print("\n🔌 串口已安全关闭，日志记录结束。")

cap.release()
cv2.destroyAllWindows()