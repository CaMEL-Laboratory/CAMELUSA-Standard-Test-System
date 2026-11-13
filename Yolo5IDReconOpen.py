import os
import sys
import cv2
import time
import torch
from deepface import DeepFace

# --- 运行环境设置 ---
yolov5_path = r"D:\YOLO_Project\FaceProGateAccess\yolov5-6.1"
sys.path.insert(0, yolov5_path)
os.chdir(yolov5_path)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator

# --- ⚙️ 核心参数调节区 ---
model_path = r"D:\YOLO_Project\FaceProGateAccess\best.pt"
FACE_DB_DIR = r"D:\YOLO_Project\FaceProGateAccess\datasets\custom_data\test\images\ID"

# 调节此值以控制 YOLO 检测的灵敏度
YOLO_CONF_THRESHOLD = 0.3 
# 调节此值以控制人脸比对严格度，距离越小，相似度越高。
STRICT_THRESHOLD = 0.30    
# 调节开门后的持续时间
OPEN_DURATION = 5          
# 调节文字边框颜色
BLUE_COLOR = (255, 0, 0)   

name_map = {0: "Amanda", 1: "Ines", 2: "Eric", 3: "Ethan"}

# --- 初始化 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(model_path, map_location=device)
model.eval()

door_opened = False
last_open_time = 0

def trigger_door_lock():
    global door_opened, last_open_time
    door_opened = True
    last_open_time = time.time()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    time_remains = 0
    if door_opened:
        elapsed = time.time() - last_open_time
        if elapsed > OPEN_DURATION: door_opened = False
        else: time_remains = OPEN_DURATION - elapsed

    img = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device).permute(2, 0, 1).float() / 255.0
    if len(img.shape) == 3: img = img.unsqueeze(0)
    
    with torch.no_grad():
        pred = model(img, augment=False, visualize=False)[0]

    pred = non_max_suppression(pred, conf_thres=YOLO_CONF_THRESHOLD, iou_thres=0.45)

    annotator = Annotator(frame, line_width=2)
    
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                raw_name = name_map.get(c, str(model.names[c]))
                
                # 状态逻辑：开门后显示 OPENED
                if door_opened:
                    label = f'OPENED | {raw_name} ({conf*100:.0f}%) | {time_remains:.1f}s'
                else:
                    x1, y1, x2, y2 = map(int, xyxy)
                    face_crop = frame[y1:y2, x1:x2]
                    try:
                        dfs = DeepFace.find(img_path=face_crop, db_path=FACE_DB_DIR, 
                                            model_name="Facenet512", detector_backend="skip", 
                                            enforce_detection=False, silent=True)
                        
                        if not dfs[0].empty:
                            dist = dfs[0].iloc[0]['distance']
                            # 识别逻辑：包含姓名、置信度、距离
                            if dist < STRICT_THRESHOLD:
                                label = f'MATCH | {raw_name} ({conf*100:.0f}%) | Dist:{dist:.3f}'
                                trigger_door_lock()
                            else:
                                label = f'Insufficient similarity | {raw_name} ({conf*100:.0f}%) | Dist:{dist:.3f}'
                        else:
                            # UNKNOWN 不显示映射出的名字
                            label = f'UNKNOWN ({conf*100:.0f}%)'
                    except: 
                        label = f'RUNNING ({conf*100:.0f}%)'
                
                annotator.box_label(xyxy, label, color=BLUE_COLOR)

    cv2.imshow("Face Recognition", annotator.result())
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]: break

cap.release()
cv2.destroyAllWindows()