import os
import sys
import cv2
import time
import shutil
import numpy as np
import torch
import serial
import face_recognition
import dlib
import logging
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from scipy.spatial import distance as dist
from ctypes import *

# ==============================================================================
# ⚙️ 全局配置参数（集中管理）
# ==============================================================================

# 1. 📁 路径配置
YOLOV5_PATH = r"/home/nvidia/yolov5"                                    # YOLOv5 项目根目录
MODEL_PATH = r"/home/nvidia/yolov5/best.pt"                            # YOLO 模型权重路径
FACE_DB_DIR = r"/home/nvidia/yolov5/ID"                                # 人脸特征库文件夹路径
SHAPE_PREDICTOR_PATH = "/home/nvidia/yolov5/shape_predictor_68_face_landmarks.dat" # Dlib 68点标定模型路径
MVSDK_PYTHON_PATH = '/opt/GeneralMVTech/MVviewer/share/Python'          # 华睿 MVSDK Python 接口路径
MVSDK_LIB_PATH = '/opt/GeneralMVTech/MVviewer/lib'                      # 华睿 MVSDK 动态库路径

# 2. ⏱️ 门禁逻辑与时间控制（单位：秒）
OPEN_DURATION = 5.0        # 屏幕开门成功提示保持时间
COOLDOWN_TIME = 30.0       # 识别成功后的开门冷却期
DETECT_COOLDOWN = 3.0      # 检测到人脸时的抓拍与日志打印最小间隔（防刷屏）
UNKNOWN_COOLDOWN = 3.0     # 未知人员/匹配失败抓拍与日志打印的冷却间隔
JOG_TIME = 2               # 继电器脉冲开门触发时间
KEEP_RECORD_DAYS = 365       # 抓拍与日志文件夹保存天数（超过自动清理）

# 3. 🔌 硬件通讯配置 (串口)
SERIAL_PORT = '/dev/ttyUSB0'  # 门禁继电器串口号
BAUDRATE = 9600               # 串口波特率

# 4. 📷 相机硬件与自适应光照配置
TARGET_FPS = 15.0            # 相机目标采集帧率
MAX_EXPOSURE_TIME_US = 30000.0  # 自动曝光上限（微秒/30ms）：防止暗光拖影
MIN_EXPOSURE_TIME_US = 100.0    # 自动曝光下限（微秒）：防止强光过曝
MAX_GAIN_RAW = 12.0             # 自动增益上限（dB）：防止夜间噪点过大

# 5. 🧠 算法推理与识别阈值配置
INFERENCE_SIZE = 320         # YOLO 输入图像尺寸 (320x320)
SKIP_RATIO = 4               # 帧检测间隔（每 4 帧做一次 YOLO 推理）
YOLO_CONF_THRESHOLD = 0.35   # YOLO 人脸检测置信度阈值
STRICT_THRESHOLD = 0.46      # 人脸欧氏距离匹配阈值

EYE_AR_THRESH = 0.24         # 眨眼判定 EAR 阈值（低于此值判定为闭眼）
EYE_AR_CONSEC_FRAMES = 1     # 眨眼判定连续帧数

# 6. 🎨 显示与标签映射
BLUE_COLOR = (255, 0, 0)     # BGR 蓝框颜色
GREEN_COLOR = (0, 255, 0)   # BGR 绿框颜色
RED_COLOR = (0, 0, 255)     # BGR 红框颜色 (用于 UNKNOWN)
YELLOW_COLOR = (0, 255, 255) # BGR 黄框颜色 (用于 冷却/提示)
NAME_MAP = {0: "Amanda", 1: "Ines", 2: "Eric", 3: "Ethan"} # 类别 ID 到姓名映射

# ==============================================================================

# 动态添加环境路径
sys.path.insert(0, YOLOV5_PATH)
if MVSDK_PYTHON_PATH not in sys.path:
    sys.path.insert(0, MVSDK_PYTHON_PATH)
os.environ["LD_LIBRARY_PATH"] = MVSDK_LIB_PATH + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# 自动兼容 SSH X11 转发与本地 Display
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    os.environ["DISPLAY"] = ":0"

# ==================== 📁 7天日期文件夹与动态日志配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDS_ROOT = os.path.join(BASE_DIR, "records")
os.makedirs(RECORDS_ROOT, exist_ok=True)

def cleanup_old_records(root_dir, keep_days):
    """自动清理超过设定期限的旧日期文件夹"""
    now = datetime.now()
    cutoff_date = now - timedelta(days=keep_days)
    
    if not os.path.exists(root_dir):
        return

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            try:
                folder_date = datetime.strptime(folder_name, "%Y-%m-%d")
                if folder_date < cutoff_date:
                    shutil.rmtree(folder_path)
                    print(f"🗑️ 已清理超过 {keep_days} 天的旧记录文件夹: {folder_name}")
            except ValueError:
                pass

def get_today_record_dir():
    """获取当天的文件夹路径，如果不存在则创建，并执行旧文件夹清理"""
    cleanup_old_records(RECORDS_ROOT, keep_days=KEEP_RECORD_DAYS)
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_dir = os.path.join(RECORDS_ROOT, today_str)
    os.makedirs(today_dir, exist_ok=True)
    return today_dir

current_record_dir = get_today_record_dir()

# 自定义带自动 Flush 的日志 Handler，确保即时写入磁盘文件
class FlushTimedRotatingFileHandler(TimedRotatingFileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

log_file_path = os.path.join(current_record_dir, "access_control.log")
file_handler = FlushTimedRotatingFileHandler(
    log_file_path,
    when="h",
    interval=2,
    backupCount=KEEP_RECORD_DAYS * 12,
    encoding="utf-8"
)

formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logger = logging.getLogger("GateAccess")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"🚀 门禁控制系统启动，当前记录保存目录: {current_record_dir}")

# ==================== 🛠️ 加载 MVSDK 动态库 ====================
original_cwd = os.getcwd()
try:
    os.chdir(MVSDK_PYTHON_PATH)
    from MVSDK import *
    from ImageConvert import *
    logger.info("✅ 成功加载华睿相机 MVSDK 库！")
except Exception as e:
    logger.error(f"❌ 加载 MVSDK 失败: {e}")
finally:
    os.chdir(original_cwd)

# ==================== 👁️ Dlib 活体检测初始化 ====================
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

(lStart, lEnd) = (36, 42)
(rStart, rEnd) = (42, 48)

EAR_COUNTER = 0
LIVENESS_PASSED = False
last_trigger_time = 0           # 全局开门时间戳初始化
last_detect_snapshot_time = 0   # 记录上次检测到人脸抓拍的时间戳
last_unknown_snapshot_time = 0  # 记录上次抓拍 UNKNOWN 的时间戳

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ==================== 📷 华睿 SDK 工业相机封装类 ====================
class DahuaCamera:
    def __init__(self):
        self.camera = None
        self.streamSource = None
        self.system = None

    def enable_auto_exposure_and_gain(self):
        """开启华睿工业相机的连续自动曝光与自动增益"""
        if not self.camera:
            return

        expAutoEnumNode = pointer(GENICAM_EnumNode())
        expAutoInfo = GENICAM_EnumNodeInfo()
        expAutoInfo.pCamera = pointer(self.camera)
        expAutoInfo.attrName = b"ExposureAuto"
        if GENICAM_createEnumNode(byref(expAutoInfo), byref(expAutoEnumNode)) == 0:
            expAutoEnumNode.contents.setValueBySymbol(expAutoEnumNode, b"Continuous")
            expAutoEnumNode.contents.release(expAutoEnumNode)
            logger.info("💡 已开启硬件连续自动曝光 (ExposureAuto: Continuous)")

        expUpperNode = pointer(GENICAM_DoubleNode())
        expUpperInfo = GENICAM_DoubleNodeInfo()
        expUpperInfo.pCamera = pointer(self.camera)
        expUpperInfo.attrName = b"AutoExposureTimeUpperLimit"
        if GENICAM_createDoubleNode(byref(expUpperInfo), byref(expUpperNode)) == 0:
            expUpperNode.contents.setValue(expUpperNode, c_double(MAX_EXPOSURE_TIME_US))
            expUpperNode.contents.release(expUpperNode)

        gainAutoEnumNode = pointer(GENICAM_EnumNode())
        gainAutoInfo = GENICAM_EnumNodeInfo()
        gainAutoInfo.pCamera = pointer(self.camera)
        gainAutoInfo.attrName = b"GainAuto"
        if GENICAM_createEnumNode(byref(gainAutoInfo), byref(gainAutoEnumNode)) == 0:
            gainAutoEnumNode.contents.setValueBySymbol(gainAutoEnumNode, b"Continuous")
            gainAutoEnumNode.contents.release(gainAutoEnumNode)
            logger.info("💡 已开启硬件连续自动增益 (GainAuto: Continuous)")

        gainUpperNode = pointer(GENICAM_DoubleNode())
        gainUpperInfo = GENICAM_DoubleNodeInfo()
        gainUpperInfo.pCamera = pointer(self.camera)
        gainUpperInfo.attrName = b"AutoGainUpperLimit"
        if GENICAM_createDoubleNode(byref(gainUpperInfo), byref(gainUpperNode)) == 0:
            gainUpperNode.contents.setValue(gainUpperNode, c_double(MAX_GAIN_RAW))
            gainUpperNode.contents.release(gainUpperNode)

    def set_camera_fps(self, fps_val):
        if not self.camera:
            return

        fpsEnableNode = pointer(GENICAM_BoolNode())
        fpsEnableInfo = GENICAM_BoolNodeInfo()
        fpsEnableInfo.pCamera = pointer(self.camera)
        fpsEnableInfo.attrName = b"AcquisitionFrameRateEnable"
        if GENICAM_createBoolNode(byref(fpsEnableInfo), byref(fpsEnableNode)) == 0:
            fpsEnableNode.contents.setValue(fpsEnableNode, c_bool(True).value)
            fpsEnableNode.contents.release(fpsEnableNode)

        fpsNode = pointer(GENICAM_DoubleNode())
        fpsInfo = GENICAM_DoubleNodeInfo()
        fpsInfo.pCamera = pointer(self.camera)
        fpsInfo.attrName = b"AcquisitionFrameRate"
        if GENICAM_createDoubleNode(byref(fpsInfo), byref(fpsNode)) == 0:
            fpsNode.contents.setValue(fpsNode, c_double(float(fps_val)).value)
            fpsNode.contents.release(fpsNode)
            logger.info(f"🎯 相机硬件采集帧率已限制为: {fps_val} FPS")

    def open(self):
        self.system = pointer(GENICAM_System())
        if GENICAM_getSystemInstance(byref(self.system)) != 0:
            logger.error("❌ 获取相机系统实例失败！")
            return False

        cameraList = pointer(GENICAM_Camera())
        cameraCnt = c_uint()
        nRet = self.system.contents.discovery(self.system, byref(cameraList), byref(cameraCnt), c_int(GENICAM_EProtocolType.typeAll))
        
        if nRet != 0 or cameraCnt.value < 1:
            logger.warning(f"⚠️ 未发现工业相机 (Status={nRet}, Count={cameraCnt.value})")
            return False

        self.camera = cameraList[0]

        try:
            self.camera.disConnect(byref(self.camera))
            time.sleep(0.5)
        except Exception:
            pass

        connect_success = False
        for attempt in range(3):
            nRet = self.camera.connect(self.camera, c_int(GENICAM_ECameraAccessPermission.accessPermissionControl))
            if nRet == 0:
                connect_success = True
                break
            time.sleep(1.5)

        if not connect_success:
            logger.error("❌ 工业相机连接失败！")
            return False

        try:
            self.enable_auto_exposure_and_gain()
            self.set_camera_fps(TARGET_FPS)
        except Exception as e:
            logger.warning(f"⚠ 设置相机硬件自适应光照参数提示: {e}")

        streamSourceInfo = GENICAM_StreamSourceInfo()
        streamSourceInfo.channelId = 0
        streamSourceInfo.pCamera = pointer(self.camera)
        self.streamSource = pointer(GENICAM_StreamSource())
        if GENICAM_createStreamSource(pointer(streamSourceInfo), byref(self.streamSource)) != 0:
            return False

        trigModeEnumNode = pointer(GENICAM_EnumNode())
        trigModeEnumNodeInfo = GENICAM_EnumNodeInfo()
        trigModeEnumNodeInfo.pCamera = pointer(self.camera)
        trigModeEnumNodeInfo.attrName = b"TriggerMode"
        GENICAM_createEnumNode(byref(trigModeEnumNodeInfo), byref(trigModeEnumNode))
        trigModeEnumNode.contents.setValueBySymbol(trigModeEnumNode, b"Off")
        trigModeEnumNode.contents.release(trigModeEnumNode)

        if self.streamSource.contents.startGrabbing(self.streamSource, c_ulonglong(0), c_int(GENICAM_EGrabStrategy.grabStrartegyLatestImage)) != 0:
            return False

        logger.info("🚀 工业相机已成功启动拉流！")
        return True

    def get_frame(self):
        if not self.streamSource:
            return None

        frame = pointer(GENICAM_Frame())
        nRet = self.streamSource.contents.getFrame(self.streamSource, byref(frame), c_uint(1000))
        if nRet != 0 or frame.contents.valid(frame) != 0:
            return None

        imageParams = IMGCNV_SOpenParam()
        imageParams.dataSize    = frame.contents.getImageSize(frame)
        imageParams.height      = frame.contents.getImageHeight(frame)
        imageParams.width       = frame.contents.getImageWidth(frame)
        imageParams.paddingX    = frame.contents.getImagePaddingX(frame)
        imageParams.paddingY    = frame.contents.getImagePaddingY(frame)
        imageParams.pixelForamt = frame.contents.getImagePixelFormat(frame)

        imageBuff = frame.contents.getImage(frame)
        userBuff = c_buffer(b'\0', imageParams.dataSize)
        memmove(userBuff, c_char_p(imageBuff), imageParams.dataSize)
        frame.contents.release(frame)

        if imageParams.pixelForamt == EPixelType.gvspPixelMono8:
            grayByteArray = bytearray(userBuff)
            cvImage = np.array(grayByteArray).reshape(imageParams.height, imageParams.width)
            cvImage = cv2.cvtColor(cvImage, cv2.COLOR_GRAY2BGR)
        else:
            rgbSize = c_int()
            rgbBuff = c_buffer(b'\0', imageParams.height * imageParams.width * 3)
            IMGCNV_ConvertToBGR24(cast(userBuff, c_void_p), byref(imageParams), cast(rgbBuff, c_void_p), byref(rgbSize))
            colorByteArray = bytearray(rgbBuff)
            cvImage = np.array(colorByteArray).reshape(imageParams.height, imageParams.width, 3)

        return cvImage

    def close(self):
        if self.streamSource:
            try:
                self.streamSource.contents.stopGrabbing(self.streamSource)
                self.streamSource.contents.release(self.streamSource)
            except Exception:
                pass
            self.streamSource = None
        if self.camera:
            try:
                self.camera.disConnect(byref(self.camera))
            except Exception:
                pass
            self.camera = None

# ==================== 🔌 串口初始化 ====================
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
    ser.break_condition = False  
    logger.info(f"✅ 串口 {SERIAL_PORT} 初始化成功，引脚已锁定！")
except Exception as e:
    logger.warning(f"⚠️ 串口未连接，程序将在无继电器模式下运行: {e}")

# ==================== 🧠 YOLO 与人脸特征库初始化 ====================
os.chdir(YOLOV5_PATH)
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(MODEL_PATH, map_location=device)
model.eval()

logger.info("正在加载已知人脸库...")
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
                    logger.error(f"载入人脸图片失败: {img_p}, 错误: {e}")
logger.info(f"成功录入 {len(known_face_encodings)} 张已知人脸特征！")

# ==================== 📸 抓拍与事件处理 ====================
def save_snapshot_image(frame, stage_tag, raw_name):
    """保存通用抓拍图像"""
    try:
        today_dir = get_today_record_dir()
        time_str = datetime.now().strftime("%H-%M-%S_%f")[:-3]
        img_name = f"{time_str}_{stage_tag}_{raw_name}.jpg"
        save_path = os.path.join(today_dir, img_name)
        cv2.imwrite(save_path, frame)
        logger.info(f"📸 [{stage_tag}] 图像抓拍已保存: {save_path}")
    except Exception as e:
        logger.error(f"❌ 保存抓拍图片失败: {e}")

def handle_face_detected(frame, raw_name, conf):
    """阶段 1：检测到人脸时在控制台/日志输出并抓图"""
    global last_detect_snapshot_time
    now = time.time()
    if now - last_detect_snapshot_time > DETECT_COOLDOWN:
        last_detect_snapshot_time = now
        logger.info(f"🔍 [检测到人脸] 候选人: {raw_name} | 置信度: {conf * 100:.1f}%")
        save_snapshot_image(frame, "DETECTED", raw_name)

def trigger_door_lock(user_name, conf, min_dist, current_frame):
    """阶段 3：人脸比对与活体均通过，触发继电器点动开门"""
    global last_trigger_time, LIVENESS_PASSED

    last_trigger_time = time.time()
    LIVENESS_PASSED = False
    
    save_snapshot_image(current_frame, "MATCH", user_name)
    logger.info(f"🔓 [MATCH 成功] 身份判定为: {user_name} ({conf * 100:.0f}%) | Dist:{min_dist:.3f}")

    if ser and ser.is_open:
        try:
            logger.info(f"🔓 [点动开门] 触发继电器 {JOG_TIME} 秒...")
            ser.break_condition = True  # 拉低电平吸合继电器
            time.sleep(JOG_TIME)        # 点动保持 2 秒
            ser.break_condition = False # 恢复高电平断开
            logger.info(f"🔒 [锁定] 继电器已复位，进入 {COOLDOWN_TIME:.0f} 秒冷却锁定。")
        except Exception as e:
            logger.error(f"❌ 继电器驱动触发失败: {e}")
    else:
        logger.warning("⚠️ 串口未打开，无法驱动继电器！")

def handle_unknown_person(frame, label_text):
    """未授权通行处理"""
    global last_unknown_snapshot_time
    now = time.time()
    if now - last_unknown_snapshot_time > UNKNOWN_COOLDOWN:
        last_unknown_snapshot_time = now
        logger.warning(f"⚠ [未授权通行/未知人员] {label_text}")
        save_snapshot_image(frame, "UNKNOWN", "Unknown")

cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)

# ==================== 🚀 主业务逻辑循环 ====================
cam = DahuaCamera()
last_printed_label = None

frame_counter = 0
last_boxes = []

while True:  
    if not cam.open():
        logger.warning("⚠️ 相机未连接，5 秒后重试...")
        time.sleep(5)
        continue

    while True:
        frame = cam.get_frame()
        if frame is None:
            cam.close()
            time.sleep(2)
            break  

        frame_counter += 1

        # 🔍 YOLO 目标检测 (每 SKIP_RATIO 帧做一次)
        if frame_counter % SKIP_RATIO == 0 or len(last_boxes) == 0:
            img_resized = cv2.resize(frame, (INFERENCE_SIZE, INFERENCE_SIZE))
            img = torch.from_numpy(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)).to(device).permute(2, 0, 1).float() / 255.0
            if len(img.shape) == 3: 
                img = img.unsqueeze(0)

            with torch.no_grad():
                pred = model(img, augment=False, visualize=False)[0]

            pred = non_max_suppression(pred, conf_thres=YOLO_CONF_THRESHOLD, iou_thres=0.45)

            last_boxes = []
            for det in pred:
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        raw_name = NAME_MAP.get(c, str(model.names[c]))
                        last_boxes.append((xyxy, float(conf), raw_name))

        if len(last_boxes) == 0:
            LIVENESS_PASSED = False

        annotator = Annotator(frame, line_width=2)
        current_label = None  
        box_color = BLUE_COLOR

        now_time = time.time()
        time_since_last_trigger = now_time - last_trigger_time

        for xyxy, conf, raw_name in last_boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            y1, y2, x1, x2 = max(0, y1), min(frame.shape[0], y2), max(0, x1), min(frame.shape[1], x2)
            
            h_f, w_f = frame.shape[:2]
            pad_x, pad_y = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
            x1_p, y1_p = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2_p, y2_p = min(w_f, x2 + pad_x), min(h_f, y2 + pad_y)

            face_crop = frame[y1_p:y2_p, x1_p:x2_p]
            if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                continue

            # ------------------------------------------------------------------
            # ⏱️ 冷却时间判断
            # ------------------------------------------------------------------
            if time_since_last_trigger < COOLDOWN_TIME:
                remains = int(COOLDOWN_TIME - time_since_last_trigger)
                current_label = f'COOLDOWN ({remains}s) | {raw_name}'
                box_color = YELLOW_COLOR
                annotator.box_label(xyxy, current_label, color=box_color)
                
                # 同步日志输出与即时刷新
                if current_label != last_printed_label:
                    logger.info(f"当前门禁状态: {current_label}")
                    last_printed_label = current_label
                continue

            # ------------------------------------------------------------------
            # 📌 1. 抓拍与检测记录
            # ------------------------------------------------------------------
            handle_face_detected(frame, raw_name, conf)

            # ------------------------------------------------------------------
            # 👁 2. 眨眼活体检测计算
            # ------------------------------------------------------------------
            if not LIVENESS_PASSED:
                crop_small = cv2.resize(face_crop, (160, 160))
                gray_crop = cv2.cvtColor(crop_small, cv2.COLOR_BGR2GRAY)
                rects = dlib_detector(gray_crop, 0)
                
                EAR_COUNTER = 0
                for rect in rects:
                    shape = dlib_predictor(gray_crop, rect)
                    shape_coords = np.zeros((68, 2), dtype="int")
                    for i in range(0, 68):
                        shape_coords[i] = (shape.part(i).x, shape.part(i).y)
                        
                    leftEAR = calculate_ear(shape_coords[lStart:lEnd])
                    rightEAR = calculate_ear(shape_coords[rStart:rEnd])
                    ear = (leftEAR + rightEAR) / 2.0
                    
                    if ear < EYE_AR_THRESH:
                        EAR_COUNTER += 1

                if EAR_COUNTER >= EYE_AR_CONSEC_FRAMES:
                    LIVENESS_PASSED = True
                    logger.info("⚡ [活体验证] 眨眼检测通过，允许开门！")

            # ------------------------------------------------------------------
            # 🔓 3. 实时特征比对与状态生成
            # ------------------------------------------------------------------
            min_dist = 1.0
            is_match = False
            try:
                rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                current_encodings = face_recognition.face_encodings(rgb_crop)

                if current_encodings and known_face_encodings:
                    current_enc = current_encodings[0]
                    distances = face_recognition.face_distance(known_face_encodings, current_enc)
                    min_dist = float(min(distances))

                    if min_dist < STRICT_THRESHOLD:
                        is_match = True
                        current_label = f'MATCH | {raw_name} ({conf * 100:.0f}%) | Dist:{min_dist:.3f}'
                        box_color = BLUE_COLOR
                    else:
                        current_label = f'UNKNOWN | Insufficient similarity ({conf * 100:.0f}%) | Dist:{min_dist:.3f}'
                        box_color = RED_COLOR
                else:
                    current_label = f'UNKNOWN ({conf * 100:.0f}%)'
                    box_color = RED_COLOR
            except Exception as e:
                current_label = f'RUNNING ({conf * 100:.0f}%)'

            # ------------------------------------------------------------------
            # 🚪 4. 驱动开门逻辑 / 显示眨眼提醒
            # ------------------------------------------------------------------
            if is_match:
                if LIVENESS_PASSED:
                    trigger_door_lock(raw_name, conf, min_dist, frame)
                else:
                    current_label = f'PLEASE BLINK | {current_label}'
            elif "UNKNOWN" in current_label:
                handle_unknown_person(frame, current_label)

            # ------------------------------------------------------------------
            # 🎨 5. 渲染画框与 Label 标签
            # ------------------------------------------------------------------
            if current_label:
                annotator.box_label(xyxy, current_label, color=box_color)

        annotated_frame = annotator.result()
        cv2.imshow("Face Recognition", annotated_frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            cam.close()
            sys.exit(0)

        # 实时打印并刷盘变化的状态信息
        if current_label and current_label != last_printed_label:
            logger.info(f"当前门禁状态: {current_label}")
            last_printed_label = current_label

cam.close()
cv2.destroyAllWindows()