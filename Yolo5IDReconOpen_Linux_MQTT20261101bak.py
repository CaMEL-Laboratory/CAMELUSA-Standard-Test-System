import os
import sys
import cv2
import time
import shutil
import socket
import threading
import numpy as np
import torch
import serial
import face_recognition
import dlib
import logging
import json
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from scipy.spatial import distance as dist
from ctypes import *
import atexit

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
KEEP_RECORD_DAYS = 365     # 抓拍与日志文件夹保存天数（超过自动清理）

# 3. 🔌 硬件通讯配置 (串口)
SERIAL_PORT = '/dev/ttyUSB0'  # 门禁继电器串口号
BAUDRATE = 9600               # 串口波特率

# 4. 🌐 MQTT 及 Home Assistant 配置
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_COMMAND_TOPIC = "home/gate/command"
MQTT_STATUS_TOPIC = "home/gate/status"

# 5. 🗣️ Alexa 语音控制配置 (SSDP / WeMo)
WEMO_PORT = 49153             # WeMo 监听端口
DEVICE_NAME = "Gate Lock"     # Alexa 语音唤醒设备名
DEVICE_UUID = "Socket-1_0-220000000001"

# 6. 📷 相机硬件与自适应光照配置
TARGET_FPS = 4.0            # 相机目标采集帧率
MAX_EXPOSURE_TIME_US = 40000.0  # 自动曝光上限（微秒/30ms）：防止暗光拖影
MIN_EXPOSURE_TIME_US = 100.0    # 自动曝光下限（微秒）：防止强光过曝
MAX_GAIN_RAW = 12.0             # 自动增益上限（dB）：防止夜间噪点过大

# 7. 🧠 算法推理与识别阈值配置
INFERENCE_SIZE = 320         # YOLO 输入图像尺寸 (320x320)
SKIP_RATIO = 2               # 帧检测间隔（每 2 帧做一次 YOLO 推理）
YOLO_CONF_THRESHOLD = 0.35   # YOLO 人脸检测置信度阈值
STRICT_THRESHOLD = 0.46      # 人脸欧氏距离匹配阈值

EYE_AR_THRESH = 0.24         # 眨眼判定 EAR 阈值（低于此值判定为闭眼）
EYE_AR_CONSEC_FRAMES = 1     # 眨眼判定连续帧数

# 8. 🎨 显示与标签映射
BLUE_COLOR = (255, 0, 0)     # BGR 蓝框颜色
GREEN_COLOR = (0, 255, 0)   # BGR 绿框颜色
RED_COLOR = (0, 0, 255)     # BGR 红框颜色 (用于 UNKNOWN)
YELLOW_COLOR = (0, 255, 255) # BGR 黄框颜色 (用于 冷却/提示)
NAME_MAP = {0: "Amanda", 1: "Ines", 2: "Eric", 3: "Ethan"} # 类别 ID 到姓名映射

# 9. 🌊 运动变化检测配置（变化率设定为 20%）
MOTION_THRESHOLD_RATIO = 0.20    # 画面变化率阈值 (20%)
MOTION_COOLDOWN_TIME = 4.0       # 运动触发抓拍的最小间隔时间（秒）

# ==============================================================================

# 动态添加环境路径
sys.path.insert(0, YOLOV5_PATH)
if MVSDK_PYTHON_PATH not in sys.path:
    sys.path.insert(0, MVSDK_PYTHON_PATH)
os.environ["LD_LIBRARY_PATH"] = MVSDK_LIB_PATH + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# 自动兼容 SSH X11 转发与本地 Display
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    os.environ["DISPLAY"] = ":0"

# ==================== 📁 日期文件夹与动态日志配置 ====================
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

# 自定义带自动 Flush 的日志 Handler
class FlushTimedRotatingFileHandler(TimedRotatingFileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

log_file_path = os.path.join(current_record_dir, "access_control.log")
file_handler = FlushTimedRotatingFileHandler(
    log_file_path, when="h", interval=2, backupCount=KEEP_RECORD_DAYS * 12, encoding="utf-8"
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

# ==================== 🛑 系统关机/信号捕获钩子（确保 shutdown 时也能记录） ====================
import signal

def system_shutdown_handler(signum=None, frame=None):
    logger.info("🛑 门禁控制系统正在停止运行 / 关机...")
    time.sleep(0.5)  # 留出半秒缓冲确保日志完整落盘
    sys.exit(0)

atexit.register(system_shutdown_handler)
signal.signal(signal.SIGTERM, system_shutdown_handler)
signal.signal(signal.SIGINT, system_shutdown_handler)


# ==================== 🚪 门禁物理状态机与严格防抖控制逻辑 ====================
CURRENT_GATE_STATE = "Locked"  # 默认初始状态为锁定
LAST_COMMAND_TIME = 0          # 后端严格防抖时间戳
mqtt_client = mqtt.Client(client_id="JetsonTX2_GateController")
_mqtt_connected_flag = False

def toggle_gate_state(trigger_source="User Action"):
    """
    脉冲切换型（Toggle 模式）核心控制函数：
    加入严格的后端防抖锁，在 30 秒冷却期内直接拒绝任何冗余的硬件触发。
    """
    global CURRENT_GATE_STATE, LAST_COMMAND_TIME
    
    current_time = time.time()
    # 严格后端防抖：若距上次触发小于 COOLDOWN_TIME，直接拦截
    if current_time - LAST_COMMAND_TIME < COOLDOWN_TIME:
        remains = int(COOLDOWN_TIME - (current_time - LAST_COMMAND_TIME))
        logger.warning(f"⏳ [{trigger_source}] 系统处于冷却期（剩余 {remains} 秒），已严格拦截重复触发！")
        return CURRENT_GATE_STATE

    LAST_COMMAND_TIME = current_time

    # 状态取反
    if CURRENT_GATE_STATE == "Locked":
        target_state = "Unlocked"
        action_name = "开启 (Unlock)"
    else:
        target_state = "Locked"
        action_name = "关闭 (Lock)"

    logger.info(f"🔊 [{trigger_source}] 触发物理脉冲，准备切换门状态为: {action_name}...")

    # 驱动串口继电器点动（脉冲）
    if ser and ser.is_open:
        try:
            ser.break_condition = True  # 拉低电平触发
            time.sleep(JOG_TIME)        # 点动保持时间
            ser.break_condition = False # 恢复断开
            logger.info(f"🔌 [{trigger_source}] 继电器脉冲执行完毕。")
        except Exception as e:
            logger.error(f"❌ [{trigger_source}] 继电器驱动失败: {e}")
    else:
        logger.warning(f"⚠️ [{trigger_source}] 串口未打开，模拟状态翻转。")

    # 更新全局门状态并同步至 MQTT
    CURRENT_GATE_STATE = target_state
    try:
        mqtt_client.publish(MQTT_STATUS_TOPIC, CURRENT_GATE_STATE, qos=1, retain=True)
        logger.info(f"📡 [MQTT Sync] 状态已同步至 HA -> {MQTT_STATUS_TOPIC}: {CURRENT_GATE_STATE}")
    except Exception as e:
        logger.error(f"❌ [MQTT Error] 状态同步失败: {e}")

    return CURRENT_GATE_STATE

def on_mqtt_connect(client, userdata, flags, rc):
    global _mqtt_connected_flag
    if rc == 0:
        if not _mqtt_connected_flag:
            logger.info("✅ 成功连接到本地 MQTT Broker!")
            _mqtt_connected_flag = True
        client.subscribe(MQTT_COMMAND_TOPIC)
        # 启动时向 HA 汇报当前默认状态
        client.publish(MQTT_STATUS_TOPIC, CURRENT_GATE_STATE, qos=1, retain=True)
    else:
        logger.error(f"❌ MQTT 连接失败，返回码: {rc}")

def on_mqtt_message(client, userdata, msg):
    """接收来自 Home Assistant 的控制指令"""
    payload = msg.payload.decode('utf-8').strip().upper()
    logger.info(f"📩 [MQTT Command] 收到来自主题 [{msg.topic}] 的指令: {payload}")
    
    # 收到指令后由带防抖的 toggle_gate_state 处理
    if payload in ["OPEN", "UNLOCK", "LOCK", "TOGGLE"]:
        toggle_gate_state(trigger_source="Home Assistant Dashboard")

mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_message = on_mqtt_message

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
except Exception as e:
    logger.error(f"❌ 无法连接 MQTT Broker ({MQTT_BROKER}:{MQTT_PORT}): {e}")

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
last_trigger_time = 0           
last_detect_snapshot_time = 0   
last_unknown_snapshot_time = 0  

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
        if not self.camera:
            return
        expAutoEnumNode = pointer(GENICAM_EnumNode())
        expAutoInfo = GENICAM_EnumNodeInfo()
        expAutoInfo.pCamera = pointer(self.camera)
        expAutoInfo.attrName = b"ExposureAuto"
        if GENICAM_createEnumNode(byref(expAutoInfo), byref(expAutoEnumNode)) == 0:
            expAutoEnumNode.contents.setValueBySymbol(expAutoEnumNode, b"Continuous")
            expAutoEnumNode.contents.release(expAutoEnumNode)

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

    def open(self):
        self.system = pointer(GENICAM_System())
        if GENICAM_getSystemInstance(byref(self.system)) != 0:
            return False
        cameraList = pointer(GENICAM_Camera())
        cameraCnt = c_uint()
        nRet = self.system.contents.discovery(self.system, byref(cameraList), byref(cameraCnt), c_int(GENICAM_EProtocolType.typeAll))
        if nRet != 0 or cameraCnt.value < 1:
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
            return False

        try:
            self.enable_auto_exposure_and_gain()
            self.set_camera_fps(TARGET_FPS)
        except Exception:
            pass

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
        port=SERIAL_PORT, baudrate=BAUDRATE,
        bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE, timeout=1
    )
    ser.break_condition = False  
    logger.info(f"✅ 串口 {SERIAL_PORT} 初始化成功，引脚已锁定！")
except Exception as e:
    logger.warning(f"⚠️ 串口未连接，程序将在无继电器模式下运行: {e}")

# ==================== 🗣️ ALEXA 语音控制模块 (WEMO/SSDP) ====================
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

LOCAL_IP = get_local_ip()

SSDP_RESPONSE = (
    "HTTP/1.1 200 OK\r\n"
    "CACHE-CONTROL: max-age=86400\r\n"
    "DATE: Thu, 15 Oct 2026 19:27:00 GMT\r\n"
    "EXT:\r\n"
    "LOCATION: http://{ip}:{port}/setup.xml\r\n"
    "OPT: \"http://schemas.upnp.org/upnp/1/0/\"; ns=01\r\n"
    "01-1000: {uuid}\r\n"
    "SERVER: Unspecified, UPnP/1.0, Unspecified\r\n"
    "ST: urn:Belkin:device:controllee:1\r\n"
    "USN: uuid:Socket-1_0-{uuid}::urn:Belkin:device:controllee:1\r\n\r\n"
)

def ssdp_listener():
    MCAST_GRP = '239.255.255.250'
    MCAST_PORT = 1900
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', MCAST_PORT))
    
    # 🌟 改进 1：强制绑定到 wlan0 的 LOCAL_IP 加入组播，避开 docker 虚拟网卡干扰
    mreq = socket.inet_aton(MCAST_GRP) + socket.inet_aton(LOCAL_IP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    logger.info(f"🎙️ SSDP 监听服务已启动，精准绑定 wlan0 IP: {LOCAL_IP}")

    while True:
        try:
            data, addr = sock.recvfrom(1024)
            decoded_data = data.decode('utf-8', errors='ignore')
            
            if b'M-SEARCH' in data and (b'upnp:rootdevice' in data or b'ssdp:all' in data or b'belkin' in data.lower()):
                resp = SSDP_RESPONSE.format(ip=LOCAL_IP, port=WEMO_PORT, uuid=DEVICE_UUID).encode()
                # 🌟 确保精准向请求源回传响应
                sock.sendto(resp, addr)
                logger.info(f"📡 [SSDP] 收到来自 {addr[0]} 的发现请求，已成功响应 WeMo 身份！")
        except Exception:
            pass

def wemo_http_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(('0.0.0.0', WEMO_PORT))
    except Exception as e:
        logger.error(f"❌ WeMo 端口绑定失败 ({WEMO_PORT}): {e}")
        return
        
    server.listen(5)

    setup_xml = (
        '<?xml version="1.0"?>'
        '<root xmlns="urn:schemas-upnp-org:device-1-0">'
        '<device>'
        '<deviceType>urn:MakerModel:device:controllee:1</deviceType>'
        f'<friendlyName>{DEVICE_NAME}</friendlyName>'
        '<manufacturer>Belkin International Inc.</manufacturer>'
        '<modelName>Emulated Socket</modelName>'
        '<modelNumber>3.1415</modelNumber>'
        f'<UDN>uuid:{DEVICE_UUID}</UDN>'
        '</device>'
        '</root>'
    )

    while True:
        try:
            client, addr = server.accept()
            data = client.recv(2048).decode('utf-8', errors='ignore')

            if "GET /setup.xml" in data:
                resp = f"HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: {len(setup_xml)}\r\n\r\n{setup_xml}"
                client.sendall(resp.encode())
            elif "GetBinaryState" in data:
                soap_resp = (
                    "HTTP/1.1 200 OK\r\nContent-Type: text/xml; charset=\"utf-8\"\r\n\r\n"
                    "<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\">"
                    "<s:Body><u:GetBinaryStateResponse xmlns:u=\"urn:Belkin:service:basicevent:1\">"
                    "<BinaryState>0</BinaryState></u:GetBinaryStateResponse></s:Body></s:Envelope>"
                )
                client.sendall(soap_resp.encode())
            elif "SetBinaryState" in data or "SOAPACTION" in data:
                threading.Thread(target=toggle_gate_state, args=("Alexa 语音",), daemon=True).start()
                soap_resp = (
                    "HTTP/1.1 200 OK\r\nContent-Type: text/xml; charset=\"utf-8\"\r\n\r\n"
                    "<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\">"
                    "<s:Body><u:SetBinaryStateResponse xmlns:u=\"urn:Belkin:service:basicevent:1\">"
                    "<BinaryState>1</BinaryState></u:SetBinaryStateResponse></s:Body></s:Envelope>"
                )
                client.sendall(soap_resp.encode())
            else:
                client.sendall(b"HTTP/1.1 200 OK\r\n\r\n")
            client.close()
        except Exception:
            pass

def start_alexa_service():
    threading.Thread(target=ssdp_listener, daemon=True).start()
    threading.Thread(target=wemo_http_server, daemon=True).start()
    logger.info("🎙️ Alexa 语音控制后台线程初始化完成！")

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

# ==================== 🌟 改进 2：将 Alexa 服务移到模型与人脸库加载完成后再启动 ====================
start_alexa_service()

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
    """人脸比对与活体均通过，触发状态翻转"""
    global last_trigger_time, LIVENESS_PASSED

    last_trigger_time = time.time()
    LIVENESS_PASSED = False
    
    if current_frame is not None:
        save_snapshot_image(current_frame, "MATCH", user_name)
    logger.info(f"🔓 [MATCH 成功] 身份判定为: {user_name} ({conf * 100:.0f}%) | Dist:{min_dist:.3f}")

    # 调用状态翻转函数
    toggle_gate_state(trigger_source=f"人脸识别: {user_name}")

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

last_motion_snapshot_time = 0   # 运动抓拍防刷屏时间戳
previous_gray_frame = None      # 帧差法上一帧缓存

while True:  
    if not cam.open():
        logger.warning("⚠️ 相机未连接，5 秒后重试...")
        time.sleep(5)
        continue

    consecutive_none_frames = 0  # 连续未获取到帧的计数器

    while True:
        frame = cam.get_frame()
        if frame is None:
            consecutive_none_frames += 1
            # 只有连续 15 帧拿不到画面才重置相机，绝不因单帧偶发超时频繁重启
            if consecutive_none_frames > 15:
                logger.warning("⚠️ 连续多次未获取到相机帧，正在重新初始化相机...")
                cam.close()
                break  
            time.sleep(0.05)
            continue

        consecutive_none_frames = 0
        frame_counter += 1
        h_f, w_f = frame.shape[:2]
        total_pixels = h_f * w_f

        # ------------------------------------------------------------------
        # 🌊 帧差法检测（变化率 >= 20% 且间隔 >= 4秒则强制抓拍留底，防刷脸）
        # ------------------------------------------------------------------
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_triggered = False
        if previous_gray_frame is not None:
            non_zero_count = np.count_nonzero(cv2.absdiff(previous_gray_frame, gray_current) > 30)
            change_ratio = non_zero_count / float(total_pixels)
            
            if change_ratio >= MOTION_THRESHOLD_RATIO:
                now_t = time.time()
                if now_t - last_motion_snapshot_time > MOTION_COOLDOWN_TIME:
                    last_motion_snapshot_time = now_t
                    motion_triggered = True
                    logger.info(f"🌊 [运动抓拍] 监听到画面变化率为 {change_ratio * 100:.1f}% (达到设定阈值 {MOTION_THRESHOLD_RATIO * 100:.0f}%)，已强制抓拍防刷脸！")
                    save_snapshot_image(frame, "MOTION", "Unknown")

        previous_gray_frame = gray_current.copy()

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
            mqtt_client.loop_stop()
            sys.exit(0)

        if current_label and current_label != last_printed_label:
            logger.info(f"当前门禁状态: {current_label}")
            last_printed_label = current_label

cam.close()
cv2.destroyAllWindows()