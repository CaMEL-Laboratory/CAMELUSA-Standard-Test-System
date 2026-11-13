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
import yaml  # 🌟 统一配置解析核心库
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from scipy.spatial import distance as dist
from ctypes import *
import atexit
import signal
from http.server import HTTPServer, BaseHTTPRequestHandler

# ==============================================================================
# ⚙️ 专业动态配置管理模块 (从 hardware_channels.yaml 加载)
# ==============================================================================

CONFIG_FILE_PATH = os.getenv("CONFIG_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "hardware_channels.yaml"))

def load_hardware_config():
    """
    专业配置加载器：支持从 YAML 中实时读取最新配置。
    若文件读取失败，返回安全的容灾默认值。
    """
    default_cfg = {
        "paths_and_hardware": {
            "yolov5_path": "/home/nvidia/yolov5",
            "model_path": "/home/nvidia/yolov5/best.pt",
            "face_db_dir": "/home/nvidia/yolov5/ID",
            "shape_predictor_path": "/home/nvidia/yolov5/shape_predictor_68_face_landmarks.dat",
            "mvsdk_python_path": "/opt/GeneralMVTech/MVviewer/share/Python",
            "mvsdk_lib_path": "/opt/GeneralMVTech/MVviewer/lib",
            "camera_id": "/dev/video0",
            "relay_serial_port": "/dev/ttyUSB0",
            "baudrate": 9600,
            "local_ip": "10.0.0.58",
            "wemo_port": 49153,
            "device_name": "Gate Lock",
            "device_uuid": "Socket-1_0-220000000001"
        },
        "gate_logic": {
            "open_duration": 5.0,
            "cooldown_time": 30.0,
            "detect_cooldown": 3.0,
            "unknown_cooldown": 3.0,
            "jog_time": 2,
            "keep_record_days": 365
        },
        "camera_control": {
            "target_fps": 4.0,
            "max_exposure_time_us": 40000.0,
            "min_exposure_time_us": 100.0,
            "max_gain_raw": 12.0
        },
        "algorithm_thresholds": {
            "inference_size": 320,
            "skip_ratio": 2,
            "face_confidence_threshold": 0.35,
            "strict_threshold": 0.46,
            "eye_ar_thresh": 0.24,
            "eye_ar_consec_frames": 1,
            "motion_threshold_ratio": 0.20,
            "motion_cooldown_time": 4.0
        },
        "ui_mapping": {
            "blue_color": [255, 0, 0],
            "green_color": [0, 255, 0],
            "red_color": [0, 0, 255],
            "yellow_color": [0, 255, 255],
            "name_map": {0: "Amanda", 1: "Ines", 2: "Eric", 3: "Ethan"}
        },
        "mqtt_config": {
            "broker": "localhost",
            "port": 1883,
            "command_topic": "home/gate/command",
            "status_topic": "home/gate/status"
        }
    }

    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:
                    # 深度合并或按需覆盖
                    for section in default_cfg:
                        if section in data and isinstance(data[section], dict):
                            default_cfg[section].update(data[section])
        except Exception as e:
            print(f"⚠️ 解析配置文件 {CONFIG_FILE_PATH} 异常: {e}，将使用内置默认值。")
            
    return default_cfg

# 加载全局初始配置快照
CONFIG = load_hardware_config()

# 1. 📁 路径配置解耦
YOLOV5_PATH = CONFIG["paths_and_hardware"]["yolov5_path"]
MODEL_PATH = CONFIG["paths_and_hardware"]["model_path"]
FACE_DB_DIR = CONFIG["paths_and_hardware"]["face_db_dir"]
SHAPE_PREDICTOR_PATH = CONFIG["paths_and_hardware"]["shape_predictor_path"]
MVSDK_PYTHON_PATH = CONFIG["paths_and_hardware"]["mvsdk_python_path"]
MVSDK_LIB_PATH = CONFIG["paths_and_hardware"]["mvsdk_lib_path"]

# 2. ⏱️ 门禁逻辑与时间控制参数
OPEN_DURATION = CONFIG["gate_logic"]["open_duration"]
COOLDOWN_TIME = CONFIG["gate_logic"]["cooldown_time"]
DETECT_COOLDOWN = CONFIG["gate_logic"]["detect_cooldown"]
UNKNOWN_COOLDOWN = CONFIG["gate_logic"]["unknown_cooldown"]
JOG_TIME = CONFIG["gate_logic"]["jog_time"]
KEEP_RECORD_DAYS = CONFIG["gate_logic"]["keep_record_days"]

# 3. 🔌 硬件通讯配置
SERIAL_PORT = CONFIG["paths_and_hardware"]["relay_serial_port"]
BAUDRATE = CONFIG["paths_and_hardware"]["baudrate"]

# 4. 🌐 MQTT 配置
MQTT_BROKER = os.getenv("MQTT_BROKER", CONFIG["mqtt_config"]["broker"])
MQTT_PORT = CONFIG["mqtt_config"]["port"]
MQTT_COMMAND_TOPIC = CONFIG["mqtt_config"]["command_topic"]
MQTT_STATUS_TOPIC = CONFIG["mqtt_config"]["status_topic"]

# 5. 🗣️ Alexa 语音控制配置
LOCAL_IP = CONFIG["paths_and_hardware"]["local_ip"]
WEMO_PORT = CONFIG["paths_and_hardware"]["wemo_port"]
DEVICE_NAME = CONFIG["paths_and_hardware"]["device_name"]
DEVICE_UUID = CONFIG["paths_and_hardware"]["device_uuid"]

# 6. 📷 相机与算法基础配置
TARGET_FPS = CONFIG["camera_control"]["target_fps"]
MAX_EXPOSURE_TIME_US = CONFIG["camera_control"]["max_exposure_time_us"]
MIN_EXPOSURE_TIME_US = CONFIG["camera_control"]["min_exposure_time_us"]
MAX_GAIN_RAW = CONFIG["camera_control"]["max_gain_raw"]

INFERENCE_SIZE = CONFIG["algorithm_thresholds"]["inference_size"]
SKIP_RATIO = CONFIG["algorithm_thresholds"]["skip_ratio"]
YOLO_CONF_THRESHOLD = CONFIG["algorithm_thresholds"]["face_confidence_threshold"]
STRICT_THRESHOLD = CONFIG["algorithm_thresholds"]["strict_threshold"]

EYE_AR_THRESH = CONFIG["algorithm_thresholds"]["eye_ar_thresh"]
EYE_AR_CONSEC_FRAMES = CONFIG["algorithm_thresholds"]["eye_ar_consec_frames"]

# 8. 🎨 显示与标签映射（支持将 YAML 的 key 转为 int）
BLUE_COLOR = tuple(CONFIG["ui_mapping"]["blue_color"])
GREEN_COLOR = tuple(CONFIG["ui_mapping"]["green_color"])
RED_COLOR = tuple(CONFIG["ui_mapping"]["red_color"])
YELLOW_COLOR = tuple(CONFIG["ui_mapping"]["yellow_color"])
NAME_MAP = {int(k): v for k, v in CONFIG["ui_mapping"]["name_map"].items()}

MOTION_THRESHOLD_RATIO = CONFIG["algorithm_thresholds"]["motion_threshold_ratio"]
MOTION_COOLDOWN_TIME = CONFIG["algorithm_thresholds"]["motion_cooldown_time"]

# ==============================================================================

# 动态添加环境路径
sys.path.insert(0, YOLOV5_PATH)
if MVSDK_PYTHON_PATH not in sys.path:
    sys.path.insert(0, MVSDK_PYTHON_PATH)
os.environ["LD_LIBRARY_PATH"] = MVSDK_LIB_PATH + ":" + os.environ.get("LD_LIBRARY_PATH", "")

if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    os.environ["DISPLAY"] = ":0"

# ==================== 📁 日期文件夹与动态日志配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDS_ROOT = os.path.join(BASE_DIR, "records")
os.makedirs(RECORDS_ROOT, exist_ok=True)

def cleanup_old_records(root_dir, keep_days):
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
    cleanup_old_records(RECORDS_ROOT, keep_days=KEEP_RECORD_DAYS)
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_dir = os.path.join(RECORDS_ROOT, today_str)
    os.makedirs(today_dir, exist_ok=True)
    return today_dir

current_record_dir = get_today_record_dir()

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

# ==================== 🛑 系统关机/信号捕获钩子 ====================
def system_shutdown_handler(signum=None, frame=None):
    logger.info(f"🛑 收到进程终止信号 (signal: {signum})，门禁控制系统正在安全退出...")
    try:
        if ser and ser.is_open:
            ser.close()
        mqtt_client.loop_stop()
    except Exception:
        pass
    time.sleep(0.5)
    sys.exit(0)

signal.signal(signal.SIGTERM, system_shutdown_handler)
signal.signal(signal.SIGINT, system_shutdown_handler)

# ==================== 🚪 门禁物理状态机与动态加载冷却逻辑 ====================
CURRENT_GATE_STATE = "Locked"  
LAST_COMMAND_TIME = 0          
mqtt_client = mqtt.Client(client_id="JetsonTX2_GateController")
_mqtt_connected_flag = False

def toggle_gate_state(trigger_source="User Action"):
    global CURRENT_GATE_STATE, LAST_COMMAND_TIME
    
    # 🌟 运行时动态拉取最新的 YAML 冷却期参数
    latest_cfg = load_hardware_config()
    current_cooldown = latest_cfg["gate_logic"].get("cooldown_time", COOLDOWN_TIME)
    
    current_time = time.time()
    if current_time - LAST_COMMAND_TIME < current_cooldown:
        remains = int(current_cooldown - (current_time - LAST_COMMAND_TIME))
        logger.warning(f"⏳ [{trigger_source}] 系统处于冷却期（剩余 {remains} 秒，动态配置），已严格拦截！")
        return CURRENT_GATE_STATE

    LAST_COMMAND_TIME = current_time

    if CURRENT_GATE_STATE == "Locked":
        target_state = "Unlocked"
        action_name = "开启 (Unlock)"
    else:
        target_state = "Locked"
        action_name = "关闭 (Lock)"

    logger.info(f"🔊 [{trigger_source}] 触发物理脉冲，准备切换门状态为: {action_name}...")

    if ser and ser.is_open:
        try:
            ser.break_condition = True  
            time.sleep(JOG_TIME)        
            ser.break_condition = False 
            logger.info(f"🔌 [{trigger_source}] 继电器脉冲执行完毕。")
        except Exception as e:
            logger.error(f"❌ [{trigger_source}] 继电器驱动失败: {e}")
    else:
        logger.warning(f"⚠️ [{trigger_source}] 串口未打开，模拟状态翻转。")

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
        client.publish(MQTT_STATUS_TOPIC, CURRENT_GATE_STATE, qos=1, retain=True)
    else:
        logger.error(f"❌ MQTT 连接失败，返回码: {rc}")

def on_mqtt_message(client, userdata, msg):
    payload = msg.payload.decode('utf-8').strip().upper()
    logger.info(f"📩 [MQTT Command] 收到来自主题 [{msg.topic}] 的指令: {payload}")
    if payload in ["OPEN", "UNLOCK", "LOCK", "TOGGLE"]:
        toggle_gate_state(trigger_source="Home Assistant Dashboard")

mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_message = on_mqtt_message

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
except Exception as e:
    logger.error(f"❌ 无法连接 MQTT Broker ({MQTT_BROKER}:{MQTT_PORT}): {e}")

# ==================== 🗣️ ALEXA 语音控制模块 ====================
SSDP_RESPONSE = (
    "HTTP/1.1 200 OK\r\n"
    "CACHE-CONTROL: max-age=86400\r\n"
    "DATE: Tue, 03 Nov 2026 00:00:00 GMT\r\n"
    "EXT:\r\n"
    "LOCATION: http://{ip}:{port}/setup.xml\r\n"
    "OPT: \"http://schemas.upnp.org/upnp/1/0/\"; ns=01\r\n"
    "01-NLS: {uuid}\r\n"
    "SERVER: Unspecified, UPnP/1.0, Unspecified\r\n"
    "ST: urn:Belkin:device:**\r\n"
    "USN: uuid:{uuid}::urn:Belkin:device:**\r\n"
    "\r\n"
)

SETUP_XML_TEMPLATE = f"""<root xmlns="urn:schemas-upnp-org:device-1-0">
<device>
<deviceType>urn:MakerModel:device:controllee:1</deviceType>
<friendlyName>{DEVICE_NAME}</friendlyName>
<manufacturer>Belkin International Inc.</manufacturer>
<modelName>Emulated Socket</modelName>
<modelNumber>3.1415</modelNumber>
<UDN>uuid:{DEVICE_UUID}</UDN>
</device>
</root>
"""

def ssdp_listener():
    MCAST_GRP = '239.255.255.250'
    MCAST_PORT = 1900
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((LOCAL_IP, MCAST_PORT))
    mreq = socket.inet_aton(MCAST_GRP) + socket.inet_aton(LOCAL_IP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    logger.info(f"🎙️ SSDP 监听服务已成功绑定到网卡 IP: {LOCAL_IP}:{MCAST_PORT}")

    last_log_times = {}
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            if addr[0].startswith("172.") or addr[0].startswith("127.") or addr[0].startswith("192.168.122."):
                continue
            if b'M-SEARCH' in data and (b'upnp:rootdevice' in data or b'ssdp:all' in data or b'belkin' in data.lower()):
                resp = SSDP_RESPONSE.format(ip=LOCAL_IP, port=WEMO_PORT, uuid=DEVICE_UUID).encode()
                sock.sendto(resp, addr)
                if addr[0] != LOCAL_IP:
                    now_t = time.time()
                    if now_t - last_log_times.get(addr[0], 0) > 10.0:
                        last_log_times[addr[0]] = now_t
                        logger.info(f"📡 [SSDP] 收到来自 {addr[0]} 的发现请求，已成功响应 WeMo 身份！")
        except Exception:
            pass

class WeMoHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/setup.xml':
            self.send_response(200)
            self.send_header('Content-Type', 'text/xml')
            self.end_headers()
            self.wfile.write(SETUP_XML_TEMPLATE.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        client_ip = self.client_address[0]
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8', errors='ignore')
        soap_action = self.headers.get('SOAPACTION', '')
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/xml')
        self.end_headers()

        if "GetBinaryState" in soap_action or "GetBinaryState" in body:
            current_bin_state = "0" if CURRENT_GATE_STATE == "Locked" else "1"
            logger.info(f"💓 [心跳同步] 收到来自 {client_ip} 的状态查询心跳，当前门状态: {CURRENT_GATE_STATE}")
            soap_resp = (
                "<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\" s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\">"
                "<s:Body><u:GetBinaryStateResponse xmlns:u=\"urn:Belkin:service:basicevent:1\">"
                f"<BinaryState>{current_bin_state}</BinaryState></u:GetBinaryStateResponse></s:Body></s:Envelope>"
            )
        elif "SetBinaryState" in soap_action or "SetBinaryState" in body:
            target_bin = "1" if "<BinaryState>1</BinaryState>" in body else "0"
            logger.info(f"🎮 [控制指令] 收到来自 {client_ip} 的操作请求 (BinaryState: {target_bin})")
            threading.Thread(target=toggle_gate_state, args=(f"Alexa 语音 ({client_ip})",), daemon=True).start()
            soap_resp = (
                "<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\" s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\">"
                "<s:Body><u:SetBinaryStateResponse xmlns:u=\"urn:Belkin:service:basicevent:1\">"
                f"<BinaryState>{target_bin}</BinaryState></u:SetBinaryStateResponse></s:Body></s:Envelope>"
            )
        else:
            soap_resp = "<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\"><s:Body></s:Body></s:Envelope>"

        self.wfile.write(soap_resp.encode('utf-8'))

    def log_message(self, format, *args):
        pass

def wemo_http_server():
    server = HTTPServer(('0.0.0.0', WEMO_PORT), WeMoHTTPHandler)
    logger.info(f"🌐 WeMo 模拟控制 HTTP 服务已启动，监听端口: {WEMO_PORT}")
    server.serve_forever()

def start_alexa_service():
    threading.Thread(target=ssdp_listener, daemon=True).start()
    threading.Thread(target=wemo_http_server, daemon=True).start()
    logger.info("🎙️ Alexa 语音控制后台线程初始化完成！")

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

# ==================== 👁️ Dlib 初始化 ====================
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

# ==================== 📷 华睿相机封装类 ====================
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

        # 🌟 核心修改：计算实际行宽（有效宽度 + paddingX 填充列）
        effective_width = imageParams.width + imageParams.paddingX

        if imageParams.pixelForamt == EPixelType.gvspPixelMono8:
            grayByteArray = bytearray(userBuff)
            # 先按包含 padding 的实际行宽 reshape
            cvImage = np.array(grayByteArray).reshape(imageParams.height, effective_width)
            # 裁剪掉右侧的 paddingX 填充列，只保留有效宽度
            if imageParams.paddingX > 0:
                cvImage = cvImage[:, :imageParams.width]
            cvImage = cv2.cvtColor(cvImage, cv2.COLOR_GRAY2BGR)
        else:
            # 彩色格式转换
            rgbSize = c_int()
            rgbBuff = c_buffer(b'\0', imageParams.height * effective_width * 3)
            IMGCNV_ConvertToBGR24(cast(userBuff, c_void_p), byref(imageParams), cast(rgbBuff, c_void_p), byref(rgbSize))
            colorByteArray = bytearray(rgbBuff)
            # 按实际行宽 reshape 后裁剪掉填充列
            cvImage = np.array(colorByteArray).reshape(imageParams.height, effective_width, 3)
            if imageParams.paddingX > 0:
                cvImage = cvImage[:, :imageParams.width, :]
                
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

start_alexa_service()

# ==================== 📸 抓拍与事件处理 ====================
def save_snapshot_image(frame, stage_tag, raw_name):
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
    global last_detect_snapshot_time
    now = time.time()
    if now - last_detect_snapshot_time > DETECT_COOLDOWN:
        last_detect_snapshot_time = now
        logger.info(f"🔍 [检测到人脸] 候选人: {raw_name} | 置信度: {conf * 100:.1f}%")
        save_snapshot_image(frame, "DETECTED", raw_name)

def trigger_door_lock(user_name, conf, min_dist, current_frame):
    global last_trigger_time, LIVENESS_PASSED
    last_trigger_time = time.time()
    LIVENESS_PASSED = False
    
    if current_frame is not None:
        save_snapshot_image(current_frame, "MATCH", user_name)
    logger.info(f"🔓 [MATCH 成功] 身份判定为: {user_name} ({conf * 100:.0f}%) | Dist:{min_dist:.3f}")

    toggle_gate_state(trigger_source=f"人脸识别: {user_name}")

def handle_unknown_person(frame, label_text):
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

last_motion_snapshot_time = 0   
previous_gray_frame = None      

while True:  
    if not cam.open():
        logger.warning("⚠️ 相机未连接，5 秒后重试...")
        time.sleep(5)
        continue

    consecutive_none_frames = 0  

    while True:
        frame = cam.get_frame()
        if frame is None:
            consecutive_none_frames += 1
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
        # 🌊 帧差法运动检测
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
                    logger.info(f"🌊 [运动抓拍] 监听到画面变化率为 {change_ratio * 100:.1f}%，已强制抓拍！")
                    save_snapshot_image(frame, "MOTION", "Unknown")

        previous_gray_frame = gray_current.copy()

        # 🔍 YOLO 目标检测 (每 SKIP_RATIO 帧做一次，动态热加载置信度阈值)
        if frame_counter % SKIP_RATIO == 0 or len(last_boxes) == 0:
            img_resized = cv2.resize(frame, (INFERENCE_SIZE, INFERENCE_SIZE))
            img = torch.from_numpy(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)).to(device).permute(2, 0, 1).float() / 255.0
            if len(img.shape) == 3: 
                img = img.unsqueeze(0)

            with torch.no_grad():
                pred = model(img, augment=False, visualize=False)[0]

            # 🌟 每次推理动态加载最新的置信度阈值
            current_conf_thresh = load_hardware_config()["algorithm_thresholds"].get("face_confidence_threshold", YOLO_CONF_THRESHOLD)
            pred = non_max_suppression(pred, conf_thres=current_conf_thresh, iou_thres=0.45)

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
        
        # 🌟 动态加载最新的冷却时间与匹配严格度阈值
        live_cfg = load_hardware_config()
        live_cooldown = live_cfg["gate_logic"].get("cooldown_time", COOLDOWN_TIME)
        live_strict_threshold = live_cfg["algorithm_thresholds"].get("strict_threshold", STRICT_THRESHOLD)

        time_since_last_trigger = now_time - last_trigger_time

        for xyxy, conf, raw_name in last_boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            y1, y2, x1, x2 = max(0, y1), min(frame.shape[0], y2), max(0, x1), min(frame.shape[1], x2)
            
            pad_x, pad_y = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
            x1_p, y1_p = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2_p, y2_p = min(w_f, x2 + pad_x), min(h_f, y2 + pad_y)

            face_crop = frame[y1_p:y2_p, x1_p:x2_p]
            if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                continue

            # ------------------------------------------------------------------
            # ⏱️ 动态冷却期判断
            # ------------------------------------------------------------------
            if time_since_last_trigger < live_cooldown:
                remains = int(live_cooldown - time_since_last_trigger)
                current_label = f'COOLDOWN ({remains}s) | {raw_name}'
                box_color = YELLOW_COLOR
                annotator.box_label(xyxy, current_label, color=box_color)
                
                if current_label != last_printed_label:
                    logger.info(f"当前门禁状态: {current_label}")
                    last_printed_label = current_label
                continue

            handle_face_detected(frame, raw_name, conf)

            # ------------------------------------------------------------------
            # 👁 眨眼活体检测
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
            # 🔓 实时特征比对 (使用动态加载的严格阈值)
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

                    if min_dist < live_strict_threshold:
                        is_match = True
                        current_label = f'MATCH | {raw_name} ({conf * 100:.0f}%) | Dist:{min_dist:.3f}'
                        box_color = BLUE_COLOR
                    else:
                        current_label = f'UNKNOWN | Insufficient similarity ({conf * 100:.0f}%) | Dist:{min_dist:.3f}'
                        box_color = RED_COLOR
                else:
                    current_label = f'UNKNOWN ({conf * 100:.0f}%)'
                    box_color = RED_COLOR
            except Exception:
                current_label = f'RUNNING ({conf * 100:.0f}%)'

            if is_match:
                if LIVENESS_PASSED:
                    trigger_door_lock(raw_name, conf, min_dist, frame)
                else:
                    current_label = f'PLEASE BLINK | {current_label}'
            elif "UNKNOWN" in current_label:
                handle_unknown_person(frame, current_label)

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