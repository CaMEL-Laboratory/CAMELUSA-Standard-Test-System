import os
os.environ["DISPLAY"] = ":0"  # 强制指定使用 Jetson 的本地图形桌面

import sys
import cv2
import time
import numpy as np
import torch
import serial  # 引入串口通信库
import face_recognition
from ctypes import *

# ==================== 🛠️ 华睿 MVSDK 动态库与路径修复 ====================
# 1. 设置系统动态库搜索路径
os.environ["LD_LIBRARY_PATH"] = "/opt/GeneralMVTech/MVviewer/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

# 2. 将 SDK 的 Python 目录加入 sys.path
mvsdk_python_path = '/opt/GeneralMVTech/MVviewer/share/Python'
if mvsdk_python_path not in sys.path:
    sys.path.insert(0, mvsdk_python_path)

# 3. 切换工作目录加载 MVSDK，解决 ../../lib/libMVSDK.so 相对路径找不到的问题
original_cwd = os.getcwd()
try:
    os.chdir(mvsdk_python_path)
    from MVSDK import *
    from ImageConvert import *
    print("✅ 成功加载华睿相机 MVSDK 库！")
except Exception as e:
    print(f"❌ 加载 MVSDK 失败: {e}")
finally:
    # 恢复原项目工作目录
    os.chdir(original_cwd)

# ==================== ⚙️ 全局变量与环境配置 ====================
last_trigger_time = 0  # 记录上次触发的时间
COOLDOWN_TIME = 30     # 冷却时间 30 秒
JOG_TIME = 2           # 点动时间 2 秒

# 运行环境设置
yolov5_path = r"/home/nvidia/yolov5"
sys.path.insert(0, yolov5_path)
os.chdir(yolov5_path)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator

# 核心参数调节区
model_path = r"/home/nvidia/yolov5/best.pt"
FACE_DB_DIR = r"/home/nvidia/yolov5/ID"

# 串口配置区
SERIAL_PORT = '/dev/ttyUSB0'  
BAUDRATE = 9600  # 默认波特率 9600

YOLO_CONF_THRESHOLD = 0.35  # YOLO 检测灵敏度
STRICT_THRESHOLD = 0.38     # 人脸比对严格度
OPEN_DURATION = 5          # 开门持续时间（秒）
BLUE_COLOR = (255, 0, 0)    # 文字边框颜色

name_map = {0: "Amanda", 1: "Ines", 2: "Eric", 3: "Ethan"}

# ==================== ☀️ 相机曝光、增益与帧率控制参数 ====================
CAM_EXPOSURE_TIME = 20000.0  # 曝光时间 (μs)，20ms 适合白天
CAM_GAIN_RAW = 5.0           # 增益 (0~24)
TARGET_FPS = 12.0            # 🔥 将相机出图帧率限制在 12 FPS (范围 10~15)

# ==================== 📷 华睿 SDK 工业相机封装类 ====================
class DahuaCamera:
    def __init__(self):
        self.camera = None
        self.streamSource = None
        self.system = None

    def set_camera_exposure_and_gain(self, exposure_us, gain_val):
        """配置相机的曝光和增益参数"""
        if not self.camera:
            return

        # 1. 禁用自动曝光，开启手动曝光模式
        expAutoEnumNode = pointer(GENICAM_EnumNode())
        expAutoInfo = GENICAM_EnumNodeInfo()
        expAutoInfo.pCamera = pointer(self.camera)
        expAutoInfo.attrName = b"ExposureAuto"
        if GENICAM_createEnumNode(byref(expAutoInfo), byref(expAutoEnumNode)) == 0:
            expAutoEnumNode.contents.setValueBySymbol(expAutoEnumNode, b"Off")
            expAutoEnumNode.contents.release(expAutoEnumNode)

        # 2. 设置曝光时间 (ExposureTime)
        expNode = pointer(GENICAM_DoubleNode())
        expInfo = GENICAM_DoubleNodeInfo()
        expInfo.pCamera = pointer(self.camera)
        expInfo.attrName = b"ExposureTime"
        if GENICAM_createDoubleNode(byref(expInfo), byref(expNode)) == 0:
            expNode.contents.setValue(expNode, c_double(exposure_us))
            expNode.contents.release(expNode)
            print(f"☀️ 已写入曝光时间: {exposure_us} μs")

        # 3. 禁用自动增益，开启手动增益
        gainAutoEnumNode = pointer(GENICAM_EnumNode())
        gainAutoInfo = GENICAM_EnumNodeInfo()
        gainAutoInfo.pCamera = pointer(self.camera)
        gainAutoInfo.attrName = b"GainAuto"
        if GENICAM_createEnumNode(byref(gainAutoInfo), byref(gainAutoEnumNode)) == 0:
            gainAutoEnumNode.contents.setValueBySymbol(gainAutoEnumNode, b"Off")
            gainAutoEnumNode.contents.release(gainAutoEnumNode)

        # 4. 设置增益 (GainRaw)
        gainNode = pointer(GENICAM_DoubleNode())
        gainInfo = GENICAM_DoubleNodeInfo()
        gainInfo.pCamera = pointer(self.camera)
        gainInfo.attrName = b"GainRaw"
        if GENICAM_createDoubleNode(byref(gainInfo), byref(gainNode)) == 0:
            gainNode.contents.setValue(gainNode, c_double(gain_val))
            gainNode.contents.release(gainNode)
            print(f"☀️ 已写入增益 (GainRaw): {gain_val}")

    def set_camera_fps(self, fps_val):
        """🔥 在相机硬件上开启并限制采集帧率 (AcquisitionFrameRate)"""
        if not self.camera:
            return

        # 1. 开启硬件帧率控制使能 (AcquisitionFrameRateEnable = True)
        fpsEnableNode = pointer(GENICAM_BoolNode())
        fpsEnableInfo = GENICAM_BoolNodeInfo()
        fpsEnableInfo.pCamera = pointer(self.camera)
        fpsEnableInfo.attrName = b"AcquisitionFrameRateEnable"
        if GENICAM_createBoolNode(byref(fpsEnableInfo), byref(fpsEnableNode)) == 0:
            fpsEnableNode.contents.setValue(fpsEnableNode, c_bool(True))
            fpsEnableNode.contents.release(fpsEnableNode)

        # 2. 写入目标采集帧率 (AcquisitionFrameRate)
        fpsNode = pointer(GENICAM_DoubleNode())
        fpsInfo = GENICAM_DoubleNodeInfo()
        fpsInfo.pCamera = pointer(self.camera)
        fpsInfo.attrName = b"AcquisitionFrameRate"
        if GENICAM_createDoubleNode(byref(fpsInfo), byref(fpsNode)) == 0:
            fpsNode.contents.setValue(fpsNode, c_double(fps_val))
            fpsNode.contents.release(fpsNode)
            print(f"🎯 相机硬件采集帧率已强行限制为: {fps_val} FPS")

    def open(self):
        # 1. 获取系统单例
        self.system = pointer(GENICAM_System())
        if GENICAM_getSystemInstance(byref(self.system)) != 0:
            print("❌ 获取相机系统实例失败！")
            return False

        # 2. 发现/枚举相机
        cameraList = pointer(GENICAM_Camera())
        cameraCnt = c_uint()
        nRet = self.system.contents.discovery(self.system, byref(cameraList), byref(cameraCnt), c_int(GENICAM_EProtocolType.typeAll))
        
        if nRet != 0 or cameraCnt.value < 1:
            print(f"⚠️ 未发现工业相机 (Status={nRet}, Count={cameraCnt.value})")
            return False

        print(f"✅ 成功找到 {cameraCnt.value} 台工业相机")
        self.camera = cameraList[0]

        # 3. 强制解锁上一次的心跳占用
        try:
            self.camera.disConnect(byref(self.camera))
            time.sleep(0.5)
        except Exception:
            pass

        # 4. 尝试连接（带 3 次重试机制）
        connect_success = False
        for attempt in range(3):
            nRet = self.camera.connect(self.camera, c_int(GENICAM_ECameraAccessPermission.accessPermissionControl))
            if nRet == 0:
                connect_success = True
                break
            print(f"⚠️ 连接相机第 {attempt + 1} 次尝试失败 (Error Code={nRet})，等待 1.5 秒后重试...")
            time.sleep(1.5)

        if not connect_success:
            print("❌ 工业相机连接失败，请检查 USB 接口或重新插拔相机！")
            return False

        print("✅ 工业相机连接成功！")

        # 5. 写入曝光、增益与帧率限制配置
        try:
            self.set_camera_exposure_and_gain(CAM_EXPOSURE_TIME, CAM_GAIN_RAW)
            self.set_camera_fps(TARGET_FPS)
        except Exception as e:
            print(f"⚠️ 设置相机硬件参数失败: {e}")

        # 6. 创建流对象
        streamSourceInfo = GENICAM_StreamSourceInfo()
        streamSourceInfo.channelId = 0
        streamSourceInfo.pCamera = pointer(self.camera)
        self.streamSource = pointer(GENICAM_StreamSource())
        if GENICAM_createStreamSource(pointer(streamSourceInfo), byref(self.streamSource)) != 0:
            print("❌ 创建相机流对象失败！")
            return False

        # 7. 设置自由拉流 (TriggerMode = Off)
        trigModeEnumNode = pointer(GENICAM_EnumNode())
        trigModeEnumNodeInfo = GENICAM_EnumNodeInfo()
        trigModeEnumNodeInfo.pCamera = pointer(self.camera)
        trigModeEnumNodeInfo.attrName = b"TriggerMode"
        GENICAM_createEnumNode(byref(trigModeEnumNodeInfo), byref(trigModeEnumNode))
        trigModeEnumNode.contents.setValueBySymbol(trigModeEnumNode, b"Off")
        trigModeEnumNode.contents.release(trigModeEnumNode)

        # 8. 开始拉流（使用最新帧覆盖策略 grabStrartegyLatestImage，确保拿到的永远是最新图像，无积压延迟）
        if self.streamSource.contents.startGrabbing(self.streamSource, c_ulonglong(0), c_int(GENICAM_EGrabStrategy.grabStrartegyLatestImage)) != 0:
            print("❌ 相机开始拉流失败！")
            return False

        print("🚀 工业相机已成功启动拉流！")
        return True

    def get_frame(self):
        """主动从 SDK 驱动缓存读取一帧 NumPy BGR 格式图像"""
        if not self.streamSource:
            return None

        frame = pointer(GENICAM_Frame())
        nRet = self.streamSource.contents.getFrame(self.streamSource, byref(frame), c_uint(1000))
        if nRet != 0 or frame.contents.valid(frame) != 0:
            return None

        # 参数提取
        imageParams = IMGCNV_SOpenParam()
        imageParams.dataSize    = frame.contents.getImageSize(frame)
        imageParams.height      = frame.contents.getImageHeight(frame)
        imageParams.width       = frame.contents.getImageWidth(frame)
        imageParams.paddingX    = frame.contents.getImagePaddingX(frame)
        imageParams.paddingY    = frame.contents.getImagePaddingY(frame)
        imageParams.pixelForamt = frame.contents.getImagePixelFormat(frame)

        # 拷贝裸数据
        imageBuff = frame.contents.getImage(frame)
        userBuff = c_buffer(b'\0', imageParams.dataSize)
        memmove(userBuff, c_char_p(imageBuff), imageParams.dataSize)

        # 及时释放底层图像帧
        frame.contents.release(frame)

        # 图像格式转换 (Mono8 或 BGR24)
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
        """安全释放相机流与设备连接"""
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
        print("🔌 工业相机资源已释放。")

# ==================== 🔌 串口初始化与防护 ====================
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
    ser.break_condition = False  # 强行锁定引脚，防止误触发
    print(f"✅ 串口 {SERIAL_PORT} 初始化成功，已强行锁定引脚，防止误触发！")
except Exception as e:
    print(f"⚠️ 串口未打开 ({e})，程序将在无门禁控制状态下正常运行预览。")

# ==================== 🧠 初始化 YOLO 与人脸库 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(model_path, map_location=device)
model.eval()

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

# --- 🚪 触发开门动作 ---
def trigger_door_lock():
    global last_trigger_time

    current_time = time.time()
    if current_time - last_trigger_time < COOLDOWN_TIME:
        print(f"⏳ 冷却中，请等待 {int(COOLDOWN_TIME - (current_time - last_trigger_time))} 秒后再次尝试。")
        return

    if ser and ser.is_open:
        try:
            print(f"🔓 [点动开门] 触发继电器 {JOG_TIME} 秒...")
            ser.break_condition = True
            time.sleep(JOG_TIME)
            ser.break_condition = False
            
            last_trigger_time = time.time()
            print("🔒 [锁定] 继电器已释放，进入 30 秒冷却期。")
        except Exception as e:
            print(f"❌ 继电器驱动触发失败: {e}")
    else:
        print("⚠️ 继电器串口未连接，跳过开门操作。")

# ==================== 🚀 主业务循环 ====================
cam = DahuaCamera()
last_printed_label = None

while True:  # 允许设备掉线自动重连循环
    if not cam.open():
        print("⚠️ 相机未连接或未响应，5 秒后尝试重新连接...")
        time.sleep(5)
        continue

    while True:
        frame = cam.get_frame()
        if frame is None:
            print("❌ 获取画面失败/超时，准备尝试重连相机...")
            cam.close()
            time.sleep(2)
            break  # 跳出内层抓图循环，进入重连机制

        time_remains = 0
        if door_opened:
            elapsed = time.time() - last_open_time
            if elapsed > OPEN_DURATION:
                door_opened = False
                if ser and ser.is_open:
                    ser.break_condition = False
                print("🔒 门禁时间到，引脚已恢复高电平，自动恢复锁定状态")
            else:
                time_remains = OPEN_DURATION - elapsed

        # YOLO 推理预处理
        img = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device).permute(2, 0, 1).float() / 255.0
        if len(img.shape) == 3: 
            img = img.unsqueeze(0)

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
                                    trigger_door_lock()
                                else:
                                    if not label.startswith('MATCH'):
                                        label = f'Insufficient similarity | {raw_name} ({conf * 100:.0f}%) | Dist:{min_dist:.3f}'
                            else:
                                if not label.startswith('MATCH'):
                                    label = f'UNKNOWN ({conf * 100:.0f}%)'
                        except Exception as e:
                            if not label.startswith('MATCH'):
                                label = f'RUNNING ({conf * 100:.0f}%)'

                    annotator.box_label(xyxy, label, color=BLUE_COLOR)

        # 🖥️ 显示窗口
        cv2.imshow("Face Recognition", annotator.result())
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            cam.close()
            if ser and ser.is_open:
                ser.break_condition = False
                ser.close()
            cv2.destroyAllWindows()
            sys.exit(0)

        if label != last_printed_label:
            print(f"当前门禁状态: {label}")
            last_printed_label = label

        # 删除了原来主循环末尾的 time.sleep(0.1)，因为现在相机本身就是按 12 FPS 节奏产生图像，
        # get_frame() 会自动按照 12 FPS 的时序进行阻塞，不再需要手动休眠。

# 安全清理
cam.close()
if ser and ser.is_open:
    try:
        ser.break_condition = False
    except:
        pass
    ser.close()
    print("\n🔌 串口引脚已安全复位并关闭。")

cv2.destroyAllWindows()