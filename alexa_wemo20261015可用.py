import socket
import threading
import time
import serial

# ==================== ⚙️ 配置参数 ====================
SERIAL_PORT = '/dev/ttyUSB0'  # 串口号
BAUDRATE = 9600               # 波特率
JOG_TIME = 2                 # 点动开门时间 (秒)

WEMO_PORT = 49153             # Fauxmo/WeMo 监听端口
DEVICE_NAME = "Gate Lock"     # Alexa 语音唤醒设备名
DEVICE_UUID = "Socket-1_0-220000000001"

# ==================== 🔌 继电器控制逻辑 ====================
def trigger_relay():
    """串口拉低/置位触发门禁继电器"""
    try:
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        print(f"🔊 [Alexa 指令] 正在触发继电器点动开门 {JOG_TIME} 秒...")
        ser.break_condition = True    # 吸合继电器
        time.sleep(JOG_TIME)
        ser.break_condition = False   # 恢复断开
        ser.close()
        print("✅ [Alexa 指令] 门锁触发成功，继电器已复位。")
        return True
    except Exception as e:
        print(f"❌ [Alexa 指令] 串口触发失败: {e}")
        return False

# ==================== 🌐 局域网网络服务 ====================
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
    """监听局域网 SSDP 广播并回应 Alexa 发现请求"""
    MCAST_GRP = '239.255.255.250'
    MCAST_PORT = 1900
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', MCAST_PORT))
    mreq = socket.inet_aton(MCAST_GRP) + socket.inet_aton('0.0.0.0')
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    print(f"📡 SSDP 广播服务启动，当前 TX2 本地 IP: {LOCAL_IP}")
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            if b'M-SEARCH' in data and (b'upnp:rootdevice' in data or b'ssdp:all' in data or b'belkin' in data.lower()):
                resp = SSDP_RESPONSE.format(ip=LOCAL_IP, port=WEMO_PORT, uuid=DEVICE_UUID).encode()
                sock.sendto(resp, addr)
        except Exception:
            pass

def wemo_http_server():
    """处理来自 Alexa 的 HTTP / SOAP 请求（完美兼容 GetBinaryState）"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', WEMO_PORT))
    server.listen(5)
    print(f"🏠 Alexa WeMo 模拟服务启动成功，监听端口: {WEMO_PORT}")

    setup_xml = f"""<?xml version="1.0"?>
<root xmlns="urn:schemas-upnp-org:device-1-0">
    <device>
        <deviceType>urn:MakerModel:device:controllee:1</deviceType>
        <friendlyName>{DEVICE_NAME}</friendlyName>
        <manufacturer>Belkin International Inc.</manufacturer>
        <modelName>Emulated Socket</modelName>
        <modelNumber>3.1415</modelNumber>
        <UDN>uuid:{DEVICE_UUID}</UDN>
    </device>
</root>"""

    while True:
        try:
            client, addr = server.accept()
            data = client.recv(2048).decode('utf-8', errors='ignore')

            # 1. 设备描述拉取
            if "GET /setup.xml" in data:
                resp = f"HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: {len(setup_xml)}\r\n\r\n{setup_xml}"
                client.sendall(resp.encode())

            # 2. 状态查询 (补全 GetBinaryState 相应，消除 Fauxmo 报错)
            elif "GetBinaryState" in data:
                soap_resp = """HTTP/1.1 200 OK\r\nContent-Type: text/xml; charset="utf-8"\r\n\r\n
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">
<s:Body><u:GetBinaryStateResponse xmlns:u="urn:Belkin:service:basicevent:1">
<BinaryState>0</BinaryState></u:GetBinaryStateResponse></s:Body></s:Envelope>"""
                client.sendall(soap_resp.encode())

            # 3. 开关动作控制 (SetBinaryState)
            elif "SetBinaryState" in data or "SOAPACTION" in data:
                threading.Thread(target=trigger_relay, daemon=True).start()
                
                soap_resp = """HTTP/1.1 200 OK\r\nContent-Type: text/xml; charset="utf-8"\r\n\r\n
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">
<s:Body><u:SetBinaryStateResponse xmlns:u="urn:Belkin:service:basicevent:1">
<BinaryState>1</BinaryState></u:SetBinaryStateResponse></s:Body></s:Envelope>"""
                client.sendall(soap_resp.encode())

            else:
                client.sendall(b"HTTP/1.1 200 OK\r\n\r\n")

            client.close()
        except Exception as e:
            pass

if __name__ == '__main__':
    t1 = threading.Thread(target=ssdp_listener, daemon=True)
    t2 = threading.Thread(target=wemo_http_server, daemon=True)
    t1.start()
    t2.start()

    print("🚀 Alexa 语音控制中间件准备就绪！按 Ctrl+C 退出。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 语音服务已停止。")