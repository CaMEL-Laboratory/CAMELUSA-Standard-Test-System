import serial
import time

# 打开串口
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

print("⚡ 切换为【真实数据脉冲模式】测试...")

try:
    while True:
        input("👉 按回车发送开门信号...")
        
        print("🔓 正在通过 TX 发送低电平脉冲...")
        # 持续发送空字符 \x00，在物理上会强行把 TX 引脚拉低到 0V
        # 9600波特率下，发送约 100 字节可以维持约 0.1 秒的低电平
        for _ in range(5):
            ser.write(b'\x00' * 100) 
            time.sleep(0.1)
            
        print("🔒 停止发送，引脚恢复默认高电平。\n")

except KeyboardInterrupt:
    ser.close()