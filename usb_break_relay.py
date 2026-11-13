import serial
import time
import sys

# 1. 初始化串口（请根据实际情况修改串口号，Jetson TX2 通常是 /dev/ttyUSB0）
port = '/dev/ttyUSB0'
try:
    ser = serial.Serial(port, 9600, timeout=1)
    # 【核心优化】启动时，立刻强行将 TXD 压死在高电平（False 对应标准串口的空闲高电平状态）
    # 这样可以确保程序一启动，继电器绝不误触发，处于安全锁定状态
    ser.break_condition = False 
    print(f"✅ 串口 {port} 初始化成功。")
except Exception as e:
    print(f"❌ 无法打开串口 {port}，请检查设备是否插入或被占用！")
    print(f"错误原因: {e}")
    sys.exit()

print("\n🔒 【安全点动控制系统】已就绪")
print("--------------------------------------------------")
print("安全保障：门禁接线在【常开端(NO)】，停电或系统死机时门会自动锁死。")
print("工作原理：利用 USB 芯片强地电平（0V）吸电流特性，彻底解决驱动不足问题。")
print("--------------------------------------------------")

try:
    while True:
        # 等待回车键
        input("👉 请按【回车键 (Enter)】触发一次开门脉冲...")
        
        print("🔓 【开门中】TXD 强行拉低至 0V（吸电流触发）...")
        # ser.break_condition = True 会在硬件上强行将 TXD 引脚拉低到 0V
        ser.break_condition = True  
        time.sleep(0.5)             # 保持 0.5 秒的点动开门时间
        
        # 恢复高电平，断开继电器
        ser.break_condition = False 
        print("🔒 【已重新锁定】TXD 恢复高电平，门禁已关闭。\n")

except KeyboardInterrupt:
    # 优雅退出，退出前必须确保引脚处于安全的高电平（不触发状态）
    ser.break_condition = False
    ser.close()
    print("\n👋 程序已安全退出，串口资源已释放。")