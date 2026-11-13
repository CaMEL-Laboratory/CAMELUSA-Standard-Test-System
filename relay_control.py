import serial

# 使用二进制模式打开
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

# 使用 bytes 对象直接发送，绝不会被转码
cmd = bytes([0x01, 0x06, 0x00, 0x08, 0x00, 0x01, 0xC9, 0xC8])

ser.write(cmd)
print("已发送原始二进制数据")
ser.close()