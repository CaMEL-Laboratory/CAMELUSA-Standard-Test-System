import serial
import time
import threading

def run_one_direction(tx_port, rx_port, baud=9600):
    print(f"\n==========================================")
    print(f"🔄 开始测试：【发送方】{tx_port}  ===>  【接收方】{rx_port}")
    print(f"==========================================")
    
    try:
        # 打开两个串口，强制关闭所有流控
        ser_rx = serial.Serial(rx_port, baud, timeout=2, rtscts=False, xonxoff=False, dsrdtr=False)
        ser_tx = serial.Serial(tx_port, baud, timeout=1, rtscts=False, xonxoff=False, dsrdtr=False)
        
        # 清空缓冲区
        ser_rx.reset_input_buffer()
        ser_tx.reset_output_buffer()
        
        # 待发送的测试数据
        test_data = bytes([0x01, 0x06, 0x00, 0x08, 0x00, 0x01, 0xC9, 0xC8])
        
        # 定义接收线程
        received_holder = []
        def rx_worker():
            # 尝试读取 8 个字节
            data = ser_rx.read(8)
            if data:
                received_holder.append(data)
                
        rx_thread = threading.Thread(target=rx_worker)
        rx_thread.start()
        
        # 稍微等待接收端准备就绪后发送
        time.sleep(0.2)
        print(f"【发送方 {tx_port}】发出数据 (Hex): {test_data.hex().upper()}")
        ser_tx.write(test_data)
        ser_tx.flush()
        
        # 等待接收线程结束
        rx_thread.join()
        
        # 打印接收结果
        if received_holder:
            received_data = received_holder[0]
            print(f"【接收方 {rx_port}】收到数据 (Hex): {received_data.hex().upper()}")
            # 对比发送和接收的数据是否一致
            if test_data == received_data:
                print(f"✅ 通信成功！数据完全一致。")
            else:
                print(f"⚠️ 收到数据，但有差异（可能存在微弱时钟漂移或信号杂波）。")
        else:
            print(f"❌ 错误：【接收方 {rx_port}】超时，未收到任何数据！")
            
        ser_rx.close()
        ser_tx.close()
        
    except Exception as e:
        print(f"💥 运行中发生异常: {e}")

if __name__ == "__main__":
    # 1. 确保权限（如果你还没有赋予权限）
    print("提示: 运行前请确保已执行：sudo chmod 666 /dev/ttyUSB0 /dev/ttyUSB1")
    
    # 2. 第一轮：USB0 -> USB1
    run_one_direction(tx_port='/dev/ttyUSB0', rx_port='/dev/ttyUSB1')
    
    # 稍微间隔一下，准备反向测试
    time.sleep(1.5)
    
    # 3. 第二轮：USB1 -> USB0
    run_one_direction(tx_port='/dev/ttyUSB1', rx_port='/dev/ttyUSB0')
    
    print("\n测试结束。")