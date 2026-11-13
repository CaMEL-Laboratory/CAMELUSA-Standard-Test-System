import time
import os

# 对应手册中的 GPIO8 (TX2原生编号 388)
GPIO_NUM = "388"
GPIO_PATH = f"/sys/class/gpio/gpio{GPIO_NUM}"

def setup_gpio():
    # 1. 导出 GPIO 端口
    if not os.path.exists(GPIO_PATH):
        try:
            with open("/sys/class/gpio/export", "w") as f:
                f.write(GPIO_NUM)
            time.sleep(0.2)  # 给系统一点点反应时间
        except Exception as e:
            print(f"导出 GPIO 失败 (可能需要 sudo 权限): {e}")
            return False
            
    # 2. 设置为输出模式 (out)
    with open(f"{GPIO_PATH}/direction", "w") as f:
        f.write("out")
    return True

def set_relay_state(state):
    # state 传入 1 为吸合，传入 0 为断开
    with open(f"{GPIO_PATH}/value", "w") as f:
        f.write(str(state))

if __name__ == "__main__":
    if setup_gpio():
        print("✅ MYNT CUBE 继电器控制通道（GPIO8）初始化成功！")
        try:
            while True:
                print("【继电器：吸合】(发出高电平)")
                set_relay_state(1)
                time.sleep(2)
                
                print("【继电器：断开】(发出低电平)")
                set_relay_state(0)
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n用户手动终止程序")
        finally:
            # 安全清理：取消导出 GPIO，保持系统干净
            if os.path.exists(GPIO_PATH):
                with open("/sys/class/gpio/unexport", "w") as f:
                    f.write(GPIO_NUM)
                print("🧹 GPIO 状态已安全清理。")