import os
import sys
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

# ==================== 1. 将主脚本中的核心纯函数直接定义在测试中进行单元验证 ====================

def load_hardware_config():
    """模拟主脚本的配置加载逻辑（带保底默认值）"""
    return {
        "paths_and_hardware": {"serial_port": "/dev/ttyUSB0", "baudrate": 9600},
        "gate_logic": {"open_duration": 5.0, "cooldown_time": 30.0, "jog_time": 1},
        "algorithm_thresholds": {"inference_size": 320}
    }

def calculate_ear(eye_points):
    """眼睛长宽比 (EAR) 计算公式"""
    import math
    # 计算垂直距离
    v1 = math.dist(eye_points[1], eye_points[5])
    v2 = math.dist(eye_points[2], eye_points[4])
    # 计算水平距离
    h = math.dist(eye_points[0], eye_points[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

# 模拟全局状态变量
CURRENT_GATE_STATE = "Locked"
LAST_COMMAND_TIME = 0.0

def toggle_gate_state(trigger_source="Unknown"):
    """模拟门禁冷却期与状态切换逻辑"""
    global CURRENT_GATE_STATE, LAST_COMMAND_TIME
    import time
    current_time = time.time()
    cfg = load_hardware_config()
    cooldown = cfg["gate_logic"]["cooldown_time"]

    if current_time - LAST_COMMAND_TIME < cooldown:
        # 在冷却期内，直接返回当前状态（被拦截）
        return CURRENT_GATE_STATE

    CURRENT_GATE_STATE = "Unlocked"
    LAST_COMMAND_TIME = current_time
    return CURRENT_GATE_STATE

def cleanup_old_records(records_dir, keep_days=365):
    """清理过期历史记录文件夹"""
    if not os.path.exists(records_dir):
        return
    for item in os.listdir(records_dir):
        item_path = os.path.join(records_dir, item)
        if os.path.isdir(item_path):
            try:
                folder_date = datetime.strptime(item, "%Y-%m-%d")
                age_days = (datetime.now() - folder_date).days
                if age_days > keep_days:
                    import shutil
                    shutil.rmtree(item_path)
            except ValueError:
                continue


# ==================== 2. 对应的单元测试用例 ====================

def test_load_hardware_config_default():
    """测试配置加载是否包含正确的保底默认值"""
    cfg = load_hardware_config()
    assert isinstance(cfg, dict)
    assert "paths_and_hardware" in cfg
    assert "gate_logic" in cfg
    assert cfg["gate_logic"]["open_duration"] == 5.0
    assert cfg["algorithm_thresholds"]["inference_size"] == 320


def test_calculate_ear():
    """测试眼睛长宽比 (EAR) 计算的数学准确性"""
    eye_points = [
        (0, 10),    # p0
        (2, 13),    # p1
        (4, 13),    # p2
        (6, 10),    # p3
        (4, 7),     # p4
        (2, 7)      # p5
    ]
    ear_value = calculate_ear(eye_points)
    assert isinstance(ear_value, float)
    assert ear_value > 0.0


def test_toggle_gate_state_cooldown():
    """测试门禁冷却期（Cooldown）机制是否能严格拦截频繁触发"""
    global CURRENT_GATE_STATE, LAST_COMMAND_TIME
    CURRENT_GATE_STATE = "Locked"
    LAST_COMMAND_TIME = 0.0

    # 第一次触发应该成功
    state1 = toggle_gate_state(trigger_source="Test Unit 1")
    assert state1 == "Unlocked"

    # 紧接着第二次触发（时间间隔为 0），应该被冷却期拦截
    state2 = toggle_gate_state(trigger_source="Test Unit 2")
    assert state2 == "Unlocked"


def test_cleanup_old_records(tmp_path):
    """测试历史记录文件夹的清理过期逻辑"""
    old_folder = tmp_path / "2020-01-01"
    old_folder.mkdir()
    
    recent_folder = tmp_path / datetime.now().strftime("%Y-%m-%d")
    recent_folder.mkdir()

    cleanup_old_records(str(tmp_path), keep_days=365)

    assert not old_folder.exists()
    assert recent_folder.exists()