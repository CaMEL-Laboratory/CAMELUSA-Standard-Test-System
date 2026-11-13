# best.pt Yolov5 视频预测测试（清晰文字改良版）
import os
import sys
import cv2
import torch

# 1. 精准定位到你真正的 YOLOv5 v6.1 源码文件夹
yolov5_path = r"D:\YOLO_Project\FaceProGateAccess\yolov5-6.1"

if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)  # 把真正的源码目录加入环境变量
os.chdir(yolov5_path)  # 把当前工作目录切换过去

# 2. 导入 v6.1 版本对应的模块
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors

# 3. 载入模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = r"D:\YOLO_Project\FaceProGateAccess\best.pt"

model = attempt_load(model_path, map_location=device)
model.eval()

# 💡【全新功能：中文/英文人名花名册映射】
# 这里你可以自己修改。如果模型测出是类别 3，就会自动显示为 "User_3" 或对应的名字
# 这样可以彻底避免数字和置信度小数连在一起的误会！
name_map = {
    0: "Amanda",
    1: "Ines",
    2: "Eric",
    3: "Ethan"  # 👈 如果你想写英文名字，可以直接写如 3: "XiaoMing"
}

# 4. 打开电脑内置摄像头（0 代表默认摄像头）
cap = cv2.VideoCapture(0)
print("正在打开摄像头，提示：如果按 'q' 没反应，请切换到英文输入法，并点击一下视频窗口再按！")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("无法获取摄像头画面")
        break

    # 将 OpenCV 默认的 BGR 图像转为 YOLOv5 认识的 RGB 图像
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 5. YOLOv5 原生推理预处理
    img = torch.from_numpy(rgb_frame).to(device)
    img = img.permute(2, 0, 1).float()  # 改变通道顺序 (H,W,C) -> (C,H,W)
    img /= 255.0  # 归一化 [0, 1]
    if len(img.shape) == 3:
        img = img.unsqueeze(0)  # 增加 batch 维度

    # 6. 模型预测
    with torch.no_grad():
        pred = model(img, augment=False, visualize=False)[0]

    # 7. 非极大值抑制（设置置信度阈值 conf_thres=0.5）
    pred = non_max_suppression(pred, conf_thres=0.2, iou_thres=0.45, classes=None, agnostic=False)

    # 8. 将识别结果画回原图 (在原始的 frame 上画框)
    annotator = Annotator(frame, line_width=3, example=str(model.names))
    
    for i, det in enumerate(pred):  # 遍历检测到的物体
        if len(det):
            # 使用 scale_coords 将框的尺寸缩放到摄像头原图大小
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            # 画框和标签
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # 类别整数
                
                # 💡【核心修改：重新排版标签文本】
                # 首先去我们自定义的花名册找名字，如果找不到，就用模型自带的名字
                real_name = name_map.get(c, model.names[c])
                
                # 将置信度乘以 100 变成百分比形式（例如 0.85 变成 85%）
                # 格式变成： "User_3 (85%)"。这样有括号、有百分号，绝对不会和类别混淆！
                label = f'{real_name} ({conf*100:.0f}%)'
                
                # 在图像上绘制矩形框和清晰的文字
                annotator.box_label(xyxy, label, color=colors(c, True))

    # 拿到画好框的画面
    annotated_frame = annotator.result()

    # 展示画面
    cv2.imshow("YOLOv5 Face Recognition", annotated_frame)

    # 兼容大小写 q 以及 Esc 键退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 27:
        print("用户退出...")
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()