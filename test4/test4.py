# 1. torch的核心作用：提供模型运行的底层支持（无需显式调用，但模型依赖它）
import torch
# 2. torchvision：辅助图像预处理/格式转换
import torchvision.transforms as transforms
# 3. ultralytics：封装YOLO模型，简化检测流程
from ultralytics import YOLO
import cv2

# 加载模型（ultralytics封装，底层用torch实现网络计算）
model = YOLO('yolov5s.pt')

# 读取图像并预处理（torchvision辅助）
img = cv2.imread("campus_bicycle.jpg")
# torchvision的transforms可以标准化图像（适配YOLO输入）
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),  # YOLOv5默认输入尺寸
    transforms.ToTensor(),  # 转为torch张量（核心依赖torch）
])
img_tensor = preprocess(img)

# 模型推理（ultralytics封装，底层用torch做GPU/CPU计算）
results = model(img_tensor)

# 输出结果
results[0].show()
