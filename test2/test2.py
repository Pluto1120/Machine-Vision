import cv2
import numpy as np

def detect_lane_lines(image_path):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像！")
        return
    height, width = img.shape[:2]
    img_copy = img.copy()

    # 2. 预处理：增强白色车道线+灰度+模糊+Canny（适配当前道路）
    # 增强白色区域对比度（针对性提升车道线清晰度）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    enhanced_white = cv2.bitwise_and(img, img, mask=white_mask)

    # 灰度+模糊+Canny（调整参数适配细车道线）
    gray = cv2.cvtColor(enhanced_white, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)  # 缩小模糊核，保留细车道线细节
    edges = cv2.Canny(blur, 30, 100)  # 降低阈值，捕捉细车道线边缘

    # 3. 兴趣区域掩码（覆盖全道路，避免漏检车道线）
    mask = np.zeros_like(edges)
    # 全道路矩形区域（适配俯视图视角，完整覆盖所有车道）
    roi_vertices = np.array([[
        (0, height),
        (0, height * 0.2),
        (width, height * 0.2),
        (width, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 4. 霍夫直线检测（优化参数，精准识别所有车道线）
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,  # 1像素高精度，适配细车道线
        theta=np.pi / 180,
        threshold=20,  # 降低阈值，识别细/短分段车道线
        minLineLength=50,  # 最小长度适配短车道线
        maxLineGap=10  # 缩小间隙，连接分段车道线
    )

    # 5. 绘制车道线（所有车道线使用单一颜色，移除多颜色判断）
    if lines is not None:
        # 统一设置单一颜色（BGR格式，可自行修改）
        lane_color = (0, 255, 0)  # 绿色（默认，可改为(255,0,0)蓝色、(0,0,255)红色等）
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 直接使用统一颜色绘制，无需再判断车道位置
            cv2.line(img_copy, (x1, y1), (x2, y2), lane_color, 4)  # 4像素线宽，更醒目

    # 6. 显示结果+保存
    cv2.imshow("Original", img)
    cv2.imshow("Lane Detection Result", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("lane_result.jpg", img_copy)

# 替换为你的道路图像实际路径
detect_lane_lines("campus_road.png")
