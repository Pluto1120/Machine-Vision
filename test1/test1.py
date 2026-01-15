import cv2
import numpy as np
import matplotlib.pyplot as plt


img_path = "architecture.png"
# 以灰度图读取（方便滤波处理，若要处理彩色图需分通道）
img = cv2.imread(img_path, 0)
# 转换为浮点型（避免计算溢出）
img = img.astype(np.float32)


def convolution(img, kernel):
    # 获取图像和核的尺寸
    img_h, img_w = img.shape
    kernel_h, kernel_w = kernel.shape
    # 计算填充大小（保证输出尺寸与输入一致）
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    # 对图像进行零填充
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    # 初始化输出图像
    output = np.zeros_like(img)

    # 遍历每个像素（卷积计算）
    for i in range(img_h):
        for j in range(img_w):
            # 提取当前窗口
            window = padded_img[i:i + kernel_h, j:j + kernel_w]
            # 卷积（点乘后求和）
            output[i, j] = np.sum(window * kernel)

    # 归一化到0-255（避免显示异常）
    output = (output - output.min()) / (output.max() - output.min()) * 255
    return output.astype(np.uint8)
# Sobel x方向核（水平边缘检测）
sobel_x_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# 执行Sobel滤波
sobel_filtered = convolution(img, sobel_x_kernel)
# 题目中给定的卷积核
given_kernel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
], dtype=np.float32)

# 执行给定核滤波
given_filtered = convolution(img, given_kernel)
def compute_histogram(img):
    # 初始化直方图数组（0-255共256个bin）
    hist = np.zeros(256, dtype=np.int32)
    # 遍历每个像素
    for pixel in img.flatten():
        hist[int(pixel)] += 1
    return hist

# 计算原图的直方图
hist = compute_histogram(img.astype(np.uint8))

# 可视化直方图
plt.figure(figsize=(8, 4))
plt.bar(range(256), hist, width=1)
plt.title("Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Count")
plt.savefig("histogram.png")  # 保存直方图
plt.close()


def glcm_texture(img, distance=1, angle=0):
    # 灰度共生矩阵（GLCM）计算（简化版：仅统计0°、距离1的共生对）
    img = img.astype(np.uint8)
    glcm = np.zeros((256, 256), dtype=np.int32)

    # 遍历图像（避免边界越界）
    for i in range(img.shape[0]):
        for j in range(img.shape[1] - 1):
            # 当前像素与右侧像素的共生对
            pixel1 = img[i, j]
            pixel2 = img[i, j + 1]
            glcm[pixel1, pixel2] += 1

    # 提取纹理特征（以对比度为例，也可加能量、熵等）
    contrast = 0
    for i in range(256):
        for j in range(256):
            contrast += (i - j) ** 2 * glcm[i, j]

    # 保存纹理特征到npy
    np.save("texture_feature.npy", np.array([contrast]))
    return contrast


# 提取纹理特征
texture_feature = glcm_texture(img)
# 保存滤波后的图像
cv2.imwrite("sobel_filtered.jpg", sobel_filtered)
cv2.imwrite("given_filtered.jpg", given_filtered)
