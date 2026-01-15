
import cv2
import numpy as np
import warnings
# 忽略NumPy和TensorFlow的兼容性警告
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model  # 新增：加载保存的模型
from PIL import Image

# 训练MNIST模型
def train_mnist_model():
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 数据预处理（核心：保证数据格式正确，避免兼容性问题）
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # 构建CNN模型（简化版，训练更快，新手友好）
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # 编译&训练（减少epochs，加快训练速度，满足基础实验需求）
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("开始训练模型...")
    model.fit(x_train, y_train, epochs=2, batch_size=32, validation_data=(x_test, y_test), verbose=1)
    print("模型训练完成！")
    return model

# 3. 预处理学号照片（分割单个数字）
def preprocess_student_id_image(image_path):
    # 读取图片→灰度化→二值化
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片：{image_path}，请检查路径是否正确")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 二值化（反相，数字为白色，背景为黑色）+ 降噪
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        # 形态学开运算，去除小噪声点
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 查找轮廓→筛选数字区域
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_rois = []
        # 按x坐标排序（保证学号数字顺序）
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 过滤太小的噪声区域（根据实际图片调整阈值）
            if w > 10 and h > 20:
                # 裁剪数字区域
                roi = binary[y:y+h, x:x+w]
                # 调整为28×28（MNIST输入格式）
                roi = cv2.resize(roi, (28, 28))
                # 增加通道维度 + 归一化
                roi = roi.reshape(1, 28, 28, 1).astype('float32') / 255.0
                digit_rois.append(roi)
        return digit_rois
    except Exception as e:
        print(f"图片预处理出错：{e}")
        return []

# 4. 识别学号
def recognize_student_id(image_path, model):
    digit_rois = preprocess_student_id_image(image_path)
    if not digit_rois:
        return "未识别到数字"
    student_id = ''
    for roi in digit_rois:
        prediction = model.predict(roi, verbose=0)
        digit = np.argmax(prediction)
        student_id += str(digit)
    return student_id

# 主程序（增加异常处理，确保model变量始终被定义）
if __name__ == '__main__':
    model = None  # 先初始化model变量，避免未定义
    try:
        # 优先加载已保存的模型（避免重复训练）
        model = load_model('mnist_digit_model.h5')
        print("成功加载已保存的模型！")
    except:
        # 加载失败则重新训练
        print("未找到已保存的模型，开始训练新模型...")
        model = train_mnist_model()
        # 保存模型（此时model已正确定义，不会报错）
        model.save('mnist_digit_model.h5')
        print("模型已保存为 mnist_digit_model.h5")


    image_path = 'student_id.jpg'  #
    # 识别学号
    result = recognize_student_id(image_path, model)
    print(f"\n最终识别结果：{result}")
