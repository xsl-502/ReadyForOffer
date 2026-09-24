# 所用软件为：PyCharm 2024.1.1

# 分类
import cv2
import os
import math
from imutils import paths

# 设置分类目录
low_light_dir = "LowLightImages"
color_bias_dir = "ColorBiasImages"
blurry_dir = "BlurryImages"
normal_dir = "NormalImages"

# 创建分类目录
for dir_path in [low_light_dir, color_bias_dir, blurry_dir, normal_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 检测是否为低光图片
def is_low_light(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r, c = gray_img.shape[:2]
    dark_sum = sum(sum(1 for colum in row if colum < 40) for row in gray_img)
    dark_prop = dark_sum / (r * c)
    return dark_prop >= 0.6

# 检测是否存在显著色偏
def has_color_bias(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    h, w = img.shape[:2]
    da = a_channel.sum() / (h * w) - 128
    db = b_channel.sum() / (h * w) - 128

    histA = [0] * 256
    histB = [0] * 256
    for i in range(h):
        for j in range(w):
            histA[a_channel[i][j]] += 1
            histB[b_channel[i][j]] += 1

    msqA = sum(float(abs(y - 128 - da)) * histA[y] / (w * h) for y in range(256))
    msqB = sum(float(abs(y - 128 - db)) * histB[y] / (w * h) for y in range(256))
    result = math.sqrt(da**2 + db**2) / math.sqrt(msqA**2 + msqB**2)

    return result > 3.5

# 检测是否为模糊图片
def is_blurry(img, threshold=250):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

# 图片分类处理
def classify_images(folder_path):
    for image_path in paths.list_images(folder_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            continue

        if is_low_light(image):
            save_path = os.path.join(low_light_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, image)
            print(f"图片 {image_path} 分类为低光")
        elif has_color_bias(image):
            save_path = os.path.join(color_bias_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, image)
            print(f"图片 {image_path} 分类为色偏")
        elif is_blurry(image):
            save_path = os.path.join(blurry_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, image)
            print(f"图片 {image_path} 分类为模糊")
        else:
            save_path = os.path.join(normal_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, image)
            print(f"图片 {image_path} 分类为色偏")

# 执行分类
if __name__ == "__main__":
    folder_path = "Attachment/Attachment 1"  # 替换为你的图片文件夹路径
    classify_images(folder_path)

#亮度检测 1
import cv2
import numpy as np
img = cv2.imread('Attachment/Attachment 1/image_115.png')
# 把图片转换为单通道的灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 获取形状以及长宽
img_shape = gray_img.shape
height, width = img_shape[0], img_shape[1]
size = gray_img.size
# 灰度图的直方图
hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
# 计算灰度图像素点偏离均值(128)程序
a = 0
ma = 0
#np.full 构造一个数组，用指定值填充其元素
reduce_matrix = np.full((height, width), 128)
shift_value = gray_img - reduce_matrix
shift_sum = np.sum(shift_value)
da = shift_sum / size
# 计算偏离128的平均偏差
for i in range(256):
    ma += (abs(i-128-da) * hist[i])
m = abs(ma / size)
# 亮度系数
k = abs(da) / m
print(k)
if k[0] > 1:
    # 过亮
    if da > 0:
        print("过亮")
    else:
        print("过暗")
else:
    print("亮度正常")

#亮度检测 2
import cv2
import numpy as np

def is_low_light(image_path, threshold=50):
    # 读取图片
    img = cv2.imread(image_path)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 计算灰度均值
    mean_brightness = np.mean(gray)
    print(f"Mean Brightness: {mean_brightness}")
    # 判断是否低光
    return mean_brightness < threshold

# 测试图片路径
image_path = "Attachment/Attachment 1/image_020.png"
if is_low_light(image_path):
    print("图片处于低光环境")
else:
    print("图片光线充足")

#亮度检测 3
import cv2
import os

def analyze_image(img, pic_path):
    """
    判断单张图片是否处于低光环境
    :param img: 读取的图片
    :param pic_path: 图片路径
    """
    # 把图片转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取灰度图矩阵的行数和列数
    r, c = gray_img.shape[:2]
    dark_sum = 0  # 偏暗的像素 初始化为0个
    piexs_sum = r * c  # 整个灰度图的像素个数为r*c

    # 遍历灰度图的所有像素
    for row in gray_img:
        for colum in row:
            if colum < 40:  # 人为设置的阈值，表示0~39的灰度值为暗
                dark_sum += 1
    dark_prop = dark_sum / piexs_sum  # 偏暗像素所占比例
    print(f"\n检测图片: {pic_path}")
    print(f"dark_sum: {dark_sum}")
    print(f"piexs_sum: {piexs_sum}")
    print(f"dark_prop: {dark_prop:.2f}")

    # 判断图片是否偏暗
    if dark_prop >= 0.75:  # 偏暗像素比例阈值
        print(f"{pic_path} is dark!")
        save_dark_image(img, pic_path)  # 保存到黑暗图片目录
    else:
        print(f"{pic_path} is bright!")

def save_dark_image(img, pic_path):
    """
    保存被判定为黑暗的图片
    :param img: 读取的图片
    :param pic_path: 原图片路径
    """
    dark_dir = "../DarkPicDir"
    # 创建黑暗图片存储目录
    if not os.path.exists(dark_dir):
        os.makedirs(dark_dir)
    
    # 提取图片文件名
    pic_name = os.path.basename(pic_path)
    save_path = os.path.join(dark_dir, pic_name)
    # 保存图片
    cv2.imwrite(save_path, img)
    print(f"图片已保存到: {save_path}")

if __name__ == "__main__":
    # 指定图片路径
    image_path = "Attachment/Attachment 1/image_019.png"  # 替换为实际图片路径
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"图片路径无效: {image_path}")
    else:
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
        else:
            analyze_image(img, image_path)

#模糊检测 3
# coding=utf-8
import cv2

def variance_of_laplacian(image):
    """
    计算图像的 Laplacian 响应的方差值
    :param image: 灰度图像
    :return: 方差值
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()

if __name__ == '__main__':
    # 指定图片路径
    image_path = "Attachment/Attachment 1/image_041.png"  # 替换为实际图片路径
    threshold = 100.0  # 设置模糊阈值

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
    else:
        # 将图片转换为灰度图片
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 计算灰度图片的方差
        fm = variance_of_laplacian(gray)
        text = "Not Blurry"

        # 判断图片是否模糊
        if fm < threshold:
            text = "Blurry"

        # 输出结果
        print(f"图片路径: {image_path}")
        print(f"清晰度: {fm:.2f}, 判断: {text}")

        # 在图片上显示清晰度结果
        cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

#偏色检测
import cv2
img = cv2.imread('Attachment\Attachment 1\image_193.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_channel, a_channel, b_channel = cv2.split(img)
h,w,_ = img.shape
da = a_channel.sum()/(h*w)-128
db = b_channel.sum()/(h*w)-128
histA = [0]*256
histB = [0]*256
for i in range(h):
    for j in range(w):
        ta = a_channel[i][j]
        tb = b_channel[i][j]
        histA[ta] += 1
        histB[tb] += 1
msqA = 0
msqB = 0
for y in range(256):
    msqA += float(abs(y-128-da))*histA[y]/(w*h)
    msqB += float(abs(y - 128 - db)) * histB[y] / (w * h)
import math
result = math.sqrt(da*da+db*db)/math.sqrt(msqA*msqA+msqB*msqB)
if result > 4:
    print("图片存在显著色偏")
else:
    print("图片色彩正常")

print("d/m = %s"%result)

#统计数据
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 图像文件夹路径（请替换为实际路径）
image_dir = "Attachment\Attachment 1"  # 替换为附件1图像文件夹路径
images = os.listdir(image_dir)

# 初始化特征存储
channel_differences = []  # 偏色特征：RGB通道均值差值
brightness_values = []    # 低光特征：灰度均值
laplacian_values = []     # 模糊特征：Laplacian梯度强度

# 遍历图像文件
for img_name in images:
    # 读取图像
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    
    # 检查是否成功读取
    if img is None:
        print(f"Failed to load image: {img_name}")
        continue

    # 计算偏色特征
    mean_r, mean_g, mean_b = np.mean(img[:, :, 2]), np.mean(img[:, :, 1]), np.mean(img[:, :, 0])
    diff_rg = abs(mean_r - mean_g) / max(mean_r, mean_g, 1e-5)
    diff_gb = abs(mean_g - mean_b) / max(mean_g, mean_b, 1e-5)
    diff_rb = abs(mean_r - mean_b) / max(mean_r, mean_b, 1e-5)
    channel_differences.append(max(diff_rg, diff_gb, diff_rb))  # 记录最大通道差值比例

    # 计算低光特征
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    mean_brightness = np.mean(gray)
    brightness_values.append(mean_brightness)  # 记录灰度均值

    # 计算模糊特征
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # Laplacian方差
    laplacian_values.append(laplacian_var)  # 记录梯度强度

# 分布可视化
def plot_distribution(data, title, xlabel, ylabel="Frequency", bins=30):
    """绘制分布直方图"""
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=bins, color="blue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

# 偏色特征分布
plot_distribution(channel_differences, 
                  title="Color Difference Distribution", 
                  xlabel="Max Channel Difference Ratio")

# 低光特征分布
plot_distribution(brightness_values, 
                  title="Brightness Distribution", 
                  xlabel="Average Brightness")

# 模糊特征分布
plot_distribution(laplacian_values, 
                  title="Laplacian Variance Distribution", 
                  xlabel="Laplacian Variance")

#写入
import cv2
import os
import math
import pandas as pd
from imutils import paths
import numpy as np

# PSNR计算
def psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:  # 避免除零
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# UCIQE计算
def uciqe(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = np.std(chroma)
    mu_c = np.mean(chroma)
    mu_s = np.mean(l)
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return c1 * sigma_c + c2 * mu_c + c3 * mu_s

# UIQM计算
def uiqm(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    laplacian_var = cv2.Laplacian(l_channel, cv2.CV_64F).var()
    contrast = np.max(l_channel) - np.min(l_channel)
    color_mean = np.mean(a_channel) + np.mean(b_channel)
    return 0.5 * laplacian_var + 0.3 * contrast + 0.2 * color_mean

# 检测是否为低光图片
def is_low_light(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r, c = gray_img.shape[:2]
    dark_sum = sum(sum(1 for colum in row if colum < 40) for row in gray_img)
    dark_prop = dark_sum / (r * c)
    return dark_prop >= 0.6

# 检测是否存在显著色偏
def has_color_bias(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    h, w = img.shape[:2]
    da = a_channel.sum() / (h * w) - 128
    db = b_channel.sum() / (h * w) - 128

    histA = [0] * 256
    histB = [0] * 256
    for i in range(h):
        for j in range(w):
            histA[a_channel[i][j]] += 1
            histB[b_channel[i][j]] += 1

    msqA = sum(float(abs(y - 128 - da)) * histA[y] / (w * h) for y in range(256))
    msqB = sum(float(abs(y - 128 - db)) * histB[y] / (w * h) for y in range(256))
    result = math.sqrt(da**2 + db**2) / math.sqrt(msqA**2 + msqB**2)

    return result > 3.5

# 检测是否为模糊图片
def is_blurry(img, threshold=250):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

# 图片分类处理并写入Excel
def classify_and_save_to_excel(folder_path, excel_path):
    results = []
    for image_path in paths.list_images(folder_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            continue

        # 分类信息
        classification = "Normal"
        if is_low_light(image):
            classification = "Low Light"
        elif has_color_bias(image):
            classification = "Color Bias"
        elif is_blurry(image):
            classification = "Blurry"

        # 计算PSNR、UCIQE、UIQM
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        psnr_value = psnr(gray_image, gray_image)  # 原始图像和自身比较（仅示例）
        uciqe_value = uciqe(image)
        uiqm_value = uiqm(image)

        # 保存结果
        results.append({
            "image file name": os.path.basename(image_path),
            "Degraded Image Classification": classification,
            "PSNR": psnr_value,
            "UCIQE": uciqe_value,
            "UIQM": uiqm_value
        })

    # 保存到Excel
    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)
    print(f"分类结果已保存到 {excel_path}")

# 主程序
if __name__ == "__main__":
    folder_path = "Attachment/Attachment 1"  # 替换为你的图片文件夹路径
    excel_path = "image_classification_results.xlsx"  # 输出的Excel文件路径
    classify_and_save_to_excel(folder_path, excel_path)

