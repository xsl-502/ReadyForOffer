#低光 1

import cv2
import numpy as np

def log_transform(image, c=1.0):
    """
    使用对数变换增强低光图像
    :param image: 输入的灰度图像
    :param c: 调整参数
    :return: 增强后的图像
    """
    # 转换为浮点类型避免溢出
    float_img = np.float32(image) + 1.0  # 加 1 防止 log(0)
    
    # 应用对数变换
    log_image = c * np.log(float_img)
    
    # 归一化到 0-255
    normalized = cv2.normalize(log_image, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(normalized)

# 示例：加载图像并增强
image_path = "Attachment/Attachment 1/image_018.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("无法加载图像，请检查路径！")
else:
    enhanced_image = log_transform(image, c=1.5)

    # 显示原图和增强后的图像
    cv2.imshow("Original Image", image)
    cv2.imshow("Log Transformed Image", enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    cv2.imwrite("log_transformed_image.jpg", enhanced_image)

#低光 2
import cv2

def histogram_equalization_color(image):
    """
    对彩色图像进行直方图均衡化
    :param image: 输入的彩色图像（BGR 格式）
    :return: 均衡化后的彩色图像
    """
    # 转换为 YUV 颜色空间
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # 对亮度通道 (Y 通道) 进行直方图均衡化
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])

    # 转回 BGR 颜色空间
    equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    return equalized_image

# 示例：加载图像并进行直方图均衡化
image_path = "Attachment/Attachment 1/image_018.png"  # 替换为实际图像路径
image = cv2.imread(image_path)

if image is None:
    print("无法加载图像，请检查路径！")
else:
    enhanced_image = histogram_equalization_color(image)

    # 显示原图和增强后的图像
    cv2.imshow("Original Image", image)
    cv2.imshow("Equalized Image", enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    cv2.imwrite("equalized_color_image.jpg", enhanced_image)
    print("彩色图像的直方图均衡化完成并保存。")


#低光 3
import cv2
import numpy as np
import matplotlib.pyplot as plt

def multi_scale_histogram_equalization(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    多尺度直方图均衡化实现，用于提升低光照图像的亮度和对比度
    :param image: 输入图像 (BGR 格式)
    :param clip_limit: CLAHE 的对比度限制阈值，默认 2.0
    :param grid_size: CLAHE 的网格尺寸，默认 (8, 8)
    :return: 增强后的图像
    """
    # 转换为 LAB 色彩空间
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # 对亮度通道 (L 通道) 应用 CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_channel_enhanced = clahe.apply(l_channel)

    # 合并增强后的亮度通道和原始的 A、B 通道
    enhanced_lab_image = cv2.merge((l_channel_enhanced, a_channel, b_channel))
    
    # 转换回 BGR 色彩空间
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

    return enhanced_image

def visualize_images(original, enhanced):
    """
    可视化原始图像和增强后图像
    :param original: 原始图像
    :param enhanced: 增强后的图像
    """
    plt.figure(figsize=(12, 6))

    # 原始图像
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))  # 转换为 RGB 显示
    plt.axis('off')

    # 增强后图像
    plt.subplot(1, 2, 2)
    plt.title("Enhanced Image")
    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))  # 转换为 RGB 显示
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    input_image = cv2.imread("Attachment/Attachment 1/image_018.png")

    if input_image is None:
        print("无法加载图像，请检查路径！")
    else:
        # 对图像进行多尺度直方图均衡化
        enhanced_image = multi_scale_histogram_equalization(input_image)

        # 可视化原始图像和增强后图像
        visualize_images(input_image, enhanced_image)

        # 保存增强后的图像
        output_path = "enhanced_image.jpg"
        cv2.imwrite(output_path, enhanced_image)
        print(f"增强后的图像已保存至 {output_path}")

#计算指标
import cv2
import numpy as np
import pandas as pd
import math
from skimage.color import rgb2lab
from skimage import img_as_float
import matplotlib.pyplot as plt

# 指标计算
def psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:  # 避免除零
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def uciqe(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    chroma = np.sqrt(a**2 + b**2)
    sigma_c = np.std(chroma)
    mu_c = np.mean(chroma)
    mu_s = np.mean(l)
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return c1 * sigma_c + c2 * mu_c + c3 * mu_s

def uiqm(image):
    lab_image = rgb2lab(img_as_float(image))
    l_channel = lab_image[:, :, 0]
    a_channel = lab_image[:, :, 1]
    b_channel = lab_image[:, :, 2]
    laplacian_var = cv2.Laplacian(l_channel, cv2.CV_64F).var()
    contrast = np.max(l_channel) - np.min(l_channel)
    color_mean = np.mean(a_channel) + np.mean(b_channel)
    return 0.5 * laplacian_var + 0.3 * contrast + 0.2 * color_mean

# 低光处理
def multi_scale_histogram_equalization(image, clip_limit=2.0, grid_size=(8, 8)):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_channel_enhanced = clahe.apply(l_channel)
    enhanced_lab_image = cv2.merge((l_channel_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

# 色偏处理
def white_balance(image):
    b, g, r = cv2.split(image)
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)
    mean_gray = (mean_r + mean_g + mean_b) / 3
    r = cv2.addWeighted(r, mean_gray / mean_r, 0, 0, 0)
    g = cv2.addWeighted(g, mean_gray / mean_g, 0, 0, 0)
    b = cv2.addWeighted(b, mean_gray / mean_b, 0, 0, 0)
    return cv2.merge([b, g, r])

# 模糊处理
def wiener_deblur(image, kernel, K=0.01):
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel, s=image.shape)
    kernel_fft_conj = np.conj(kernel_fft)
    deblurred_fft = (image_fft * kernel_fft_conj) / (np.abs(kernel_fft) ** 2 + K)
    deblurred = np.fft.ifft2(deblurred_fft).real
    return cv2.normalize(deblurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 分类处理主函数
def process_and_classify_images(excel_path, output_excel):
    df = pd.read_excel(excel_path)
    results = []

    for idx, row in df.iterrows():
        image_name = row['image file name']
        classification = row['Degraded Image Classification']
        image = cv2.imread(image_name)

        if image is None:
            print(f"无法加载图像: {image_name}")
            continue

        # 根据分类处理图像
        if classification == "Low Light":
            enhanced_image = multi_scale_histogram_equalization(image)
        elif classification == "Color Bias":
            enhanced_image = white_balance(image)
        elif classification == "Blurry":
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getGaussianKernel(5, 1.0) * cv2.getGaussianKernel(5, 1.0).T
            enhanced_image = wiener_deblur(gray_image, kernel)
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
        else:  # 如果是正常图像，跳过处理
            enhanced_image = image

        # 计算指标
        psnr_value = psnr(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY))
        uciqe_value = uciqe(enhanced_image)
        uiqm_value = uiqm(enhanced_image)

        # 保存结果
        results.append({
            "image file name": image_name,
            "Degraded Image Classification": classification,
            "PSNR": psnr_value,
            "UCIQE": uciqe_value,
            "UIQM": uiqm_value
        })

    # 写入到新的 Excel 文件
    result_df = pd.DataFrame(results)
    result_df.to_excel(output_excel, index=False)
    print(f"处理结果已保存至 {output_excel}")

# 主程序
if __name__ == "__main__":
    input_excel = "input_images.xlsx"  # 输入 Excel 文件路径
    output_excel = "output_results.xlsx"  # 输出 Excel 文件路径
    process_and_classify_images(input_excel, output_excel)

#模糊 1

import cv2
import numpy as np
import math
from skimage.color import rgb2lab
from skimage import img_as_float
import matplotlib.pyplot as plt

def psnr(original, enhanced):
    """
    计算 PSNR
    :param original: 原始图像
    :param enhanced: 增强后的图像
    :return: PSNR 值
    """
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:  # 避免除零
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def uciqe(image):
    """
    计算 UCIQE
    :param image: 输入图像 (BGR 格式)
    :return: UCIQE 值
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    chroma = np.sqrt(a**2 + b**2)
    sigma_c = np.std(chroma)
    mu_c = np.mean(chroma)
    mu_s = np.mean(l)
    # 经验权重
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return c1 * sigma_c + c2 * mu_c + c3 * mu_s

def uiqm(image):
    """
    计算 UIQM
    :param image: 输入图像 (BGR 格式)
    :return: UIQM 值
    """
    lab_image = rgb2lab(img_as_float(image))
    l_channel = lab_image[:, :, 0]
    a_channel = lab_image[:, :, 1]
    b_channel = lab_image[:, :, 2]

    laplacian_var = cv2.Laplacian(l_channel, cv2.CV_64F).var()
    contrast = np.max(l_channel) - np.min(l_channel)
    color_mean = np.mean(a_channel) + np.mean(b_channel)
    return 0.5 * laplacian_var + 0.3 * contrast + 0.2 * color_mean

def wiener_deblur(image, kernel, K=0.01):
    """
    使用 Wiener 去卷积进行图像去模糊
    :param image: 模糊图像
    :param kernel: 模糊核
    :param K: 噪声控制参数
    :return: 去模糊后的图像
    """
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel, s=image.shape)

    kernel_fft_conj = np.conj(kernel_fft)
    deblurred_fft = (image_fft * kernel_fft_conj) / (np.abs(kernel_fft) ** 2 + K)
    deblurred = np.fft.ifft2(deblurred_fft).real

    deblurred = cv2.normalize(deblurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return deblurred

if __name__ == "__main__":
    # 加载图像
    image_path = "Attachment/Attachment 1/image_259.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("无法加载图像，请检查路径！")
    else:
        # 创建高斯模糊核
        kernel = cv2.getGaussianKernel(5, 1.0) * cv2.getGaussianKernel(5, 1.0).T

        # 使用 Wiener 去模糊
        deblurred_image = wiener_deblur(image, kernel)

        # 转换去模糊图像为 BGR 以计算 UCIQE 和 UIQM
        deblurred_image_bgr = cv2.cvtColor(deblurred_image, cv2.COLOR_GRAY2BGR)

        # 计算指标
        psnr_value = psnr(image, deblurred_image)
        uciqe_value = uciqe(deblurred_image_bgr)
        uiqm_value = uiqm(deblurred_image_bgr)

        # 输出指标结果
        print(f"PSNR (去模糊后): {psnr_value:.2f}")
        print(f"UCIQE (去模糊后): {uciqe_value:.2f}")
        print(f"UIQM (去模糊后): {uiqm_value:.2f}")

        # 显示原始图像和去模糊后的图像
        cv2.imshow("Original Image", image)
        cv2.imshow("Deblurred Image", deblurred_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存去模糊结果
        output_path = "deblurred_image.jpg"
        cv2.imwrite(output_path, deblurred_image)
        print(f"去模糊后的图像已保存至 {output_path}")

# 模糊 2
import cv2
import numpy as np
import math
from skimage.color import rgb2lab
from skimage import img_as_float
import matplotlib.pyplot as plt

# Unsharp Masking 实现
def unsharp_mask(image, strength=1.5, blur_kernel=(5, 5)):
    """
    使用 Unsharp Masking 进行图像锐化
    :param image: 输入的灰度图像或彩色图像
    :param strength: 锐化强度（默认 1.5）
    :param blur_kernel: 高斯模糊核大小（默认 (5, 5)）
    :return: 锐化后的图像
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    high_freq = cv2.subtract(gray, blurred)
    sharpened = cv2.addWeighted(gray, 1.0, high_freq, strength, 0)

    return sharpened

# PSNR 计算
def psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# UCIQE 计算
def uciqe(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    chroma = np.sqrt(a**2 + b**2)
    sigma_c = np.std(chroma)
    mu_c = np.mean(chroma)
    mu_s = np.mean(l)
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return c1 * sigma_c + c2 * mu_c + c3 * mu_s

# UIQM 计算
def uiqm(image):
    lab_image = rgb2lab(img_as_float(image))
    l_channel = lab_image[:, :, 0]
    a_channel = lab_image[:, :, 1]
    b_channel = lab_image[:, :, 2]

    laplacian_var = cv2.Laplacian(l_channel, cv2.CV_64F).var()
    contrast = np.max(l_channel) - np.min(l_channel)
    color_mean = np.mean(a_channel) + np.mean(b_channel)
    return 0.5 * laplacian_var + 0.3 * contrast + 0.2 * color_mean

# 主程序
if __name__ == "__main__":
    # 加载图像
    image_path = "Attachment/Attachment 1/image_259.png"  # 替换为实际图像路径
    image = cv2.imread(image_path)

    if image is None:
        print("无法加载图像，请检查路径！")
    else:
        # 使用 Unsharp Masking 进行锐化
        sharpened_image = unsharp_mask(image, strength=1.5, blur_kernel=(5, 5))

        # 计算锐化后的图像的指标
        psnr_value = psnr(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), sharpened_image)
        uciqe_value = uciqe(cv2.cvtColor(sharpened_image, cv2.COLOR_GRAY2BGR))  # 转回 BGR 供 UCIQE 使用
        uiqm_value = uiqm(cv2.cvtColor(sharpened_image, cv2.COLOR_GRAY2BGR))  # 转回 BGR 供 UIQM 使用

        # 输出指标结果
        print(f"PSNR (锐化后): {psnr_value:.2f}")
        print(f"UCIQE (锐化后): {uciqe_value:.2f}")
        print(f"UIQM (锐化后): {uiqm_value:.2f}")

        # 显示原始图像和锐化后的图像
        cv2.imshow("Original Image", image)
        cv2.imshow("Sharpened Image", sharpened_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果
        output_path = "sharpened_image.jpg"
        cv2.imwrite(output_path, sharpened_image)
        print(f"锐化后的图像已保存至 {output_path}")

#模糊 3
import cv2
import numpy as np
import math
from skimage.color import rgb2lab
from skimage import img_as_float
import matplotlib.pyplot as plt

def psnr(original, enhanced):
    """
    计算 PSNR
    :param original: 原始图像
    :param enhanced: 增强后的图像
    :return: PSNR 值
    """
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:  # 避免除零
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def uciqe(image):
    """
    计算 UCIQE
    :param image: 输入图像 (BGR 格式)
    :return: UCIQE 值
    """
    # 转换为 LAB 空间
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    chroma = np.sqrt(a**2 + b**2)
    sigma_c = np.std(chroma)
    mu_c = np.mean(chroma)
    mu_s = np.mean(l)
    # 经验权重
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return c1 * sigma_c + c2 * mu_c + c3 * mu_s

def uiqm(image):
    """
    计算 UIQM
    :param image: 输入图像 (BGR 格式)
    :return: UIQM 值
    """
    # 转为 LAB 空间
    lab_image = rgb2lab(img_as_float(image))
    l_channel = lab_image[:, :, 0]
    a_channel = lab_image[:, :, 1]
    b_channel = lab_image[:, :, 2]

    # 清晰度测度 (UISM)
    laplacian_var = cv2.Laplacian(l_channel, cv2.CV_64F).var()
    # 对比度测度 (UIConM)
    contrast = np.max(l_channel) - np.min(l_channel)
    # 颜色测度 (UICM)
    color_mean = np.mean(a_channel) + np.mean(b_channel)
    return 0.5 * laplacian_var + 0.3 * contrast + 0.2 * color_mean

def laplacian_sharpening(image, lambda_factor=1.5):
    """
    拉普拉斯锐化增强
    :param image: 输入图像 (BGR 格式)
    :param lambda_factor: 锐化强度系数
    :return: 原始图像和增强后的图像
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    laplacian = cv2.Laplacian(smoothed_image, cv2.CV_64F)
    sharpened_image = gray_image + lambda_factor * laplacian
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
    return gray_image, cv2.cvtColor(sharpened_image, cv2.COLOR_GRAY2BGR)

def show_images(original, enhanced):
    """
    显示原始图像和增强后图像
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Enhanced Image")
    plt.imshow(enhanced, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 示例用法
if __name__ == "__main__":
    input_image = cv2.imread("Attachment/Attachment 1/image_259.png")
    if input_image is None:
        print("无法加载图像，请检查路径！")
    else:
        # 对图像进行拉普拉斯锐化
        original_image, enhanced_image = laplacian_sharpening(input_image)

        # 计算增强后图像的指标
        psnr_value = psnr(cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY))
        uciqe_value = uciqe(enhanced_image)  # 计算锐化后的 UCIQE
        uiqm_value = uiqm(enhanced_image)  # 计算锐化后的 UIQM

        # 输出指标结果
        print(f"PSNR (锐化后): {psnr_value:.2f}")
        print(f"UCIQE (锐化后): {uciqe_value:.2f}")
        print(f"UIQM (锐化后): {uiqm_value:.2f}")

        # 显示原始图像和增强后的图像
        show_images(cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY))

        # 保存增强后的图像
        output_path = "sharpened_image.jpg"
        cv2.imwrite(output_path, enhanced_image)
        print(f"锐化后的图像已保存至 {output_path}")

#偏色增强

import cv2
import numpy as np

def white_balance(image):
    """
    使用简单的白平衡方法调整图像颜色
    :param image: 输入的 RGB 图像
    :return: 增强后的图像
    """
    # 拆分图像为 R、G、B 三个通道
    (b, g, r) = cv2.split(image)

    # 计算每个通道的均值
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)

    # 计算整体灰度均值（参考值）
    mean_gray = (mean_r + mean_g + mean_b) / 3

    # 调整每个通道的增益
    r = cv2.addWeighted(r, mean_gray / mean_r, 0, 0, 0)
    g = cv2.addWeighted(g, mean_gray / mean_g, 0, 0, 0)
    b = cv2.addWeighted(b, mean_gray / mean_b, 0, 0, 0)

    # 合并调整后的通道
    balanced_image = cv2.merge([b, g, r])

    return balanced_image

# 示例：加载图像并进行白平衡增强
#Attachment\Attachment 1\image_285.jpg
image_path = "Attachment/Attachment 1/image_285.jpg"  # 替换为实际图像路径
image = cv2.imread(image_path)

if image is None:
    print("无法加载图像，请检查路径！")
else:
    balanced_image = white_balance(image)

    # 显示原图和增强图
    cv2.imshow("Original Image", image)
    cv2.imshow("White Balanced Image", balanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    # output_path = "white_balanced_image.jpg"
    # cv2.imwrite(output_path, balanced_image)
    # print(f"增强后的图像已保存至 {output_path}")

#偏色增强 2

import cv2
import numpy as np
import matplotlib.pyplot as plt

def estimate_transmission(image, omega=0.95, patch_size=15):
    """
    透射率估算，基于暗通道假设
    :param image: 输入图像 (BGR 格式)
    :param omega: 调节系数
    :param patch_size: 邻域窗口大小
    :return: 透射率图
    """
    dark_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(dark_channel, kernel)
    transmission = 1 - omega * (dark_channel / 255.0)
    return np.clip(transmission, 0.1, 1)

def estimate_background(image, dark_channel):
    """
    背景光估算
    :param image: 输入图像 (BGR 格式)
    :param dark_channel: 暗通道图
    :return: 背景光强度 (BGR 格式)
    """
    flat_image = image.reshape(-1, 3)
    flat_dark = dark_channel.ravel()
    top_indices = np.argsort(flat_dark)[-int(0.001 * len(flat_dark)):]
    return np.max(flat_image[top_indices], axis=0)

def recover_image(image, transmission, background, t_min=0.1):
    """
    图像恢复
    :param image: 输入图像 (BGR 格式)
    :param transmission: 透射率图
    :param background: 背景光强度 (BGR 格式)
    :param t_min: 最小透射率
    :return: 恢复后的图像
    """
    transmission = np.clip(transmission, t_min, 1)
    # 将单通道扩展为三通道
    transmission = np.expand_dims(transmission, axis=2)
    transmission = np.repeat(transmission, 3, axis=2)

    recovered = (image - background) / transmission + background
    return np.clip(recovered, 0, 255).astype(np.uint8)

def clahe_enhancement(image):
    """
    对比度校正 (CLAHE)
    :param image: 输入图像 (BGR 格式)
    :return: 增强后的图像
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel_enhanced = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_channel_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def color_distortion_correction(image):
    """
    颜色失真增强主函数
    :param image: 输入图像 (BGR 格式)
    :return: 增强后的图像
    """
    transmission = estimate_transmission(image)
    dark_channel = np.min(image, axis=2)
    background = estimate_background(image, dark_channel)
    restored_image = recover_image(image, transmission, background)
    enhanced_image = clahe_enhancement(restored_image)
    return enhanced_image

def visualize_images(original, enhanced):
    """
    可视化原始图像和增强后图像
    :param original: 原始图像
    :param enhanced: 增强后的图像
    """
    plt.figure(figsize=(12, 6))

    # 原始图像
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))  # 转换为 RGB 显示
    plt.axis('off')

    # 增强后图像
    plt.subplot(1, 2, 2)
    plt.title("Enhanced Image")
    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))  # 转换为 RGB 显示
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 示例使用
if __name__ == "__main__":
    input_image = cv2.imread("Attachment/Attachment 1/image_285.jpg")
    
    if input_image is None:
        print("无法加载图像，请检查路径！")
    else:
        # 对图像进行颜色失真校正
        enhanced_image = color_distortion_correction(input_image)

        # 可视化原始图像和增强后图像
        visualize_images(input_image, enhanced_image)

        # 保存增强后的图像
        output_path = "color_enhanced_image.jpg"
        cv2.imwrite(output_path, enhanced_image)
        print(f"增强后的图像已保存至 {output_path}")

