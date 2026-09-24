import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

# 光传输函数计算
def compute_transmission(depth, turbidity, wavelength):
    """
    计算光传输函数 t(x)
    :param depth: 水深 (m)
    :param turbidity: 浊度 (NTU)
    :param wavelength: 光波长 (nm)
    :return: 光传输系数 t(x)
    """
    # 吸收系数 (文献参考值)
    absorption_coefficients = {
        450: 0.015,  # 蓝光
        550: 0.025,  # 绿光
        650: 0.035   # 红光
    }
    
    # 散射系数 (假设与浊度成正比)
    scattering_coefficient = 0.001 * turbidity
    
    # 总衰减系数
    k = absorption_coefficients.get(wavelength, 0.025) + scattering_coefficient
    
    # 计算光传输函数
    return np.exp(-k * depth)

# 水下退化模拟
def simulate_underwater_image(image, depth, turbidity, ambient_light=255):
    """
    模拟水下图像退化 I(x) = J(x) * t(x) + B(t(x))
    :param image: 清晰图像 J(x) (BGR 格式)
    :param depth: 水深 (m)
    :param turbidity: 浊度 (NTU)
    :param ambient_light: 环境光强度 (0-255)
    :return: 退化后的图像 I(x)
    """
    # 分离 RGB 通道
    b, g, r = cv2.split(image.astype(np.float32))
    
    # 计算光传输函数 t(x)
    t_b = compute_transmission(depth, turbidity, 450)  # 蓝光
    t_g = compute_transmission(depth, turbidity, 550)  # 绿光
    t_r = compute_transmission(depth, turbidity, 650)  # 红光
    
    # 模拟直接光衰减
    r = r * t_r
    g = g * t_g
    b = b * t_b
    
    # 引入背景光 B(t(x))
    r += ambient_light * (1 - t_r)
    g += ambient_light * (1 - t_g)
    b += ambient_light * (1 - t_b)
    
    # 合并通道并裁剪到 [0, 255]
    degraded_image = cv2.merge([b, g, r])
    degraded_image = np.clip(degraded_image, 0, 255).astype(np.uint8)
    
    return degraded_image

# 模糊效应模拟
def add_blur(image, blur_strength=5):
    """
    添加模糊效应
    :param image: 输入图像
    :param blur_strength: 模糊强度 (核大小)
    :return: 模糊后的图像
    """
    return cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

# 批量生成退化图像
def generate_simulated_images(input_folder, output_folder, depths, turbidities, ambient_lights):
    """
    根据不同参数生成模拟图像
    :param input_folder: 输入清晰图像文件夹
    :param output_folder: 输出退化图像文件夹
    :param depths: 水深列表
    :param turbidities: 浊度列表
    :param ambient_lights: 环境光强度列表
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for image_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, image_name)
        image = cv2.imread(input_path)
        if image is None:
            print(f"无法读取图像: {image_name}")
            continue
        
        for depth in depths:
            for turbidity in turbidities:
                for ambient_light in ambient_lights:
                    # 模拟退化图像
                    degraded_image = simulate_underwater_image(image, depth, turbidity, ambient_light)
                    degraded_image = add_blur(degraded_image, blur_strength=5)
                    
                    # 保存退化图像
                    output_name = f"{os.path.splitext(image_name)[0]}_d{depth}_t{turbidity}_a{ambient_light}.jpg"
                    output_path = os.path.join(output_folder, output_name)
                    cv2.imwrite(output_path, degraded_image)
                    print(f"已生成: {output_name}")

def visualize_images(input_image_path, output_image):
    """
    可视化输入图像和退化后的输出图像
    :param input_image_path: 输入清晰图像路径
    :param output_image: 输出退化后的图像 (numpy array)
    """
    # 读取输入图像
    input_image = cv2.imread(input_image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

    # 创建并排显示
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Input Image (J(x))")
    plt.imshow(input_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Output Image (I(x))")
    plt.imshow(output_image)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
def save_comparison_image(input_image_path, output_image, save_path):
    """
    将输入图像和退化图像拼接并保存
    :param input_image_path: 输入清晰图像路径
    :param output_image: 输出退化后的图像 (numpy array)
    :param save_path: 拼接图像保存路径
    """
    # 读取输入图像
    input_image = cv2.imread(input_image_path)

    # 调整输入和输出图像尺寸一致
    h, w = input_image.shape[:2]
    output_image_resized = cv2.resize(output_image, (w, h))

    # 拼接图像
    comparison_image = cv2.hconcat([input_image, output_image_resized])

    # 保存拼接图像
    cv2.imwrite(save_path, comparison_image)
    print(f"Comparison image saved at: {save_path}")

if __name__ == "__main__":
    # 输入清晰图像路径
    input_image_path = "Attachment/Attachment 1/image_001.png"  # 替换为实际路径

    # 读取清晰图像
    input_image = cv2.imread(input_image_path)

    # 模拟参数
    depth = 5 # 水深 (m)
    turbidity = 10  # 浊度 (NTU)
    ambient_light = 100  # 环境光强度

    # 生成退化图像
    output_image = simulate_underwater_image(input_image, depth, turbidity, ambient_light)
    output_image = add_blur(output_image, blur_strength=5)

    # 可视化输入和输出
    visualize_images(input_image_path, output_image)

    # # 保存拼接图像
    # save_path = "comparison_image.jpg"  # 替换为保存路径
    # save_comparison_image(input_image_path, output_image, save_path)
