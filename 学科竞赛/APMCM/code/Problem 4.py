#bilateralFilter

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn
from skimage import color
import copy


def bilateral_filter(data, edge=None, edge_min=None, edge_max=None,
                     sigma_spatial=None, sigma_range=None,
                     sampling_spatial=None, sampling_range=None):
    """
    使用双边网格应用双边滤波或交叉双边滤波。

    参数:
    ----------
    data : numpy.ndarray
        要过滤的输入灰度图像。必须是双精度浮点数的2D数组。

    edge : numpy.ndarray, 可选
        引导滤波过程的边缘图像。如果为 None 或为空，则使用 `data` 作为边缘图像。
        必须是与 `data` 大小相同的双精度浮点数2D数组。

    edge_min : float, 可选
        边缘图像中的最小值。默认为 `edge` 数组的最小值。

    edge_max : float, 可选
        边缘图像中的最大值。默认为 `edge` 数组的最大值。

    sigma_spatial : float, 可选
        空间域高斯核的标准差。默认为 `min(width, height) / 16`。

    sigma_range : float, 可选
        范围域高斯核的标准差。默认为 `(edge_max - edge_min) / 10`。

    sampling_spatial : float, 可选
        空间域的下采样因子。默认为 `sigma_spatial`。

    sampling_range : float, 可选
        范围域的下采样因子。默认为 `sigma_range`。

    返回:
    -------
    output : numpy.ndarray
        经过滤波的图像，与 `data` 的尺寸相同。

    抛出异常:
    ------
    ValueError:
        如果输入数据不是2D数组或不是双精度浮点类型。
        如果 `data` 和 `edge` 尺寸不相同。

    说明:
    -----
    此函数通过将输入图像数据映射到基于空间和强度信息的3D网格，应用高斯滤波，
    然后将结果插值回原始图像分辨率，来执行双边滤波。
    """

    # 确保输入数据是2D双精度数组
    if data.ndim > 2:
        raise ValueError("data 必须是大小为 [height, width] 的灰度图像")
    if not isinstance(data, np.ndarray) or data.dtype != np.float64:
        raise ValueError("data 必须是双精度浮点数类型的数组")

    # 如果未提供边缘图像，则使用 data
    if edge is None:
        edge = data
    elif edge.size == 0:
        edge = data

    # 确保边缘图像是2D双精度数组
    if edge.ndim > 2:
        raise ValueError("edge 必须是大小为 [height, width] 的灰度图像")
    if not isinstance(edge, np.ndarray) or edge.dtype != np.float64:
        raise ValueError("edge 必须是双精度浮点数类型的数组")

    input_height, input_width = data.shape

    # 如果未提供 edge_min 和 edge_max，则自动计算
    if edge_min is None:
        edge_min = np.min(edge)
    if edge_max is None:
        edge_max = np.max(edge)

    edge_delta = edge_max - edge_min

    # 如果未提供空间域的高斯标准差 sigma_spatial 和范围域的高斯标准差  sigma_range，则设置默认值
    if sigma_spatial is None:
        sigma_spatial = min(input_width, input_height) / 16
    if sigma_range is None:
        sigma_range = 0.1 * edge_delta

    # 如果未提供空间域的采样因子 sampling_spatial 和范围域的采样因子 sampling_range，则设置默认值
    if sampling_spatial is None:
        sampling_spatial = sigma_spatial
    if sampling_range is None:
        sampling_range = sigma_range

    # 确保 data 和 edge 图像具有相同的尺寸
    if data.shape != edge.shape:
        raise ValueError("data 和 edge 必须具有相同的尺寸")

    # 计算用于下采样的派生参数
    derived_sigma_spatial = sigma_spatial / sampling_spatial
    derived_sigma_range = sigma_range / sampling_range

    # 设置网格的填充
    padding_xy = int(2 * derived_sigma_spatial) + 1
    padding_z = int(2 * derived_sigma_range) + 1

    # 分配用于数据和权重的3D网格
    downsampled_width = int((input_width - 1) / sampling_spatial) + 1 + 2 * padding_xy
    downsampled_height = int((input_height - 1) / sampling_spatial) + 1 + 2 * padding_xy
    downsampled_depth = int(edge_delta / sampling_range) + 1 + 2 * padding_z

    grid_data = np.zeros((downsampled_height, downsampled_width, downsampled_depth))
    grid_weights = np.zeros((downsampled_height, downsampled_width, downsampled_depth))

    # 计算空间和范围维度的下采样索引
    jj, ii = np.meshgrid(np.arange(input_width), np.arange(input_height))
    di = np.round(ii / sampling_spatial) + padding_xy + 1
    dj = np.round(jj / sampling_spatial) + padding_xy + 1
    dz = np.round((edge - edge_min).astype(int) / sampling_range) + padding_z + 1

    # 执行散射操作（可能有更快的方法可以实现）
    # 通常会使用 downsampled_weights[di, dj, dk] = 1，但在这里我们需要
    # 执行一个求和操作，以实现盒式下采样
    # 边界检查并填充数据和权重到3D网格中
    for k in range(data.size):
        data_z = data.flat[k]
        if not np.isnan(data_z):
            # 将索引显式转换为整数类型
            i = int(di.flat[k])
            j = int(dj.flat[k])
            z = int(dz.flat[k])

            # 检查索引是否在有效范围内
            if (0 <= i < downsampled_height and
                    0 <= j < downsampled_width and
                    0 <= z < downsampled_depth):
                grid_data[i, j, z] += data_z
                grid_weights[i, j, z] += 1

    # 应用高斯滤波来平滑网格
    blurred_grid_data = gaussian_filter(grid_data,
                                        sigma=[derived_sigma_spatial, derived_sigma_spatial, derived_sigma_range])
    blurred_grid_weights = gaussian_filter(grid_weights,
                                           sigma=[derived_sigma_spatial, derived_sigma_spatial, derived_sigma_range])

    # 使用权重归一化网格
    blurred_grid_weights[blurred_grid_weights == 0] = -2  # 避免除以零
    normalized_blurred_grid = blurred_grid_data / blurred_grid_weights
    normalized_blurred_grid[blurred_grid_weights < -1] = 0  # 将未定义区域设置为0

    # 插值回原始分辨率
    di = ii / sampling_spatial + padding_xy + 1
    dj = jj / sampling_spatial + padding_xy + 1
    dz = (edge - edge_min) / sampling_range + padding_z + 1

    # 插值上采样网格以获得最终输出图像
    output = interpn((np.arange(normalized_blurred_grid.shape[0]),
                      np.arange(normalized_blurred_grid.shape[1]),
                      np.arange(normalized_blurred_grid.shape[2])),
                     normalized_blurred_grid, np.stack([di, dj, dz], axis=-1),
                     method='linear', bounds_error=False, fill_value=0)
    # 线性映射
    output = (output - output.min()) / (output.max() - output.min()) * 100
    output = output.astype(np.uint8)

    return output


if __name__ == '__main__':
    # 读取输入图像
    image_bgr = cv2.imread('image/273.jpg')

    # 将 BGR 图像转换为 RGB
    image_rbg = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 归一化 RGB 图像（将像素值缩放到 [0, 1]）
    image_rbg_normalized = image_rbg / 255.0

    # 将 RGB 图像转换为 Lab 颜色空间
    lab2 = color.rgb2lab(image_rbg_normalized)

    # 深拷贝 Lab 图像到 temp
    temp = copy.deepcopy(lab2)

    # 对 Lab 图像的亮度通道 (L) 应用自定义双边滤波器
    lab2[:, :, 0] = np.uint8(bilateral_filter(np.double(lab2[:, :, 0])))  # 手动实现的双边滤波器

    # 对 temp 图像的亮度通道 (L) 使用 OpenCV 的双边滤波器
    temp[:, :, 0] = cv2.bilateralFilter(temp[:, :, 0].astype(np.float32), d=6, sigmaColor=5, sigmaSpace=5)  # 调用库函数

    # 将滤波后的 Lab 图像转换回 RGB 颜色空间
    display_image = color.lab2rgb(lab2)

    # 转换为 8 位图像（像素值范围为 [0, 255]）
    display_image = (display_image * 255).astype(np.uint8)

    # 将 RGB 图像转换回 BGR（用于 OpenCV 显示）
    display_brg_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

    # 同样处理 temp 图像
    temp_image = color.lab2rgb(temp)
    temp_image = (temp_image * 255).astype(np.uint8)
    temp_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2BGR)

    # 使用 OpenCV 显示原始图像、自动滤波后图像和手动滤波后图像
    cv2.imshow("auto", temp_image)  # OpenCV 的双边滤波结果
    cv2.imshow("before", image_bgr)  # 原始图像
    cv2.imshow("after", display_brg_image)  # 自定义双边滤波结果

    # 等待用户按下任意键
    cv2.waitKey(0)

# fusion

import cv2
import numpy as np


# 创建高斯滤波器掩码
def filterMask(img):
    """
    创建高斯滤波器掩码，用于生成高斯金字塔。

    参数:
    - img: 输入图像，用于确定数据类型。

    返回:
    - mask: 生成的高斯滤波掩码。
    """
    h = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0]
    mask = np.zeros((len(h), len(h)), dtype=img.dtype)
    for i in range(len(h)):
        for j in range(len(h)):
            mask[i, j] = h[i] * h[j]
    return mask

# 构建高斯金字塔
def build_gaussian_pyramid(img, levels):
    """
    构建高斯金字塔，用于多分辨率图像处理。

    参数:
    - img: 输入图像。
    - levels: 金字塔层数。

    返回:
    - gaussPyr: 高斯金字塔的每一层图像列表。
    """
    gaussPyr = [img.copy()]  # 第一层为原始图像
    current_img = img.copy()

    for i in range(1, levels):
        # 先进行下采样，再进行高斯滤波
        current_img = cv2.pyrDown(current_img)  # 使用 pyrDown 进行降采样
        gaussPyr.append(current_img.copy())  # 将当前图像添加到金字塔中

    return gaussPyr


# 构建拉普拉斯金字塔
def build_laplacian_pyramid(img, levels):
    """
    构建拉普拉斯金字塔，通过高斯金字塔的差异生成。

    参数:
    - img: 输入图像。
    - levels: 金字塔的层数。

    返回:
    - lapPyr: 包含拉普拉斯金字塔各层的列表。
    """
    gaussPyr = build_gaussian_pyramid(img, levels)
    lapPyr = []

    for i in range(levels - 1):
        # 计算当前层与下层之间的差值，得到拉普拉斯层
        size = (gaussPyr[i].shape[1], gaussPyr[i].shape[0])
        upsampled = cv2.pyrUp(gaussPyr[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussPyr[i], upsampled)
        lapPyr.append(laplacian)

    lapPyr.append(gaussPyr[-1])  # 最顶层为高斯金字塔的最高层
    return lapPyr


# 拉普拉斯金字塔重建图像
def reconstructLaplacianPyramid(pyramid):
    """
    从拉普拉斯金字塔重建图像。

    参数:
    - pyramid: 拉普拉斯金字塔的各层图像列表。

    返回:
    - 重建后的图像。
    """
    reconstructed_img = pyramid[-1]  # 从最顶层开始重建
    for i in range(len(pyramid) - 2, -1, -1):
        upsampled = cv2.pyrUp(reconstructed_img, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        reconstructed_img = cv2.add(pyramid[i], upsampled)
    return reconstructed_img


# 融合两幅图像
def fuseTwoImages(img1, img2, w1, w2, level):
    """
    使用多分辨率方法融合两幅图像。

    参数:
    - w1: 第一幅图像的权重图。
    - img1: 第一幅图像。
    - w2: 第二幅图像的权重图。
    - img2: 第二幅图像。
    - level: 金字塔的层数。

    返回:
    - 融合后的图像。
    """
    # 构建权重的高斯金字塔
    weight1 = build_gaussian_pyramid(w1.astype(np.float32), level)
    weight2 = build_gaussian_pyramid(w2.astype(np.float32), level)

    # 将图像转换为浮点类型
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # 分离 RGB 通道
    bCnl1, gCnl1, rCnl1 = cv2.split(img1)
    bCnl2, gCnl2, rCnl2 = cv2.split(img2)

    # 构建各通道的拉普拉斯金字塔
    bPyr1 = build_laplacian_pyramid(bCnl1, level)
    gPyr1 = build_laplacian_pyramid(gCnl1, level)
    rPyr1 = build_laplacian_pyramid(rCnl1, level)

    bPyr2 = build_laplacian_pyramid(bCnl2, level)
    gPyr2 = build_laplacian_pyramid(gCnl2, level)
    rPyr2 = build_laplacian_pyramid(rCnl2, level)

    # 对各层的拉普拉斯金字塔进行融合
    fused_bPyr = [
        cv2.add(
            cv2.multiply(bPyr1[i], weight1[i], dtype=cv2.CV_32F),
            cv2.multiply(bPyr2[i], weight2[i], dtype=cv2.CV_32F)
        ) for i in range(level)
    ]
    fused_gPyr = [
        cv2.add(
            cv2.multiply(gPyr1[i], weight1[i], dtype=cv2.CV_32F),
            cv2.multiply(gPyr2[i], weight2[i], dtype=cv2.CV_32F)
        ) for i in range(level)
    ]
    fused_rPyr = [
        cv2.add(
            cv2.multiply(rPyr1[i], weight1[i], dtype=cv2.CV_32F),
            cv2.multiply(rPyr2[i], weight2[i], dtype=cv2.CV_32F)
        ) for i in range(level)
    ]

    # 重建融合后的图像通道
    bChannel = reconstructLaplacianPyramid(fused_bPyr)
    gChannel = reconstructLaplacianPyramid(fused_gPyr)
    rChannel = reconstructLaplacianPyramid(fused_rPyr)

    # 合并通道
    fusion = cv2.merge((bChannel, gChannel, rChannel))

    # 线性映射到 [0, 255] 范围
    min_val, max_val = fusion.min(), fusion.max()
    fusion = 255 * (fusion - min_val) / (max_val - min_val)
    return fusion.astype(np.uint8)

    # # 将结果裁剪到 0-255 范围并转换为 uint8 类型
    # return np.clip(fusion, 0, 255).astype(np.uint8)

# 主程序：图像融合的示例调用
if __name__ == "__main__":
    # 读取两个待融合的图像
    img1 = cv2.imread('../balanced_image.jpg')
    img2 = cv2.imread('../balanced_image.jpg')

    # 计算权重（这里假设 W1 和 W2 已经计算好，简单设置为全 1 的权重图）
    W1 = np.ones(img1.shape[:2])
    W2 = np.ones(img2.shape[:2])

    # 执行图像融合
    fused_image = fuseTwoImages(img1, img2, W1, W2, 5)

    # 显示融合后的图像结果
    cv2.imshow('Fused Image', fused_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
