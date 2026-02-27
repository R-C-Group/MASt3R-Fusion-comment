"""
image.py — 图像梯度计算模块

功能描述:
    使用 Scharr 算子（一种改良的 Sobel 算子）计算图像在 x 和 y 方向的梯度。
    Scharr 算子相比 Sobel 算子具有更好的旋转不变性。

    在 SLAM 系统中，图像梯度用于:
    - 特征检测和描述
    - 直接法位姿估计中的光度梯度计算
    - 图像质量评估
"""

import torch
import torch.nn.functional as F


def img_gradient(img):
    """
    使用 Scharr 算子计算图像的 x 和 y 方向梯度。
    
    Scharr 卷积核（归一化后）:
        Gx = 1/32 * | -3  0  3 |    Gy = 1/32 * | -3 -10 -3 |
                     |-10  0 10 |                 |  0   0  0 |
                     | -3  0  3 |                 |  3  10  3 |
    
    使用 reflect 模式填充边界像素，避免边界伪影。
    
    参数:
        img: Tensor, shape (B, C, H, W) — 输入图像
             B = 批次大小, C = 通道数, H = 高度, W = 宽度
        
    返回:
        gx: Tensor, shape (B, C, H, W) — x 方向梯度
        gy: Tensor, shape (B, C, H, W) — y 方向梯度
    """
    device = img.device
    dtype = img.dtype
    b, c, h, w = img.shape

    # 定义 x 方向 Scharr 卷积核（水平梯度检测）
    gx_kernel = (1.0 / 32.0) * torch.tensor(
        [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    )
    gx_kernel = gx_kernel.repeat(c, 1, 1, 1)  # 扩展到每个通道

    # 定义 y 方向 Scharr 卷积核（垂直梯度检测）
    gy_kernel = (1.0 / 32.0) * torch.tensor(
        [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    )
    gy_kernel = gy_kernel.repeat(c, 1, 1, 1)

    # 使用反射填充处理边界（避免边界伪影）
    # groups=c 表示每个通道独立卷积（分组卷积/深度可分离卷积）
    gx = F.conv2d(
        F.pad(img, (1, 1, 1, 1), mode="reflect"),
        gx_kernel,
        groups=img.shape[1],
    )

    gy = F.conv2d(
        F.pad(img, (1, 1, 1, 1), mode="reflect"),
        gy_kernel,
        groups=img.shape[1],
    )

    return gx, gy
