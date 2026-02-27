"""
geometry.py — 几何运算工具模块

功能描述:
    提供 SLAM 系统中常用的 3D 几何运算函数，包括:
    - 反对称矩阵生成
    - 3D 点到射线分解（方向+深度）
    - 射线约束（反投影到相机光线上）
    - Sim3 变换及其雅可比矩阵
    - 标定投影（3D→像素坐标 + 对数深度）
    - 反投影（像素坐标 + 深度 → 3D 点）
    - 像素坐标网格生成
    
    这些函数是位姿估计和优化的基础构建块。
"""

import torch
import torch.nn.functional as F


def skew_sym(x):
    """
    计算 3D 向量的反对称矩阵（叉积矩阵）。
    
    给定向量 x = [a, b, c]，返回:
        [x]× = | 0  -c   b |
                | c   0  -a |
                |-b   a   0 |
    
    用途: 叉积运算 x × y = [x]× · y
    
    参数:
        x: Tensor, shape (..., 3)
    返回:
        Tensor, shape (..., 3, 3) — 反对称矩阵
    """
    return torch.stack(
        [
            torch.zeros_like(x[..., 0]),   x[..., 2],                  -x[..., 1],
            -x[..., 2],                    torch.zeros_like(x[..., 0]),  x[..., 0],
            x[..., 1],                    -x[..., 0],                   torch.zeros_like(x[..., 0]),
        ],
        dim=-1,
    ).reshape(x.shape[:-1] + (3, 3))


def point_to_dist(X):
    """
    计算 3D 点到原点的欧氏距离。
    
    参数:
        X: Tensor, shape (..., 3) — 3D 坐标
    返回:
        d: Tensor, shape (..., 1) — 到原点的距离
    """
    d = torch.sqrt(torch.sum(X**2, dim=-1, keepdim=True))
    return d


def point_to_ray_dist(X):
    """
    将 3D 点分解为射线方向和深度（沿射线方向的距离）。
    
    同时计算分解过程中的雅可比矩阵 dP/dX，用于优化中的梯度传播。
    
    数学原理:
        对于点 X = [x, y, z]：
        - 射线距离 d = ||X|| = sqrt(x² + y² + z²)
        - 射线方向 r = X / d（单位向量）
        
        我们将 X 分解为 (r, d)，并计算 d(r,d)/dX 的雅可比矩阵。
        
    参数:
        X: Tensor, shape (B, N, 3) — 3D 点坐标
        
    返回:
        rd: Tensor, shape (B, N, 4) — [射线方向(3), 深度(1)]
        drd_dP: Tensor, shape (B, N, 4, 3) — 雅可比矩阵
    """
    d = torch.sqrt(torch.sum(X**2, dim=-1, keepdim=True))  # 距离: ||X||
    r = X / d  # 单位射线方向: X / ||X||

    # 计算雅可比: d(r)/dX
    # r = X/d, dr/dX = (I - r*r^T) / d
    eye = torch.eye(3, device=X.device).expand_as(skew_sym(X))
    rrt = torch.einsum("...i,...j->...ij", r, r)  # r * r^T (外积)
    dr_dP = (eye - rrt) / d[..., None]  # 射线方向对坐标的雅可比

    # d(d)/dX = r^T（距离对坐标的雅可比就是射线方向本身）
    dd_dP = r[..., None, :]

    rd = torch.cat([r, d], dim=-1)               # [方向(3), 深度(1)]
    drd_dP = torch.cat([dr_dP, dd_dP], dim=-2)   # 完整雅可比 [4×3]

    return rd, drd_dP


def constrain_points_to_ray(img_size, Xs, K):
    """
    将 3D 点约束到相机射线上。
    
    当使用已标定相机时，MASt3R 预测的 3D 点可能不完全在像素对应的射线上。
    此函数将点投影回对应像素的射线方向，只保留深度信息。
    
    数学原理:
        对于像素 (u, v)，相机射线方向为:
            r = [(u-cx)/fx, (v-cy)/fy, 1]
        将 MASt3R 预测的深度 z 乘以规范化的射线方向:
            X_constrained = z * r / ||r||
    
    参数:
        img_size: (H, W) — 图像尺寸
        Xs: Tensor, shape (B, H*W, 3) — MASt3R 预测的 3D 点
        K: Tensor, shape (3, 3) — 相机内参矩阵
        
    返回:
        X_new: Tensor, shape (B, H*W, 3) — 约束到射线上的 3D 点
    """
    device = Xs.device
    dtype = Xs.dtype
    # 生成像素坐标网格
    p = get_pixel_coords(Xs.shape[0], img_size, device=device, dtype=dtype).view(
        *Xs.shape[:-1], 2
    )
    # 反投影到归一化平面方向
    dP_dz_x = (p[..., 0] - K[0, 2]) / K[0, 0]
    dP_dz_y = (p[..., 1] - K[1, 2]) / K[1, 1]
    dP_dz = torch.stack([dP_dz_x, dP_dz_y, torch.ones_like(dP_dz_x)], dim=-1)

    # 用预测的深度（z 分量）乘以射线方向
    X_new = Xs[..., 2:3] * dP_dz
    return X_new


def act_Sim3(X, pC, return_J=True):
    """
    Sim3 变换作用于 3D 点集，并可选地返回雅可比矩阵。
    
    Sim3 变换: P_world = s * R * P_camera + t
    
    参数:
        X: lietorch.Sim3 — 世界到相机的 Sim3 变换
        pC: Tensor, shape (B, N, 3) — 相机坐标系下的 3D 点
        return_J: bool — 是否返回雅可比矩阵
        
    返回:
        pW: Tensor, shape (B, N, 3) — 世界坐标系下的 3D 点
        J: Tensor, shape (B, N, 3, 7) — 3D 点对 Sim3 参数的雅可比（可选）
    """
    pW = X.act(pC)  # 应用 Sim3 变换

    if return_J:
        s = X.data[..., -1:, None]  # 尺度因子 s
        R = X.matrix()[..., :3, :3]  # 旋转矩阵 R
        rpC = torch.einsum("...ij,...nj->...ni", R, pC)  # R * pC 旋转后的点

        # 雅可比矩阵 J = d(pW)/d([t, θ, s])，其中 θ 是旋转向量
        # d(pW)/dt = I (平移的雅可比)
        J_t = torch.eye(3, device=pW.device, dtype=pW.dtype).expand(
            *pW.shape, 3
        )  # (B, N, 3, 3)

        # d(pW)/dθ = -s * [R*pC]×  (旋转的雅可比，使用叉积矩阵)
        J_rot = -s * skew_sym(rpC)  # (B, N, 3, 3)

        # d(pW)/ds = R * pC  (尺度的雅可比)
        J_s = rpC.unsqueeze(-1)  # (B, N, 3, 1)

        # 完整雅可比: [平移(3), 旋转(3), 尺度(1)]
        J = torch.cat([J_t, J_rot, J_s], dim=-1)  # (B, N, 3, 7)
        return pW, J

    return pW


def project_calib(P, K, img_size):
    """
    标定投影: 将 3D 点投影到像素坐标 + 对数深度。
    
    同时计算投影过程的雅可比矩阵 dp/dP。
    
    投影模型:
        u = fx * (X/Z) + cx
        v = fy * (Y/Z) + cy
        d = log(Z)
        
    返回的是 (u, v, log_z)，其中 log_z 是对数深度，
    用对数深度而非直接深度可以更好地处理远距离点。
    
    参数:
        P: Tensor, shape (..., 3) — 相机坐标系下的 3D 点 [X, Y, Z]
        K: Tensor, shape (3, 3) — 相机内参矩阵
        img_size: (H, W) — 图像尺寸
        
    返回:
        p: Tensor, shape (..., 3) — 投影结果 [u, v, log(Z)]
        valid: Tensor, shape (...,) — 有效性掩码（深度大于 depth_eps 的点）
        dp_dP: Tensor, shape (..., 3, 3) — 投影的雅可比矩阵
    """
    C_thresh = 1.5  # 置信度阈值（此处未使用）

    # 针孔投影: u = fx·X/Z + cx, v = fy·Y/Z + cy
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = P[..., 0]
    y = P[..., 1]
    z = P[..., 2] + 1e-8  # 加小量避免除零

    depth_eps = 1e-1  # 最小有效深度阈值

    u = fx * (x / z) + cx       # 像素 x 坐标
    v = fy * (y / z) + cy       # 像素 y 坐标
    log_z = torch.log(z)        # 对数深度

    p = torch.stack([u, v, log_z], dim=-1)
    valid = z > depth_eps  # 只有深度大于阈值的点有效

    # 计算投影的雅可比矩阵 dp/dP
    # du/dX = fx/Z, du/dY = 0, du/dZ = -fx*X/Z²
    # dv/dX = 0, dv/dY = fy/Z, dv/dZ = -fy*Y/Z²
    # d(logZ)/dX = 0, d(logZ)/dY = 0, d(logZ)/dZ = 1/Z
    dp_dP = torch.stack(
        [
            fx / z,
            torch.zeros_like(z),
            -fx * x / (z * z),
            torch.zeros_like(z),
            fy / z,
            -fy * y / (z * z),
            torch.zeros_like(z),
            torch.zeros_like(z),
            1.0 / z,
        ],
        dim=-1,
    ).reshape(*P.shape[:-1], 3, 3)

    return p, valid, dp_dP


def backproject(p, z, K):
    """
    反投影: 从像素坐标和深度恢复 3D 点。
    
    数学原理:
        X = (u - cx) / fx * Z
        Y = (v - cy) / fy * Z
        
    参数:
        p: Tensor, shape (..., 2) — 像素坐标 [u, v]
        z: Tensor, shape (..., 1) — 深度值 Z
        K: Tensor, shape (3, 3) — 相机内参矩阵
        
    返回:
        P: Tensor, shape (..., 3) — 3D 点坐标 [X, Y, Z]
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (p[..., 0:1] - cx) / fx * z
    y = (p[..., 1:2] - cy) / fy * z
    return torch.cat([x, y, z], dim=-1)


def get_pixel_coords(b, img_size, device, dtype):
    """
    生成图像像素坐标网格。
    
    对于 H×W 的图像，生成每个像素中心点的坐标 (u, v)，
    其中 u ∈ [0.5, W-0.5]，v ∈ [0.5, H-0.5]。
    
    参数:
        b: int — 批次大小
        img_size: (H, W) — 图像尺寸
        device: 设备
        dtype: 数据类型
        
    返回:
        p: Tensor, shape (b, H*W, 2) — 像素坐标
    """
    h, w = img_size
    uv = torch.meshgrid(
        torch.arange(w, device=device, dtype=dtype),
        torch.arange(h, device=device, dtype=dtype),
        indexing="xy",
    )
    # 加 0.5 使坐标位于像素中心
    p = torch.stack(uv, dim=-1).reshape(1, -1, 2).expand(b, -1, -1) + 0.5
    return p
