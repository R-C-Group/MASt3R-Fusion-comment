"""
lietorch_utils.py — Lie 群工具函数

功能描述:
    提供 Sim3（相似变换群，7DoF）到 SE3（特殊欧氏群，6DoF）的转换。
    
    Sim3 = 平移(3) + 旋转(3) + 尺度(1) = 7DoF
    SE3  = 平移(3) + 旋转(3) = 6DoF
    
    在可视化和评估中，通常需要 SE3 位姿（不需要尺度信息），
    因此需要从 Sim3 中提取 SE3 分量。
"""

import einops       # 张量重排工具
import lietorch     # Lie 群数学库
import torch


def as_SE3(X):
    """
    从 Sim3 变换中提取 SE3 变换（丢弃尺度分量）。
    
    如果输入已经是 SE3，则直接返回。
    
    Sim3 数据格式: [tx, ty, tz, qx, qy, qz, qw, s]（8个参数）
    SE3 数据格式:  [tx, ty, tz, qx, qy, qz, qw]（7个参数）
    
    参数:
        X: lietorch.Sim3 或 lietorch.SE3 — 输入变换
        
    返回:
        T_WC: lietorch.SE3 — 提取出的 SE3 变换
    """
    if isinstance(X, lietorch.SE3):
        return X  # 已经是 SE3，直接返回
    
    # 从 Sim3 数据中分离: 平移(3), 四元数(4), 尺度(1)
    t, q, s = einops.rearrange(X.data.detach().cpu(), "... c -> (...) c").split(
        [3, 4, 1], -1
    )
    # 只取平移和四元数（丢弃尺度）构造 SE3
    T_WC = lietorch.SE3(torch.cat([t, q], dim=-1))
    return T_WC
