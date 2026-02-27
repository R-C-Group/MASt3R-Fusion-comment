"""
nonlinear_optimizer.py — 非线性优化辅助工具

功能描述:
    提供非线性优化过程中常用的工具函数:
    1. 收敛判断 — 基于相对误差和步长大小判断优化是否收敛
    2. 鲁棒核函数 — Huber 和 Tukey 鲁棒核，用于抑制异常值的影响

    在 SLAM 系统中，优化过程经常遇到异常匹配（外点），
    使用鲁棒核函数可以降低外点对优化结果的影响。
"""

import math
import torch


def check_convergence(
    iter,
    rel_error_threshold,
    delta_norm_threshold,
    old_cost,
    new_cost,
    delta,
    verbose=False,
):
    """
    检查优化是否已收敛。
    
    收敛条件（满足任一即可）:
    1. 相对代价下降量 < rel_error_threshold（代价变化很小）
    2. 参数更新步长 < delta_norm_threshold（参数几乎不变了）
    
    参数:
        iter: int — 当前迭代次数
        rel_error_threshold: float — 相对误差收敛阈值
        delta_norm_threshold: float — 步长收敛阈值
        old_cost: float — 上一次迭代的代价
        new_cost: float — 当前迭代的代价
        delta: Tensor — 参数更新向量
        verbose: bool — 是否打印详细信息
        
    返回:
        converged: bool — 是否已收敛
    """
    cost_diff = old_cost - new_cost                   # 代价下降量
    rel_dec = math.fabs(cost_diff / old_cost)          # 相对下降比例
    delta_norm = torch.linalg.norm(delta)              # 更新步长的范数

    converged = rel_dec < rel_error_threshold or delta_norm < delta_norm_threshold
    if verbose:
        print(
            f"{iter=} | {new_cost=} {cost_diff=} {rel_dec=} {delta_norm=} | {converged=}"
        )

    # print(f"{iter=} | {new_cost=} {cost_diff=} {rel_dec=} {delta_norm=} | {converged=}")
    return converged


def huber(r, k=1.345):
    """
    Huber 鲁棒核函数 — 计算残差权重。
    
    Huber 核在残差较小时等价于 L2 范数，在残差较大时等价于 L1 范数，
    从而对异常值（outlier）具有鲁棒性。
    
    权重公式:
        w(r) = 1,        if |r| < k
        w(r) = k / |r|,  if |r| >= k
    
    参数:
        r: Tensor — 残差值
        k: float — 阈值参数（默认 1.345，对应 95% 效率）
        
    返回:
        w: Tensor — 每个残差对应的权重（0~1 之间）
    """
    unit = torch.ones((1), dtype=r.dtype, device=r.device)
    r_abs = torch.abs(r)
    mask = r_abs < k
    w = torch.where(mask, unit, k / r_abs)
    return w


def tukey(r, t=4.6851):
    """
    Tukey 双权鲁棒核函数 — 计算残差权重。
    
    Tukey 核比 Huber 核更加激进：当残差超过阈值 t 时，
    权重直接变为 0（完全忽略该残差）。
    
    权重公式:
        w(r) = (1 - (r/t)²)²,  if |r| < t
        w(r) = 0,               if |r| >= t
    
    参数:
        r: Tensor — 残差值
        t: float — 阈值参数（默认 4.6851，对应 95% 效率）
        
    返回:
        w: Tensor — 每个残差对应的权重
    """
    zero = torch.tensor(0.0, dtype=r.dtype, device=r.device)
    r_abs = torch.abs(r)
    tmp = 1 - torch.square(r_abs / t)
    tmp2 = tmp * tmp
    w = torch.where(r_abs < t, tmp2, zero)
    return w
