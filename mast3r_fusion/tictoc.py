"""
tictoc.py — CUDA 同步计时器

功能描述:
    提供简单易用的 GPU 计时工具，通过 CUDA 事件实现精确的 GPU 操作计时。
    
    使用方式:
        from mast3r_fusion.tictoc import tic, toc
        tic()                    # 开始计时
        # ... GPU 操作 ...
        elapsed = toc("推理")    # 停止计时并打印耗时
    
    注意: 支持嵌套调用（基于栈的 LIFO 结构），
    内层的 toc 对应最近的 tic。
"""

import torch


class Timer:
    """
    Simple timer which takes forces cuda synchronization.
    基于 CUDA 事件的同步计时器。
    
    使用 CUDA 事件而非 time.time() 的原因:
    - CUDA 操作是异步的，time.time() 无法准确测量 GPU 耗时
    - CUDA 事件会在 GPU 流水线中精确标记时间点
    - 调用 toc 时会强制 CUDA 同步，确保测量准确
    """

    def __init__(self):
        self.timers_start = []  # 计时起点的栈（LIFO 结构，支持嵌套）

    def start(self):
        """
        记录一个计时起点（入栈）。
        """
        start_t = torch.cuda.Event(enable_timing=True)
        start_t.record()
        self.timers_start.append(start_t)

    def stop(self, tag=None):
        """
        停止计时并返回耗时（秒）。
        
        参数:
            tag: str — 可选的标签，用于标识打印的计时信息
            
        返回:
            elapsed_time_s: float — 耗时（秒）
        """
        end_t = torch.cuda.Event(enable_timing=True)
        end_t.record()
        torch.cuda.synchronize()  # 强制等待 GPU 操作完成
        start_t = self.timers_start.pop()  # 从栈中弹出最近的起点
        tag = f"{tag}: " if tag else ""
        elapsed_time_s = start_t.elapsed_time(end_t) / 1000  # 毫秒 → 秒
        print(f"{tag}Elapsed {elapsed_time_s}s")
        return elapsed_time_s


# 全局计时器实例，提供便捷的 tic/toc 函数
_global_timer = Timer()
tic = _global_timer.start   # tic() — 开始计时
toc = _global_timer.stop     # toc("标签") — 停止计时
