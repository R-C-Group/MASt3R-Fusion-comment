"""
multiprocess_utils.py — 多进程通信工具

功能描述:
    提供多进程间消息传递的辅助工具:
    1. try_get_msg() — 非阻塞地从队列获取消息
    2. FakeQueue — 模拟队列（当不需要可视化时使用，避免空指针异常）
    3. new_queue() — 工厂函数，根据需求创建真实或虚拟队列

    在 MASt3R-Fusion 中，主进程和可视化进程通过消息队列通信。
    当使用 --no-viz 参数禁用可视化时，使用 FakeQueue 替代真实队列。
"""

import queue


def try_get_msg(q):
    """
    尝试从消息队列中非阻塞地获取一条消息。
    
    如果队列为空，返回 None 而不是阻塞等待。
    主要用于主循环中检查可视化进程的控制消息（暂停/终止等）。
    
    参数:
        q: Queue — 消息队列（真实或虚拟）
        
    返回:
        msg: 消息对象 或 None（队列为空时）
    """
    try:
        msg = q.get_nowait()
    except queue.Empty:
        msg = None
    return msg


class FakeQueue:
    """
    虚拟消息队列 — 当禁用可视化时使用。
    
    实现了与 multiprocessing.Queue 相同的接口，
    但所有操作都是空操作:
    - put: 丢弃消息
    - get_nowait: 总是抛出 Empty 异常
    - qsize: 总是返回 0
    - empty: 总是返回 True
    """
    def put(self, arg):
        """丢弃消息（空操作）"""
        del arg

    def get_nowait(self):
        """总是抛出 Empty 异常"""
        raise queue.Empty

    def qsize(self):
        """总是返回 0"""
        return 0

    def empty(self):
        """总是返回 True"""
        return True


def new_queue(manager, use_fake=False):
    """
    创建消息队列的工厂函数。
    
    参数:
        manager: multiprocessing.Manager — 进程管理器
        use_fake: bool — 是否使用虚拟队列
                         True: 禁用可视化时，返回 FakeQueue
                         False: 启用可视化时，返回真实的共享队列
        
    返回:
        Queue 实例（FakeQueue 或 manager.Queue）
    """
    if use_fake:
        return FakeQueue()
    return manager.Queue()
