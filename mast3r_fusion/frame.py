"""
frame.py — 帧与关键帧管理模块

功能描述:
    定义了 SLAM 系统中帧（Frame）的数据结构和跨进程共享的关键帧管理。
    
    核心类:
    1. Frame — 单帧数据容器（图像、特征、位姿、3D点云、置信度等）
    2. SharedKeyframes — 跨进程共享的关键帧滑动窗口缓冲区
    3. SharedStates — 跨进程共享的系统状态（模式、当前帧、优化队列等）
    4. Mode — 系统运行模式枚举（INIT/初始化, TRACKING/跟踪, RELOC/重定位, TERMINATED/终止）
    
    关键设计:
    - 使用 torch.Tensor.share_memory_() 实现 GPU 数据的进程间共享
    - 滑动窗口（roll_up）机制控制内存使用
    - 支持多种点云更新策略（first, recent, best_score, weighted 等）
"""

import dataclasses
from enum import Enum
from typing import Optional
import lietorch
import torch
from mast3r_fusion.mast3r_utils import resize_img
from mast3r_fusion.config import config


class Mode(Enum):
    """
    系统运行模式枚举。
    """
    INIT = 0 # 初始化模式：处理第一帧，建立初始点云
    TRACKING = 1 # 跟踪模式：正常帧到关键帧的位姿跟踪
    RELOC = 2 # 重定位模式：跟踪丢失后的恢复
    TERMINATED = 3 # 终止：系统已停止运行


@dataclasses.dataclass
class Frame:
    """
    单帧数据容器，存储一帧图像的所有关联信息。
    
    """
    frame_id: int # 数据集中的帧索引
    img: torch.Tensor # 归一化（0~1）图像张量 (1, 3, H, W)
    img_shape: torch.Tensor # 图像尺寸 [H, W]
    img_true_shape: torch.Tensor # 缩放前的原始图像尺寸
    uimg: torch.Tensor # 未归一化的图像（用于可视化，0~1）
    T_WC: lietorch.Sim3 = lietorch.Sim3.Identity(1) # 世界到相机的 Sim3 变换（7DoF: 平移+旋转+尺度）
    X_canon: Optional[torch.Tensor] = None # 相机坐标系下的 3D 点云 (H*W, 3)
    C: Optional[torch.Tensor] = None # 每个 3D 点的置信度
    feat: Optional[torch.Tensor] = None # MASt3R 编码器特征（用于匹配和解码）
    pos: Optional[torch.Tensor] = None # ViT patch 位置编码
    N: int = 0 # 点云更新次数（用于加权平均）
    N_updates: int = 0 # 跟踪更新次数
    K: Optional[torch.Tensor] = None # 相机内参矩阵
    T_CkCf: lietorch.Sim3 = lietorch.Sim3.Identity(1) # 关键帧到当前帧的相对变换（跟踪结果）
    ref_kf: int = 0 # 参考关键帧的索引（用于跟踪和重定位）

    def get_score(self, C):
        """
        计算置信度图的综合得分（用于点云更新决策）。
        
        参数:
            C: Tensor — 置信度图
        返回:
            score: float — 中位数或均值得分
        """
        filtering_score = config["tracking"]["filtering_score"]
        if filtering_score == "median":
            score = torch.median(C)  # Is this slower than mean? Is it worth it?
        elif filtering_score == "mean":
            score = torch.mean(C)
        return score

    def update_pointmap(self, X: torch.Tensor, C: torch.Tensor):
        """
        更新帧的 3D 点云和置信度图。
        
        支持多种融合策略:
        - "first": 只保留首次观测（不更新）
        - "recent": 只保留最新观测
        - "best_score": 如果新观测的整体得分更高，则替换
        - "indep_conf": 逐像素按置信度选择（取更高置信度的）
        - "weighted_pointmap": 按观测次数加权平均 3D 坐标
        - "weighted_spherical": 在球坐标系下加权平均（射线方向+深度）
        
        参数:
            X: Tensor (H*W, 3) — 新的 3D 点云
            C: Tensor (H*W, 1) — 新的置信度图
        """
        filtering_mode = config["tracking"]["filtering_mode"]
        # `Frame` 里的点图并不是一次性定型的，而是允许被后续观测持续修正。
        # 不同 filtering mode 本质上是在做不同的“多次观测融合策略”。

        if self.N == 0:
            # 首次观测，直接赋值
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
            self.N_updates = 1
            if filtering_mode == "best_score":
                self.score = self.get_score(C)
            return

        if filtering_mode == "first":
            # 只保留第一次观测，强调稳定性，不追求后续细化。
            if self.N_updates == 1:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
        elif filtering_mode == "recent":
            # 始终相信最新观测，适合快速响应，但会更容易受瞬时噪声影响。
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
        elif filtering_mode == "best_score":
            # 用全局置信度分数做“一票否决”，而不是逐像素融合。
            new_score = self.get_score(C)
            if new_score > self.score:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
                self.score = new_score
        elif filtering_mode == "indep_conf":
            # 逐像素竞争，谁的置信度更高就保留谁。
            new_mask = C > self.C
            self.X_canon[new_mask.repeat(1, 3)] = X[new_mask.repeat(1, 3)]
            self.C[new_mask] = C[new_mask]
            self.N = 1
        elif filtering_mode == "weighted_pointmap":
            # 直接在欧式坐标下做加权平均，简单但对角度变化较敏感。
            self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
            self.C = self.C + C
            self.N += 1
        elif filtering_mode == "weighted_spherical":
            # 先转球坐标再平均，等价于把“方向”和“距离”拆开融合，
            # 对视线方向变化更友好。

            def cartesian_to_spherical(P):
                r = torch.linalg.norm(P, dim=-1, keepdim=True)
                x, y, z = torch.tensor_split(P, 3, dim=-1)
                phi = torch.atan2(y, x)
                theta = torch.acos(z / r)
                spherical = torch.cat((r, phi, theta), dim=-1)
                return spherical

            def spherical_to_cartesian(spherical):
                r, phi, theta = torch.tensor_split(spherical, 3, dim=-1)
                x = r * torch.sin(theta) * torch.cos(phi)
                y = r * torch.sin(theta) * torch.sin(phi)
                z = r * torch.cos(theta)
                P = torch.cat((x, y, z), dim=-1)
                return P

            spherical1 = cartesian_to_spherical(self.X_canon)
            spherical2 = cartesian_to_spherical(X)
            spherical = ((self.C * spherical1) + (C * spherical2)) / (self.C + C)

            self.X_canon = spherical_to_cartesian(spherical)
            self.C = self.C + C
            self.N += 1

        self.N_updates += 1
        return

    def get_average_conf(self):
        """获取平均置信度（累积置信度除以观测次数）"""
        return self.C / self.N if self.C is not None else None



def create_frame(i, img, T_WC, img_size=512, device="cuda:0"):
    """
    创建帧对象。
    
    流程:
    1. 将图像缩放到 MASt3R 要求的尺寸（默认 512 像素）
    2. 提取归一化图像和原始图像
    3. 可选的降采样（减少点云密度）
    4. 封装成 Frame 数据类
    
    参数:
        i: int — 帧索引
        img: ndarray — 原始图像
        T_WC: Sim3 — 初始位姿估计
        img_size: int — MASt3R 输入图像尺寸
        device: str — 计算设备
    返回:
        Frame 对象
    """
    # MASt3R 对输入分辨率和裁剪方式很敏感，这一步既是缩放，也是统一输入协议。
    img = resize_img(img, img_size)                     # 缩放图像到 MASt3R 所需尺寸
    rgb = img["img"].to(device=device)                   # 归一化图像张量 (1, 3, H, W)
    img_shape = torch.tensor(img["true_shape"], device=device)  # 实际图像尺寸
    img_true_shape = img_shape.clone()
    uimg = torch.from_numpy(img["unnormalized_img"]) / 255.0    # 未归一化图像 (0~1, 用于可视化)
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        # 这里的降采样会影响后续点图密度、匹配规模和显存占用，
        # 是速度/精度/内存三者之间的直接权衡。
        uimg = uimg[::downsample, ::downsample]
        img_shape = img_shape // downsample
    frame = Frame(i, rgb, img_shape, img_true_shape, uimg, T_WC)
    return frame



class SharedStates:
    """
    跨进程共享的系统状态。
    
    在多进程架构中（主进程 + 可视化进程），
    此类维护了两个进程共享的状态信息：
    - 当前运行模式（INIT/TRACKING/RELOC/TERMINATED）
    - 暂停状态
    - 当前帧数据（位姿、图像、点云等）
    - 后端优化任务队列
    - 因子图边信息（用于可视化）
    """
    def __init__(self, manager, h, w, dtype=torch.float32, device="cuda"):
        self.h, self.w = h, w
        self.dtype = dtype
        self.device = device

        self.lock = manager.RLock()                      # 可重入锁（保护共享数据）
        self.paused = manager.Value("i", 0)              # 暂停标志
        self.mode = manager.Value("i", Mode.INIT)        # 当前运行模式
        self.reloc_sem = manager.Value("i", 0)           # 重定位信号量
        self.global_optimizer_tasks = manager.list()      # 后端优化任务队列
        self.edges_ii = manager.list()                    # 因子图边 - 源关键帧索引
        self.edges_jj = manager.list()                    # 因子图边 - 目标关键帧索引

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)

        # fmt:off
        # shared state for the current frame (used for reloc/visualization)
        self.dataset_idx = torch.zeros(1, device=device, dtype=torch.int).share_memory_()
        self.img = torch.zeros(3, h, w, device=device, dtype=dtype).share_memory_()
        self.uimg = torch.zeros(h, w, 3, device="cpu", dtype=dtype).share_memory_()
        self.img_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.T_WC = lietorch.Sim3.Identity(1, device=device, dtype=dtype).data.share_memory_()
        self.X = torch.zeros(h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C = torch.zeros(h * w, 1, device=device, dtype=dtype).share_memory_()
        self.feat = torch.zeros(1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos = torch.zeros(1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        # fmt: on

    def set_frame(self, frame):
        with self.lock:
            # 当前帧会被可视化进程和重定位逻辑反复读取，因此这里做的是
            # “完整快照式覆盖”，而不是增量更新。
            self.dataset_idx[:] = frame.frame_id
            self.img[:] = frame.img
            self.uimg[:] = frame.uimg
            self.img_shape[:] = frame.img_shape
            self.img_true_shape[:] = frame.img_true_shape
            self.T_WC[:] = frame.T_WC.data
            self.X[:] = frame.X_canon
            self.C[:] = frame.C
            self.feat[:] = frame.feat
            self.pos[:] = frame.pos

    def get_frame(self):
        with self.lock:
            # 这里返回的是基于共享内存构造的轻量 `Frame` 视图，
            # 方便上层继续沿用统一的数据结构接口。
            frame = Frame(
                int(self.dataset_idx[0]),
                self.img,
                self.img_shape,
                self.img_true_shape,
                self.uimg,
                lietorch.Sim3(self.T_WC),
            )
            frame.X_canon = self.X
            frame.C = self.C
            frame.feat = self.feat
            frame.pos = self.pos
            return frame

    def queue_global_optimization(self, idx):
        with self.lock:
            self.global_optimizer_tasks.append(idx)

    def queue_reloc(self):
        with self.lock:
            self.reloc_sem.value += 1

    def dequeue_reloc(self):
        with self.lock:
            if self.reloc_sem.value == 0:
                return
            self.reloc_sem.value -= 1

    def get_mode(self):
        with self.lock:
            return self.mode.value

    def set_mode(self, mode):
        with self.lock:
            self.mode.value = mode

    def pause(self):
        with self.lock:
            self.paused.value = 1

    def unpause(self):
        with self.lock:
            self.paused.value = 0

    def is_paused(self):
        with self.lock:
            return self.paused.value == 1

import cv2
class SharedKeyframes:
    """
    跨进程共享的关键帧滑动窗口缓冲区。
    
    核心设计:
    - 使用固定大小的 GPU 张量缓冲区（默认 64 帧）
    - 通过 share_memory_() 实现多进程共享
    - roll_up() 实现滑动窗口：当缓冲满时，丢弃最旧的帧
    - 通过 rollup_sum 记录累计丢弃的帧数，保证全局索引一致性
    
    存储的数据:
    - dataset_idx: 数据集中的帧 ID
    - img: 归一化图像 (B,3,H,W)
    - uimg: 原始图像 (B,H,W,3)
    - T_WC: Sim3 位姿
    - X: 3D 点云 (B,H*W,3)
    - C: 置信度 (B,H*W,1)
    - feat: MASt3R 编码器特征
    - pos: patch 位置编码
    - K: 相机内参矩阵
    """
    def __init__(self, manager, h, w, buffer=64, dtype=torch.float32, device="cuda"):
        self.lock = manager.RLock()                  # 可重入锁（跨进程访问保护）
        self.n_size = manager.Value("i", 0)          # 当前缓冲区中的关键帧数量
        self.rollup_sum = manager.Value("i", 0)      # 累计丢弃的关键帧数（全局偏移）

        self.h, self.w = h, w
        self.buffer = buffer     # 缓冲区大小（最多存储的关键帧数）
        self.dtype = dtype
        self.device = device

        self.feat_dim = 1024                        # MASt3R 特征维度
        self.num_patches = h * w // (16 * 16)       # patch 数量（图像尺寸 / patch 大小²）

        # fmt:off
        # 初始化共享内存张量（跨进程共享 GPU 内存）
        self.dataset_idx = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()           # 数据集帧 ID
        self.img = torch.zeros(buffer, 3, h, w, device=device, dtype=dtype).share_memory_()              # 归一化图像
        self.uimg = torch.zeros(buffer, h, w, 3, device="cpu", dtype=dtype).share_memory_()              # 原始图像（CPU，可视化用）
        self.img_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()       # 图像尺寸
        self.img_true_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()  # 原始图像尺寸
        self.T_WC = torch.zeros(buffer, 1, lietorch.Sim3.embedded_dim, device=device, dtype=dtype).share_memory_()  # Sim3 位姿
        self.X = torch.zeros(buffer, h * w, 3, device=device, dtype=dtype).share_memory_()               # 3D 点云
        self.C = torch.zeros(buffer, h * w, 1, device=device, dtype=dtype).share_memory_()               # 置信度
        self.N = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()                      # 观测次数
        self.N_updates = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()              # 跟踪更新次数
        self.feat = torch.zeros(buffer, 1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()  # 编码器特征
        self.pos = torch.zeros(buffer, 1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()          # patch 位置
        self.is_dirty = torch.zeros(buffer, 1, device=device, dtype=torch.bool).share_memory_()          # 脏标志（标记是否需要可视化重绘）
        self.K = torch.zeros(3, 3, device=device, dtype=dtype).share_memory_()                           # 相机内参

        # fmt: on

    def __getitem__(self, idx) -> Frame:
        """
        通过全局索引获取关键帧。
        
        注意: idx 是全局索引（从整个序列开始计数），
        实际缓冲区中的位置 = idx - rollup_sum。
        """
        with self.lock:
            # print('get:',idx,self.rollup_sum.value)
            # 关键点在于 `idx` 是“全局关键帧编号”，而底层共享缓存是滑窗局部编号；
            # 两者靠 `rollup_sum` 做映射。
            kf = Frame(
                int(self.dataset_idx[idx-self.rollup_sum.value]),
                self.img[idx-self.rollup_sum.value],
                self.img_shape[idx-self.rollup_sum.value],
                self.img_true_shape[idx-self.rollup_sum.value],
                self.uimg[idx-self.rollup_sum.value],
                lietorch.Sim3(self.T_WC[idx-self.rollup_sum.value]),
            )
            kf.X_canon = self.X[idx-self.rollup_sum.value]
            kf.C = self.C[idx-self.rollup_sum.value]
            kf.feat = self.feat[idx-self.rollup_sum.value]
            kf.pos = self.pos[idx-self.rollup_sum.value]
            kf.N = int(self.N[idx-self.rollup_sum.value])
            kf.N_updates = int(self.N_updates[idx-self.rollup_sum.value])
            if config["use_calib"]:
                kf.K = self.K
            return kf

    def __setitem__(self, idx, value: Frame) -> None:
        """将 Frame 对象写入共享缓冲区的指定位置。"""
        with self.lock:
            self.n_size.value = max(idx + 1, self.n_size.value)

            # set the attributes
            # 将 Frame 的所有属性复制到共享内存张量中
            self.dataset_idx[idx] = value.frame_id
            self.img[idx] = value.img
            self.uimg[idx] = value.uimg
            self.img_shape[idx] = value.img_shape
            self.img_true_shape[idx] = value.img_true_shape
            self.T_WC[idx] = value.T_WC.data
            self.X[idx] = value.X_canon
            self.C[idx] = value.C
            self.feat[idx] = value.feat
            self.pos[idx] = value.pos
            self.N[idx] = value.N
            self.N_updates[idx] = value.N_updates
            self.is_dirty[idx] = True       # 标记为脏（需要重新渲染）
            return idx

    def __len__(self):
        with self.lock:
            return self.n_size.value

    def append(self, value: Frame):
        """将新关键帧追加到缓冲区末尾。"""
        with self.lock:
            self[self.n_size.value] = value

    def pop_last(self):
        """弹出最后一个关键帧。"""
        with self.lock:
            self.n_size.value -= 1

    def last_keyframe(self) -> Optional[Frame]:
        """获取最后一个（最新的）关键帧。"""
        with self.lock:
            if self.n_size.value == 0:
                return None
            return self[self.n_size.value - 1 + self.rollup_sum.value]

    def update_T_WCs(self, T_WCs, idx) -> None:
        """更新指定关键帧的位姿（优化后回写）。"""
        with self.lock:
            self.T_WC[idx - self.rollup_sum.value] = T_WCs.data

    def get_dirty_idx(self):
        """获取所有需要重新渲染的关键帧索引（并重置脏标志）。"""
        with self.lock:
            idx = torch.where(self.is_dirty)[0] + self.rollup_sum.value
            self.is_dirty[:] = False
            return idx

    def set_intrinsics(self, K):
        """设置共享的相机内参矩阵。"""
        assert config["use_calib"]
        with self.lock:
            self.K[:] = K

    def get_intrinsics(self):
        """获取共享的相机内参矩阵。"""
        assert config["use_calib"]
        with self.lock:
            return self.K
    
    def roll_up(self, rollup=256):
        """
        滑动窗口: 丢弃最旧的 rollup 个关键帧以释放内存。
        
        实现方式: 将所有张量循环左移 `rollup` 个位置，
        并同步更新 `n_size` 与 `rollup_sum`。

        注意这里丢弃的是“共享缓存中的旧关键帧副本”，不是逻辑上的历史消失：
        图优化、H5 保存、全局编号都会继续保留它们的影响。
        
        参数:
            rollup: int — 要丢弃的帧数（默认 256）
        """
        with self.lock:
            # 循环左移所有共享内存张量
            self.dataset_idx[:]       = torch.roll(self.dataset_idx,       -rollup, dims=0)
            self.img[:]               = torch.roll(self.img,               -rollup, dims=0)
            self.uimg[:]              = torch.roll(self.uimg,              -rollup, dims=0)
            self.img_shape[:]         = torch.roll(self.img_shape,         -rollup, dims=0)
            self.img_true_shape[:]    = torch.roll(self.img_true_shape,    -rollup, dims=0)
            self.T_WC[:]              = torch.roll(self.T_WC,              -rollup, dims=0)
            self.X[:]                 = torch.roll(self.X,                 -rollup, dims=0)
            self.C[:]                 = torch.roll(self.C,                 -rollup, dims=0)
            self.N[:]                 = torch.roll(self.N,                 -rollup, dims=0)
            self.N_updates[:]         = torch.roll(self.N_updates,         -rollup, dims=0)
            self.feat[:]              = torch.roll(self.feat,              -rollup, dims=0)
            self.pos[:]               = torch.roll(self.pos,               -rollup, dims=0)
            self.is_dirty[:]          = torch.roll(self.is_dirty,          -rollup, dims=0)
    
            self.n_size.value -= rollup         # 缓冲区有效帧数减少
            self.rollup_sum.value += rollup     # 累计偏移增加

