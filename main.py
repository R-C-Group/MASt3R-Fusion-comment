"""
main.py — MASt3R-Fusion 在线视觉-惯性 SLAM 主程序

这个文件负责把“前端跟踪、关键帧管理、后端优化、结果落盘、可视化”
五条链路真正串起来。阅读本文件时，建议始终带着两个问题：

1. 当前这一帧处于什么模式？
   - INIT: 用单目推理初始化第一张关键帧。
   - TRACKING: 估计当前帧相对最新关键帧的位姿，并决定是否升格为新关键帧。
   - RELOC: 跟踪失败后的保守恢复模式，先重新生成点图，再等待后端介入。

2. 当前这一帧的数据会流到哪里？
   图像 -> MASt3R 编码/解码 -> Frame/Keyframe -> FactorGraph -> 轨迹文件 / H5 / graph.pkl

因此，`main.py` 既不是纯前端，也不是纯后端，而是系统状态机和数据调度中心。
它本身不实现复杂几何，但决定了哪些模块在什么时机被调用，以及结果何时回写。
"""

import argparse          # 命令行参数解析
import datetime          # 日期时间工具（用于日志命名）
import pathlib           # 路径操作
import sys               # 系统退出
import time              # 计时
import cv2               # OpenCV 图像处理
import lietorch          # Lie 群数学库（Sim3, SE3 等）
import torch             # PyTorch 深度学习框架
import tqdm              # 进度条
import yaml              # YAML 配置文件解析

# 导入因子图优化模块
from mast3r_fusion.global_opt import FactorGraph

# 导入配置管理
from mast3r_fusion.config import load_config, config, set_global_config

# 导入数据集加载器
from mast3r_fusion.dataloader import Intrinsics, load_dataset

# 导入评估工具
import mast3r_fusion.evaluate as eval

# 导入帧管理模块（运行模式、共享关键帧、共享状态、帧创建）
from mast3r_fusion.frame import Mode, SharedKeyframes, SharedStates, create_frame

# 导入 MASt3R 相关工具（模型加载、检索器加载、单目推理）
from mast3r_fusion.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)

# 导入多进程通信工具
from mast3r_fusion.multiprocess_utils import new_queue, try_get_msg

# 导入帧跟踪器
from mast3r_fusion.tracker import FrameTracker

# 导入可视化模块
from mast3r_fusion.visualization import WindowMsg, run_visualization

import torch.multiprocessing as mp   # PyTorch 多进程
import numpy as np                   # 数值计算
from scipy.spatial.transform import Rotation  # 旋转表示转换

import pickle   # 序列化（保存因子图）
import io        # 字节流（用于 HDF5 序列化）
import h5py      # HDF5 文件读写


def find_valid_numbers(a, b):
    """
    从检索结果列表 b 中筛选出对帧 a 有效的回环/共视候选帧索引。
    
    过滤规则:
    1. 排除与当前帧 a 过于接近的帧（差值 ≤ 1）
    2. 对于一组临近的帧（相邻 20 帧以内），只保留最早的那一个（去重）
    3. 特别保留 a-2（即前第二帧），保证连续性约束
    
    参数:
        a: 当前帧索引（为当前关键帧在全局关键帧编号上的索引）
        b: 检索返回的候选帧索引列表（为检索器返回的候选帧索引列表，顺序通常按相似度）
        
    返回:
        result: 过滤后的有效候选帧索引列表
    """
    # 这里不是做“最优候选”搜索，而是把检索结果压缩成少量代表帧：
    # 在线后端只需要少数高价值边，就能兼顾实时性和局部几何约束质量。
    result = []
    for i, c in enumerate(b):
        if abs(c - a) <= 1: # 如果数字过于接近（若是当前帧），跳过
            continue 
        close_indices = [j for j, d in enumerate(b) if abs(d - c) <= 20] # 找出所有与c接近的数字，正负20范围内
        if i == min(close_indices) or c == a - 2 :
            result.append(c)
    return result


def run_backend(states, keyframes):#应该是MASt3R-Fusion相比起MASt3R SLAM最关键的差异。
    """
    运行一次后端优化迭代。
    
    流程:
    1. 检查当前模式，若在初始化或暂停状态则跳过
    2. 从全局优化任务队列中取出一个待处理的关键帧索引
    3. 构建因子图:
       a. 添加与前一帧的连续约束（n_consec=1）
       b. 使用图像检索数据库查找共视帧
       c. 过滤检索结果（find_valid_numbers + 距离<20帧）
       d. 将所有视觉边添加到因子图
    4. 调用 solve_GN_calib 进行 LM 优化求解
    5. 处理 V-I 初始化信号（首次 V-I 初始化完成后更新状态）
    6. 从任务队列中移除已处理的任务
    
    参数:
        states: SharedStates 共享状态对象
        keyframes: SharedKeyframes 共享关键帧缓冲区
    """
    mode = states.get_mode() #读当前模式 

    # 如果还在初始化或已暂停，直接返回，不执行后段优化
    if mode == Mode.INIT or states.is_paused():
        return

    # 从全局优化任务队列中取一个关键帧索引
    idx = -1
    with states.lock:
        if len(states.global_optimizer_tasks) > 0:
            idx = states.global_optimizer_tasks[0] #得到本次要处理的全局关键帧索引 idx（若队列非空）。
    if idx == -1:
        return

    # ===========================
    # 因子图边的构建：本轮整段逻辑都围绕上面取到的 idx：为它构图、优化。
    # ===========================
    # 当前待优化关键帧 `idx` 会连向两类历史帧：
    # 1. 连续帧，保证局部里程计链不断；
    # 2. 局部共视帧，补充视觉几何约束。
    kf_idx = []  # 需要与当前关键帧建立约束的关键帧索引列表

    # k to previous consecutive keyframes
    # 添加与前 n_consec 个连续关键帧的约束
    n_consec = 1  # 只连接前一个关键帧
    for j in range(min(n_consec, idx)):
        kf_idx.append(idx - 1 - j)
    frame = keyframes[idx]  # 获取当前关键帧

    # find local(!) co-visible frames
    # 使用图像检索数据库查找局部共视帧
    retrieval_inds = retrieval_database.update(
        frame,
        add_after_query=True,  # 查询后将当前帧添加到数据库
        k=config["retrieval"]["k"],            # 检索 Top-K 个候选
        min_thresh=config["retrieval"]["min_thresh"],  # 最低相似度阈值
    )

    # 过滤检索结果
    retrieval_inds_selected = []
    retrieval_inds = find_valid_numbers(idx, retrieval_inds)  # 去重和过滤

    # 在线阶段这里只保留“局部”检索边。
    # 真正跨时域的远距离回环会留到 `main_loop.py` 离线处理，
    # 避免主线程因大量长程候选而失去实时性。
    for kkk in retrieval_inds:
        if np.fabs(idx - kkk) < 20:
            retrieval_inds_selected.append(kkk)
    kf_idx += retrieval_inds_selected

    # 检查是否有非连续的回环候选（用于日志打印）
    lc_inds = set(retrieval_inds)
    lc_inds.discard(idx - 1)  # 排除前一帧（连续帧不算回环）
    if len(lc_inds) > 0:
        print("Database retrieval", idx, ": ", lc_inds)

    kf_idx = set(kf_idx)  # Remove duplicates by using set
    kf_idx.discard(idx)  # Remove current kf idx if included
    kf_idx = list(kf_idx)  # convert to list

    # `frame_idx` 与 `kf_idx` 一一对应，组成若干条 (历史关键帧, 当前关键帧) 约束边。
    frame_idx = [idx] * len(kf_idx)
    
    # ===========================
    # 添加因子到因子图
    # ===========================
    print('[INFO] add factor',time.time())
    if kf_idx:
        # 进行 MASt3R 对称匹配并添加视觉因子
        factor_graph.add_factors(
            kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
        )
    print('[INFO] add factor.',time.time())

    # 共享给可视化进程的只是图拓扑，不是全部优化变量。
    # 这样能展示局部连接关系，又不会让跨进程通信变得过重。
    with states.lock:
        states.edges_ii[:] = factor_graph.ii.cpu().tolist()
        states.edges_jj[:] = factor_graph.jj.cpu().tolist()

    # ===========================
    # Gauss-Newton + LM 优化求解，前面添加了因子，此处做联合优化
    # ===========================
    factor_graph.solve_GN_calib(config["use_calib"]) #执行求解时会有VI初始化
    
    # the fisrt time that VI init is finished
    # transform current states
    # 首次 V-I 初始化完成后，状态空间会从“纯视觉 Sim3”切换到
    # “视觉 + IMU 联合估计”。这里再做一次优化，是为了让重力、
    # 速度、偏置和外参约束稳定传播到窗口内所有关键帧。
    if factor_graph.init_vi_signal: #VI初始化完成标志
        factor_graph.solve_GN_calib(config["use_calib"])  # 再优化一次
        factor_graph.init_vi_signal = False  # 清除信号
        states.T_WC[:] = factor_graph.frames.last_keyframe().T_WC[:].data

        # 初始化完成后需要把已有关键帧整体重写一次。
        # 原因是 VI 对齐会整体改变尺度、姿态基准甚至部分平移。
        for i in range(int(keyframes.n_size.value)):
            frame_id = keyframes.dataset_idx[i].item()
            dd = keyframes.T_WC[i].data.cpu().numpy()[0]    # 位姿: [tx,ty,tz,qx,qy,qz,qw,s]
            bb = factor_graph.bs[i].vector()                 # IMU 偏置: [ba_x,ba_y,ba_z,bg_x,bg_y,bg_z]
            # 输出格式: 时间戳 位姿(8) 偏置(6) 帧ID 关键帧标志(1)
            factor_graph.fp.writelines('%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %d 1\n' % (factor_graph.poses_stamps[frame_id],
                                                                                 dd[0].item(),
                                                                                 dd[1].item(),
                                                                                 dd[2].item(),
                                                                                 dd[3].item(),
                                                                                 dd[4].item(),
                                                                                 dd[5].item(),
                                                                                 dd[6].item(),
                                                                                 dd[7].item(),
                                                                                 bb[0],bb[1],bb[2],
                                                                                 bb[3],bb[4],bb[5],
                                                                                 frame_id))
            factor_graph.fp.flush()


    # 后端任务队列按先进先出串行消费，主线程只负责往里塞新关键帧编号。
    with states.lock:
        if len(states.global_optimizer_tasks) > 0:
            idx = states.global_optimizer_tasks.pop(0)


# ===========================
# 主程序入口
# ===========================
if __name__ == "__main__":
    # ===========================
    # 初始化设置
    # ===========================
    mp.set_start_method("spawn", force=True) #使用spawn 模式以避免死锁和内存冲突
    torch.backends.cuda.matmul.allow_tf32 = True   # 启用 TF32 加速矩阵乘法
    torch.set_grad_enabled(False)                   # 全局关闭梯度计算（推理模式）
    device = "cuda:0" #默认了采用设备号0
    save_frames = False                             # 是否保存原始帧图像
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")  # 当前时间戳（用于日志）

    # ===========================
    # 命令行参数解析
    # ===========================
    # 一系列参数的读入
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")  # 数据集路径
    parser.add_argument("--config", default="config/base.yaml")     # 配置文件路径
    parser.add_argument("--save-as", default="default")             # 结果保存目录名
    parser.add_argument("--no-viz", action="store_true")            # 是否禁用可视化
    parser.add_argument("--calib", default="config/intrinsics_zyx.yaml")  # 相机-IMU 标定文件
    parser.add_argument("--imu_path", default="")                   # IMU 数据文件路径
    parser.add_argument("--imu_dt", type = float, default=-0.0)     # IMU 时间偏移（秒）
    parser.add_argument("--stamp_path", default="")                 # 时间戳文件路径
    parser.add_argument("--result_path", default="result.txt")      # 结果输出文件路径
    parser.add_argument("--start_from", type =  int, default=0)     # 起始帧索引
    parser.add_argument("--end_at", type =  int, default=-1)        # 结束帧索引（-1 表示全部）
    parser.add_argument("--save_h5", action="store_true")           # 是否保存帧数据到 HDF5


    args = parser.parse_args()
    load_config(args.config)  # 加载并合并配置文件


    # 如果需要保存 HDF5 数据（供后续回环检测和全局优化使用）
    if args.save_h5:
        f_h5 = h5py.File('data.h5', "w")
    
    # ===========================
    # 创建多进程通信队列
    # 用 Manager() 托管出来的 manager.Queue()，队列对象通过Manager进程代理，跨进程传消息更稳妥，符合「主进程 + 独立可视化进程」结构
    # ===========================
    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)  # 主进程 → 可视化进程 的消息队列
    viz2main = new_queue(manager, args.no_viz)  # 可视化进程 → 主进程 的消息队列

    # ===========================
    # 加载数据集
    # ===========================
    # `dataset` 不只是图像读取器，还统一封装了时间戳、可选标定、
    # 缩放后图像尺寸等后续模块都会依赖的元数据。
    dataset = load_dataset(args.dataset, args.stamp_path) #按工程封装读序列（图像、时间戳、图像尺寸策略等
    dataset.subsample(config["dataset"]["subsample"], args.start_from, args.end_at)  # 子采样
    h, w = dataset.get_img_shape()[0]  # 获取缩放后的图像尺寸 (高, 宽)
    
    # ===========================
    # 加载相机内参与标定
    # ===========================
    if args.calib and config["use_calib"]:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        # 根据标定文件创建 Intrinsics 对象（含畸变矫正映射表）
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
            False, intrinsics.get("model","pinhole"), intrinsics.get("scale",1), intrinsics.get("height_new",None)
        )
    # 如果标定文件指定了新的图像高度（Mei 全景模型裁剪用）
    if not (intrinsics.get("height_new",None) is None): #若 YAML 里 height_new 非空，按比例改 h（全景等裁剪高度）。
        h = intrinsics.get("height_new",None) * w // intrinsics["width"]

    # ===========================
    # 创建共享数据结构
    # ===========================
    # 这两个对象是多进程系统的“共享内存骨架”：
    # - `SharedKeyframes` 保存关键帧滑窗
    # - `SharedStates` 保存当前帧与系统模式
    keyframes = SharedKeyframes(manager, h, w)   # 跨进程共享的关键帧缓冲区（与frame.py 里 rollup_sum 配合做滑窗。）
    states = SharedStates(manager, h, w)         # 跨进程共享的系统状态
    # 可视化进程与主进程通过这两个share_memory_()实现跨进程共享。

    # ===========================
    # 启动可视化进程（可选）
    # ===========================
    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()

    # ===========================
    # 加载 MASt3R 模型
    # ===========================
    model = load_mast3r(device=device)
    model.share_memory()  # 模型参数共享内存，支持多进程访问

    # ===========================
    # 标定（标定参数）检查
    # ===========================
    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)

    # 获取相机内参矩阵 K（3×3）
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)  # 设置关键帧的共享内参

    # ===========================
    # 准备结果保存目录
    # ===========================
    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"\

        if traj_file.exists():
            traj_file.unlink()  # 删除上一次的轨迹文件
        if recon_file.exists():
            recon_file.unlink()  # 删除上一次的重建文件

    # ===========================
    # 创建核心组件
    # ===========================
    tracker = FrameTracker(model, keyframes, device)  # 帧跟踪器 （前端 当前帧 ↔ 最后关键帧 匹配与 Sim3 优化）
    last_msg = WindowMsg()                             # 上一次可视化窗口消息

    # `FactorGraph` 名字上像“图结构”，实际上它同时维护视觉边、IMU 状态、
    # 当前滑窗变量、边缘化先验以及导出到离线阶段所需的图因子。
    factor_graph = FactorGraph(model, keyframes, K, device, args) #后端因子图、IMU、边缘化、轨迹文件 fp、frames_to_save 等；args 里常有 IMU 路径等。
    factor_graph.poses_stamps = dataset.timestamps  # 设置时间戳映射
    
    # 创建图像检索数据库
    retrieval_database = load_retriever(model)

    # ===========================
    # 主循环 — 逐帧处理
    # ===========================
    i = 0                      # 当前帧索引，图像索引
    fps_timer = time.time()    # FPS 计时器

    frames = []                # 可选的原始帧缓存

    while True:
        mode = states.get_mode() # 获取当前系统模式（INIT/TRACKING/RELOC）

        # 检查可视化窗口消息（暂停/终止）
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated: # 如果收到终止信号，跳出循环结束程序
            states.set_mode(Mode.TERMINATED)
            break

        # 处理暂停状态
        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        # 检查是否已处理完所有帧
        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        # ===========================
        # 读取当前帧图像
        # ===========================
        timestamp, img = dataset[i]#根据索引读取图像和对应时间戳
        # time.sleep(0.2)
        if save_frames:
            frames.append(img)

        # ===========================
        # 构建初始位姿
        # ===========================
        # 定义 IMU 到相机的初始相对旋转（绕 x 轴旋转 -90°）
        # 这是一个从 IMU 坐标系到相机坐标系的变换:
        # IMU: x前 y右 z下  →  相机: x右 y下 z前
        TSim3 = lietorch.Sim3.Identity(1, device='cpu')
        Tic0 = np.array([1, 0,  0, 0,
                         0, 0,  1, 0,
                         0,-1,  0, 0,
                         0, 0,  0, 1]).reshape([4,4]) 
        TTTc = Tic0
        qqq = Rotation.from_matrix(TTTc[0:3,0:3]).as_quat()  # 转为四元数 [qx,qy,qz,qw]
        # 填充 Sim3 数据: [tx, ty, tz, qx, qy, qz, qw, s]
        TSim3[0].data[0] = TTTc[0,3]
        TSim3[0].data[1] = TTTc[1,3]
        TSim3[0].data[2] = TTTc[2,3]
        TSim3[0].data[3] = qqq[0]
        TSim3[0].data[4] = qqq[1]
        TSim3[0].data[5] = qqq[2]
        TSim3[0].data[6] = qqq[3]

        # get frames last camera pose
        # 第一帧只能依赖人工指定的相机/IMU 初始对齐；
        # 后续每一帧则沿用上一时刻位姿作为初值，以保证局部优化更容易收敛。
        T_WC = (
            TSim3
            if i == 0
            else states.get_frame().T_WC
        )
        # 创建帧对象:缩放图像、提取图像特征张量等
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        # ===========================
        # 模式分发
        # ===========================
        # 真正重要的不只是三种模式本身，而是每种模式向共享状态和优化队列
        # 写入了什么，从而决定后端下一步会如何解释当前帧。
        if mode == Mode.INIT:
            # ---- 初始化模式 ----
            # Initialize via mono inference, and encoded features neeed for database
            # 使用单目推理（自身配对）获取初始 3D 点云和置信度
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)  # 更新帧的点云
            keyframes.append(frame)                 # 添加为第一个关键帧
            # 队列中存的是“全局关键帧编号”，不是当前共享缓存里的局部位置。
            states.queue_global_optimization(len(keyframes) - 1 + keyframes.rollup_sum.value)  # 排入优化队列
            states.set_mode(Mode.TRACKING)          # 切换到跟踪模式（应该只有第一帧如此）
            states.set_frame(frame)                 # 更新共享状态中的当前帧
            i += 1
            continue

        if mode == Mode.TRACKING: #只有在完成 INIT 之后、且未进入 RELOC 时，才用「当前帧 vs 最新关键帧」做前端跟踪。
            # ---- 跟踪模式 ----
            # 跟踪当前帧到最后一个关键帧的相对位姿
            # 返回:
            # - add_new_kf: 是否需要把当前帧升格为关键帧
            # - match_info: 前端匹配的中间结果（点云/置信度/Q 等列表），主要供可视化/调试（主流程里未再使用）
            # - try_reloc: 前端是否认为当前局部跟踪已不可靠。在 tracker.py 里，仅当「进入优化的有效匹配比例 match_frac 低于 min_match_frac」时，会 return False, [], True，即 try_reloc=True，表示这一帧可匹配几何太少，无法可靠估计位姿，主程序于是 set_mode(Mode.RELOC)，下一帧会走单目重算点图等
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)  # 跟踪丢失，切换到重定位模式
            states.set_frame(frame) #都会把当前帧（含已更新或退化的位姿/点图）写进 SharedStates，供可视化和共享内存读者使用
        elif mode == Mode.RELOC:
            # ---- 重定位模式 ----
            # RELOC 模式先恢复“当前帧自己的点图表达”，
            # 让系统重新拥有一个可参与检索/匹配的视觉载体。
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()  # 排入重定位队列
        else:
            raise Exception("Invalid mode")
        
        # ===========================
        # IMU 辅助关键帧选择
        # ===========================
        # using IMU prediction to adjust keyframe selectiion
        # 纯视觉关键帧选择容易被纹理质量影响，而 IMU 对“真实运动量”更敏感。
        # 因此在 VI 初始化完成后，用 IMU 预测来纠偏关键帧插入判据。
        if factor_graph.enable_ms and frame.frame_id>100: #enable_ms 是一个布尔运行标志，默认 False，只在 视觉–惯性（VI）初始化成功完成之后被设为 True
            dd_old = keyframes.last_keyframe().T_WC.data.cpu().numpy()[0]
            dd_new = states.T_WC[0].data.cpu().numpy()
            # 利用 IMU 预积分预测从上一关键帧到当前帧的位姿变化
            dT, wTc_pred, pred_dt = factor_graph.predict_pose(frame.frame_id)
            if pred_dt > 5.0: # if prediction is too long, just use visual tracking
                pass #do nothing
            else:
                # 如果 IMU 预测旋转较大（>30°）但视觉没有要求新关键帧，则强制插入
                if (not add_new_kf) and  np.linalg.norm(Rotation.from_matrix(dT[0:3,0:3]).as_rotvec())>30.0/57.3:
                    add_new_kf = True
                    tracker.reset_idx_f2k()  # 重置匹配初始化索引
                # 如果 IMU 预测位移很小（<1m）且旋转很小（<5°），则不需要新关键帧
                if add_new_kf and (np.linalg.norm(dT[0:3,3]) < 1.0 and np.linalg.norm(Rotation.from_matrix(dT[0:3,0:3]).as_rotvec())<5.0/57.3):
                    add_new_kf = False
                    tracker.idx_f2k = tracker.idx_f2k_backup  # 恢复匹配索引
    
        # ===========================
        # 插入新关键帧
        # ===========================
        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1 + keyframes.rollup_sum.value)

        # ===========================
        # 运行后端优化
        # ===========================
        print('[INFO] backend',time.time())
        run_backend(states, keyframes)
        
        # ===========================
        # 保存帧数据到 HDF5（可选）
        # ===========================
        print(factor_graph.frames_to_save)
        if args.save_h5:
            for iframe in factor_graph.frames_to_save:
                frame_temp = keyframes[iframe] 
                buffer = io.BytesIO()
                # 这里保存的是“离线阶段能够复原关键帧”的最小充分信息：
                # feat/pos 用于重新解码或匹配，X/C/T_WC 用于恢复点图和初始轨迹。
                torch.save({
                    'feat': frame_temp.feat.cpu(),         # MASt3R 编码器特征
                    'pos': frame_temp.pos.cpu(),           # patch 位置编码
                    'X': frame_temp.X_canon.cpu(),         # 相机坐标系 3D 点
                    'C': frame_temp.C.cpu(),               # 置信度
                    'K': frame_temp.K.cpu(),               # 内参矩阵
                    'N': frame_temp.N,                     # 点云更新次数
                    'uimg': (frame_temp.uimg * 255).to(torch.uint8).cpu().numpy(),  # 原始图像
                    'img_shape': frame_temp.img_shape.cpu(),  # 图像尺寸
                    'T_WC': frame_temp.T_WC.data.cpu(),    # 世界到相机位姿
                    'id': frame_temp.frame_id,             # 帧 ID
                }, buffer)
                buffer.seek(0)
                f_h5.create_dataset(f"frame_{iframe}", data=np.void(buffer.read()))
        factor_graph.frames_to_save = []  # 清空待保存列表



        # ===========================
        # 写入轨迹结果
        # ===========================
        # write results
        dd = states.T_WC[0].data.cpu().numpy() # visual tracking 当前帧的视觉跟踪位姿
        frame_id = frame.frame_id
        try:
            bb = factor_graph.bs[-1].vector()  # 最新的 IMU 偏置
        except:
            bb = np.zeros(6)

        # 在线输出阶段允许用 IMU 预测位姿覆盖视觉跟踪结果，
        # 本质上是在短时尺度上信任惯性传播的平滑性。
        if factor_graph.enable_ms and frame.frame_id>100 and 'wTc_pred' in locals() and pred_dt < 5.0: # IMU prediction
            dd = np.concatenate([wTc_pred[0:3,3],Rotation.from_matrix(wTc_pred[0:3,0:3]).as_quat(),np.array([1.0])])

        # 输出格式: 时间戳 位姿(8) 偏置(6) 帧ID 关键帧标志(0=普通帧)
        factor_graph.fp.writelines('%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %d 0\n' % (factor_graph.poses_stamps[frame_id],
                                                                             dd[0].item(),
                                                                             dd[1].item(),
                                                                             dd[2].item(),
                                                                             dd[3].item(),
                                                                             dd[4].item(),
                                                                             dd[5].item(),
                                                                             dd[6].item(),
                                                                             dd[7].item(),
                                                                             bb[0],bb[1],bb[2],
                                                                             bb[3],bb[4],bb[5],
                                                                             frame_id))
        factor_graph.fp.flush()
        
        # 如果插入了新关键帧，也写入关键帧的优化后位姿
        if add_new_kf:
            dd = keyframes.last_keyframe().T_WC.data.cpu().numpy()[0]
            frame_id = keyframes.last_keyframe().frame_id
            bb = factor_graph.bs[-1].vector()
            # 关键帧标志=1
            factor_graph.fp.writelines('%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %d 1\n' % (factor_graph.poses_stamps[frame_id],
                                                                                 dd[0].item(),
                                                                                 dd[1].item(),
                                                                                 dd[2].item(),
                                                                                 dd[3].item(),
                                                                                 dd[4].item(),
                                                                                 dd[5].item(),
                                                                                 dd[6].item(),
                                                                                 dd[7].item(),
                                                                                 bb[0],bb[1],bb[2],
                                                                                 bb[3],bb[4],bb[5],
                                                                                 frame_id))
            factor_graph.fp.flush()

        # run_backend_orig(states, keyframes)
        print('[INFO] backend.',time.time())


        # ===========================
        # 滑动窗口管理
        # ===========================
        # handling sliding window
        # notice that we main very few frames to save GPU memory usage
        # generally 8 GB is enough
        # 这里只是把共享缓存里的旧关键帧挪出显存，不代表历史信息丢失；
        # 它们对应的约束已经转移到图因子、H5 文件和全局索引偏移里了。
        if len(keyframes) > 30:
            keyframes.roll_up(15)

        # ===========================
        # 性能日志
        # ===========================
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1


    # ===========================
    # 结束处理 — 保存剩余数据
    # ===========================
    # finally 
    # 结束时补存尾部关键帧，避免“还没被边缘化、因此还没写盘”的窗口末尾数据丢失。
    last_pin = factor_graph.get_unique_kf_idx()[-1]
    for iframe in range(factor_graph.last_pin,last_pin+1):
        frame_temp = keyframes[iframe] 
        buffer = io.BytesIO()
        torch.save({
            'feat': frame_temp.feat.cpu(), 
            'pos': frame_temp.pos.cpu(),   
            'X': frame_temp.X_canon.cpu(),
            'C': frame_temp.C.cpu(),
            'K': frame_temp.K.cpu(),
            'N': frame_temp.N,
            'uimg': (frame_temp.uimg * 255).to(torch.uint8).cpu().numpy(),
            'img_shape': frame_temp.img_shape.cpu(),
            'T_WC': frame_temp.T_WC.data.cpu(),
            'id': frame_temp.frame_id,
        }, buffer)
        buffer.seek(0)
        f_h5.create_dataset(f"frame_{iframe}", data=np.void(buffer.read()))

    # 保存完整的因子图（供后续全局优化使用）
    factor_graph.save_graph('graph.pkl')

    # if dataset.save_results:
    #     save_dir, seq_name = eval.prepare_savedir(args, dataset)
    #     eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
    #     eval.save_reconstruction(
    #         save_dir,
    #         f"{seq_name}.ply",
    #         keyframes,
    #         last_msg.C_conf_threshold,
    #     )
    #     eval.save_keyframes(
    #         save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
    #     )
    # if save_frames:
    #     savedir = pathlib.Path(f"logs/frames/{datetime_now}")
    #     savedir.mkdir(exist_ok=True, parents=True)
    #     for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
    #         frame = (frame * 255).clip(0, 255)
    #         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    states.set_mode(Mode.TERMINATED)
    if not args.no_viz:
        viz.join()  # 等待可视化进程退出
