"""
evaluate.py — 评估与结果保存模块

功能描述:
    提供 SLAM 系统运行结果的保存功能:
    1. 保存相机轨迹（TUM 格式: timestamp tx ty tz qx qy qz qw）
    2. 保存 3D 重建点云（PLY 格式）
    3. 保存关键帧图像
    
    这些输出可以用于后续的定量评估（如 ATE, RPE 等指标）。
"""

import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
from mast3r_fusion.dataloader import Intrinsics
from mast3r_fusion.frame import SharedKeyframes
from mast3r_fusion.lietorch_utils import as_SE3
from mast3r_fusion.config import config
from mast3r_fusion.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement


def prepare_savedir(args, dataset):
    """
    准备结果保存目录。
    
    目录结构: logs/{save_as}/
    
    参数:
        args: argparse.Namespace — 命令行参数（包含 save_as 字段）
        dataset: Dataset — 数据集对象（用于获取序列名称）
        
    返回:
        save_dir: Path — 保存目录路径
        seq_name: str — 数据集序列名称
    """
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem  # 数据集文件夹名作为序列名
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    """
    保存相机轨迹到文件（TUM 格式）。
    
    格式: timestamp tx ty tz qx qy qz qw
    
    如果提供了内参信息，会用标定参数修正位姿。
    
    参数:
        logdir: str/Path — 保存目录
        logfile: str — 文件名
        timestamps: array — 帧时间戳列表
        frames: SharedKeyframes — 关键帧缓冲区
        intrinsics: Intrinsics — 可选，相机内参（用于位姿修正）
    """
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]  # 获取该关键帧的时间戳
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)  # 从 Sim3 提取 SE3
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)  # 标定修正
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    """
    保存 3D 重建结果为 PLY 点云文件。
    
    流程:
    1. 遍历所有关键帧
    2. 将每帧的 3D 点从相机坐标系变换到世界坐标系
    3. 按置信度阈值过滤低质量点
    4. 合并所有关键帧的点云并保存
    
    参数:
        savedir: str/Path — 保存目录
        filename: str — 文件名（.ply 格式）
        keyframes: SharedKeyframes — 关键帧缓冲区
        c_conf_threshold: float — 置信度阈值（低于此值的点被过滤）
    """
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        # 如果使用标定，需要将点约束到相机射线上
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        # 将相机坐标系的点变换到世界坐标系
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        # 获取对应的颜色
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        # 按置信度过滤
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_ply(savedir / filename, pointclouds, colors)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    """
    保存所有关键帧的图像文件。
    
    文件名使用时间戳命名，格式为 PNG。
    
    参数:
        savedir: str/Path — 保存目录
        timestamps: array — 帧时间戳列表
        keyframes: SharedKeyframes — 关键帧缓冲区
    """
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )


def save_ply(filename, points, colors):
    """
    将点云保存为 PLY 格式文件。
    
    PLY 格式包含每个顶点的 (x, y, z, red, green, blue) 属性。
    
    参数:
        filename: str/Path — 输出文件路径
        points: ndarray, shape (N, 3) — 3D 坐标
        colors: ndarray, shape (N, 3) — RGB 颜色值 (0-255)
    """
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    # 创建结构化数组，包含坐标和颜色信息
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),     # x 坐标 (float32)
            ("y", "f4"),     # y 坐标 (float32)
            ("z", "f4"),     # z 坐标 (float32)
            ("red", "u1"),   # 红色通道 (uint8)
            ("green", "u1"), # 绿色通道 (uint8)
            ("blue", "u1"),  # 蓝色通道 (uint8)
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)
