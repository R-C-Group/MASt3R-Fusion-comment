import torch
from mast3r_fusion.frame import Frame
from mast3r_fusion.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from mast3r_fusion.nonlinear_optimizer import check_convergence, huber
from mast3r_fusion.config import config
from mast3r_fusion.mast3r_utils import mast3r_match_asymmetric
"""
tracker.py — 帧跟踪器模块

功能描述:
    实现 FrameTracker 类，负责将当前帧注册到最近的关键帧坐标系中。
    这是在线 SLAM 系统的视觉前端核心组件。
    
    跟踪流程:
    1. MASt3R 非对称匹配（当前帧 → 关键帧）
    2. 位姿估计（两种模式）:
       a. 无标定模式: 基于射线方向+深度距离的残差 (opt_pose_ray_dist_sim3)
       b. 有标定模式: 基于像素重投影+对数深度的残差 (opt_pose_calib_sim3)
    3. 关键帧选择: 基于匹配覆盖率决定是否插入新关键帧
    4. 点云更新: 利用新观测更新关键帧的 3D 点云
    
    优化方法:
    - Gauss-Newton 法，通过 Cholesky 分解求解正规方程
    - 支持 Huber 鲁棒核函数抑制外点影响
    
    关键类:
    - FrameTracker: 帧到关键帧的位姿跟踪器
    
    关键函数:
    - track(frame): 跟踪一帧，返回是否需要新关键帧
    - opt_pose_ray_dist_sim3(): 无标定位姿优化（射线距离残差）
    - opt_pose_calib_sim3(): 有标定位姿优化（重投影残差）
"""

import lietorch
import time

class FrameTracker:
    def __init__(self, model, frames, device):
        self.cfg = config["tracking"]
        self.model = model
        self.keyframes = frames
        self.device = device

        self.reset_idx_f2k()
        self.fp = open('tracker.log','wt')

    # `idx_f2k` 保存“当前帧像素 -> 关键帧像素”的上一次匹配索引。
    # 下次跟踪时把它作为初始化，可以显著减少迭代投影匹配的搜索范围。
    def reset_idx_f2k(self):
        try:
            self.idx_f2k_backup = self.idx_f2k
        except:
            pass
        self.idx_f2k = None

    def track(self, frame: Frame):
        keyframe = self.keyframes.last_keyframe()

        # 非对称匹配只解码“当前帧看关键帧”这一方向，
        # 这是在线跟踪场景的速度优先设计：相比对称匹配更省时。
        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
            self.model, frame, keyframe, idx_i2j_init=self.idx_f2k
        )
        # 保存匹配索引用作下一帧的初始化，相当于延续一个局部光流/配准先验。
        self.idx_f2k = idx_f2k.clone()

        # Get rid of batch dim
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]

        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

        # 当前帧先更新自己的点图；后面位姿估计完成后，再把观测反投到关键帧坐标系，
        # 用来增量更新关键帧点图。
        frame.update_pointmap(Xff, Cff)

        use_calib = config["use_calib"]
        img_size = frame.img.shape[-2:]
        if use_calib:
            K = keyframe.K
        else:
            K = None

        # 把后续优化真正需要的量整理出来：
        # - 当前帧与关键帧上的对应 3D 点
        # - 当前的初始位姿
        # - 置信度和可选的标定投影观测
        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
            frame, keyframe, idx_f2k, img_size, use_calib, K
        )

        # 只有在“匹配存在 + 两侧 3D 置信度够高 + 描述子置信度够高”时，
        # 该观测才被允许进入位姿优化。
        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]

        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
        valid_kf = valid_match_k & valid_Q

        match_frac = valid_opt.sum() / valid_opt.numel()
        if match_frac < self.cfg["min_match_frac"]:
            print(f"Skipped frame {frame.frame_id}")
            return False, [], True
        print('[INFO] track',time.time())
        try:
            # 根据是否有标定，切换到两种不同残差形式：
            # - 无标定: 射线方向 + 距离残差
            # - 有标定: 像素重投影 + 对数深度残差
            if not use_calib:
                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
                )
            else:
                T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                    Xf,
                    Xk,
                    T_WCf,
                    T_WCk,
                    Qk,
                    valid_opt,
                    meas_k,
                    valid_meas_k,
                    K,
                    img_size,
                )
            print('[INFO] track.',time.time())
        except Exception as e:
            print(f"Cholesky failed {frame.frame_id}")
            # 一旦正规方程失稳，这里退化成“继承关键帧位姿”而不是直接崩溃。
            # 这样上层可以继续运行，并把是否重定位交给更高层逻辑判断。
            T_WCf = lietorch.Sim3(T_WCk.data.clone())
            T_CkCf = T_WCf.inv() * T_WCk

        frame.T_WC = T_WCf
        frame.ref_kf = keyframe.frame_id
        frame.T_CkCf = T_CkCf
        dd = keyframe.T_WC[0].data.cpu().numpy()
        dd2 = frame.T_WC[0].data.cpu().numpy()
        self.fp.writelines('%d %d %f %f %f %f %f %f\n'%(keyframe.frame_id,frame.frame_id,dd[0],dd[1],dd[2],dd2[0],dd2[1],dd2[2]));self.fp.flush()

        # 跟踪完成后，把当前帧在关键帧坐标系下的 3D 点融合回关键帧点图，
        # 等价于用更多观测逐步修正关键帧的稠密几何表示。
        Xkk = T_CkCf.act(Xkf)
        keyframe.update_pointmap(Xkk, Ckf)
        # write back the fitered pointmap
        self.keyframes[len(self.keyframes) - 1] = keyframe

        # 关键帧选择本质是在判断：
        # “当前视角是否已经偏离旧关键帧 enough，使得继续复用旧关键帧会损伤配准质量？”
        n_valid = valid_kf.sum()
        match_frac_k = n_valid / valid_kf.numel()
        unique_frac_f = (
            torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )

        new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]

        # 一旦升格为新关键帧，旧的像素对应初始化就失效了，
        # 必须重置，让下一帧重新以恒等映射为起点匹配。
        if new_kf:
            self.reset_idx_f2k()

        return (
            new_kf,
            [
                keyframe.X_canon,
                keyframe.get_average_conf(),
                frame.X_canon,
                frame.get_average_conf(),
                Qkf,
                Qff,
            ],
            False,
        )

    def get_points_poses(self, frame, keyframe, idx_f2k, img_size, use_calib, K=None):
        Xf = frame.X_canon
        Xk = keyframe.X_canon
        T_WCf = frame.T_WC
        T_WCk = keyframe.T_WC

        # Average confidence
        Cf = frame.get_average_conf()
        Ck = keyframe.get_average_conf()

        meas_k = None
        valid_meas_k = None

        if use_calib:
            # 标定模式下，先把点约束回相机射线，
            # 这样后续重投影残差才与像素几何模型一致。
            Xf = constrain_points_to_ray(img_size, Xf[None], K).squeeze(0)
            Xk = constrain_points_to_ray(img_size, Xk[None], K).squeeze(0)

            # 关键帧的观测量 = 像素坐标 + 对数深度。
            # 对数深度而不是原始深度，能让远近点的尺度变化更平衡。
            uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
            uv_k = uv_k.view(-1, 2)
            meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)
            # Avoid any bad calcs in log
            valid_meas_k = Xk[..., 2:3] > self.cfg["depth_eps"]
            meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        return Xf[idx_f2k], Xk, T_WCf, T_WCk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    def solve(self, sqrt_info, r, J):
        # 这里显式写成加权最小二乘的正规方程，便于控制鲁棒核与信息矩阵。
        # `sqrt_info` 既编码测量噪声，也编码 Huber 权重。
        whitened_r = sqrt_info * r
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * (b.T @ b).item()

        L = torch.linalg.cholesky(H, upper=False)
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost

    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid):
        last_error = 0
        sqrt_info_ray = 1 / self.cfg["sigma_ray"] * valid * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg["sigma_dist"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)

        # 在当前实现里虽然状态用 Sim3 表示，但这里求的是“相对位姿更新”，
        # 尺度不会被自由漂移地单独放开。
        T_CkCf = T_WCk.inv() * T_WCf

        # 把关键帧上的 3D 点预先变成“射线方向 + 到相机距离”的混合表示，
        # 这样残差同时兼顾朝向和深度，不依赖显式相机标定。
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
            # 预测当前帧点投到关键帧坐标系下的几何量，与关键帧观测直接做残差。
            r = rd_k - rd_f_Ck
            # 链式法则: 几何量对点的导数 * 点对位姿的导数。
            J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # 相对位姿收敛后再恢复到世界系绝对位姿。
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf

    def opt_pose_calib_sim3(
        self, Xf, Xk, T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, K, img_size
    ):
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        # 标定模式下直接优化重投影残差，要求点、内参与像素模型相互一致。
        T_CkCf = T_WCk.inv() * T_WCf

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
            )
            valid2 = valid_proj & valid_meas_k
            sqrt_info2 = valid2 * sqrt_info

            # 观测是关键帧的像素+对数深度，预测值来自当前帧点变换后的投影结果。
            r = meas_k - pzf_Ck
            # 对像素/深度投影链式求导，得到位姿更新方向。
            J = -dpzf_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # 把收敛后的相对位姿重新挂回世界坐标系。
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf
