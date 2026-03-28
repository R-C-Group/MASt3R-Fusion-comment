# MASt3R-Fusion 代码阅读指南

## 一、项目总览

**MASt3R-Fusion** 是一个基于 [MASt3R](https://github.com/naver/mast3r)（多视角立体匹配网络）的 **视觉-惯性 SLAM（Simultaneous Localization And Mapping，同步定位与建图）** 系统。它将 MASt3R 的深度学习密集匹配能力与 IMU（惯性测量单元）数据融合，通过 GTSAM 因子图优化框架实现精确的位姿估计和三维重建。

### 核心特性

| 特性         | 说明                                                              |
| ------------ | ----------------------------------------------------------------- |
| **视觉前端** | 基于 MASt3R 的密集特征提取与匹配                                  |
| **位姿跟踪** | 基于 Sim3（相似变换群）的帧到关键帧位姿优化                       |
| **后端优化** | GTSAM 因子图 + LM 优化器，支持视觉因子、IMU 预积分因子、GNSS 因子 |
| **回环检测** | 基于图像检索的回环检测与回环约束                                  |
| **全局优化** | 离线全局位姿图优化，融合视觉、IMU、GNSS 数据                      |
| **滑动窗口** | 边缘化机制实现有限内存下的实时运行                                |

---

## 二、项目目录结构

```
MASt3R-Fusion-comment/
├── main.py                          # 🚀 主程序入口：在线视觉-惯性 SLAM
├── main_loop.py                     # 🔁 回环检测主程序
├── main_global_optimization.py      # 🌐 离线全局优化主程序
├── config/                          # ⚙️ 配置文件
│   ├── base_kitti360.yaml           #    KITTI360 数据集基础配置
│   ├── base_subt_handheld.yaml      #    SubT 数据集配置
│   └── intrinsics_*.yaml            #    相机内参与 IMU-相机外参
├── mast3r_fusion/                   # 📦 核心包
│   ├── config.py                    #    配置文件加载与合并
│   ├── dataloader.py                #    数据集加载器（多种格式支持）
│   ├── frame.py                     #    帧（Frame）与关键帧（Keyframe）管理
│   ├── geometry.py                  #    几何运算（投影、反投影、射线约束等）
│   ├── global_opt.py                #    ⭐ 核心：因子图 + 位姿优化
│   ├── tracker.py                   #    帧跟踪器（帧到关键帧位姿估计）
│   ├── matching.py                  #    密集匹配（迭代投影 + 亚像素精细化）
│   ├── mast3r_utils.py              #    MASt3R 模型加载与推理封装
│   ├── retrieval_database.py        #    图像检索数据库（倒排索引）
│   ├── vio_utils.py                 #    视觉-惯性对齐（VIO 初始化）
│   ├── visualization.py             #    3D 可视化窗口（OpenGL）
│   ├── visualization_utils.py       #    可视化辅助工具
│   ├── nonlinear_optimizer.py       #    非线性优化工具（收敛判断、鲁棒核函数）
│   ├── multiprocess_utils.py        #    多进程通信工具
│   ├── lietorch_utils.py            #    Lie 群工具（Sim3 → SE3 转换）
│   ├── image.py                     #    图像梯度计算
│   ├── tictoc.py                    #    计时器
│   ├── evaluate.py                  #    评估与结果保存
│   ├── geoFunc/                     #    大地坐标与惯性导航工具库
│   │   ├── const_value.py           #       WGS-84 常量
│   │   ├── trans.py                 #       坐标变换（ECEF↔ENU、姿态角↔旋转矩阵）
│   │   └── data_utils.py            #       IMU 数据池、图像数据集加载
│   └── backend/                     #    C++ 后端加速模块
├── evaluation/                      # 📊 评估脚本
│   ├── evaluate_kitti360.py
│   ├── evaluate_subt.py
│   └── check_h5.py
├── thirdparty/                      # 第三方依赖
├── gtsam/                           # GTSAM 因子图优化库
└── resources/                       # OpenGL 着色器资源
```

---

## 三、系统架构与数据流

### 3.1 整体运行流程

```
┌─────────────────────────────────────────────────────────────┐
│                     main.py 主循环                           │
│                                                             │
│  1. 加载模型 (MASt3R)                                        │
│  2. 加载数据集                                               │
│  3. 循环读取每一帧图像:                                       │
│     ├─ INIT 模式: 单目推理初始化第一帧 → 切换到 TRACKING       │
│     ├─ TRACKING 模式: tracker 跟踪当前帧到上一关键帧           │
│     │   └─ 判断是否需要插入新关键帧                            │
│     ├─ RELOC 模式: 重定位（跟踪丢失时触发）                   │
│     └─ 运行后端优化:                                          │
│         ├─ 图像检索找回环候选                                  │
│         ├─ 添加视觉因子到因子图                                │
│         ├─ 添加 IMU 预积分因子                                 │
│         ├─ Gauss-Newton 求解位姿                               │
│         └─ 触发 V-I 初始化（7 帧后）                          │
│  4. 保存因子图 & 帧数据到 h5/pkl                              │
│  5. 可选：启动 3D 可视化进程                                  │
└─────────────────────────────────────────────────────────────┘
            ↓ 输出 graph.pkl + data.h5
┌─────────────────────────────────────────────────────────────┐
│                 main_loop.py 回环检测                         │
│                                                             │
│  1. 从 h5 加载所有关键帧                                      │
│  2. 使用图像检索查找回环候选                                  │
│  3. 基于置信度地图过滤候选                                    │
│  4. 对每对回环候选进行 MASt3R 匹配 + 位姿求解                 │
│  5. 输出 graph_loop.pkl                                      │
└─────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────┐
│           main_global_optimization.py 全局优化                │
│                                                             │
│  1. 加载 graph.pkl + graph_loop.pkl                          │
│  2. 构建包含所有因子的 GTSAM 因子图:                          │
│     ├─ 视觉因子（Hessian → GTSAM 线性容器因子）              │
│     ├─ IMU 预积分因子                                         │
│     ├─ 回环因子（带 Cauchy 鲁棒核）                          │
│     ├─ GNSS 因子（可选）                                     │
│     └─ 外参约束因子                                           │
│  3. 迭代 6 轮 LM 优化                                        │
│  4. 输出优化后的结果                                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 符号约定（GTSAM 变量）

| 符号   | 含义                                   | 维度         |
| ------ | -------------------------------------- | ------------ |
| `X(i)` | 第 i 帧的**相机位姿** (Pose3)          | SE(3) = 6DoF |
| `S(i)` | 第 i 帧的**尺度因子**                  | 1            |
| `Z(i)` | 第 i 帧的 **IMU 坐标系位姿** (Pose3)   | SE(3)        |
| `V(i)` | 第 i 帧的**速度**                      | 3            |
| `B(i)` | 第 i 帧的 **IMU 偏置** (加速度+陀螺仪) | 6            |
| `C(i)` | **相机-IMU 外参** (Tic)                | SE(3)        |

---

## 四、核心模块详解

### 4.1 `main.py` — 在线 SLAM 主程序

**文件位置**: [main.py](file:///c:/Users/Kwan_/Desktop/MASt3R-Fusion-comment/main.py)

这是系统的主入口，执行在线的视觉-惯性 SLAM 流程。

**关键流程**:
1. **初始化**（L132-228）：加载 MASt3R 模型、配置、数据集、内参、创建共享关键帧与状态、启动可视化进程
2. **主循环**（L236-407）：逐帧处理每张图像
   - `Mode.INIT`：用单目推理初始化第一帧的点云和置信度
   - `Mode.TRACKING`：`FrameTracker.track()` 跟踪当前帧相对于最新关键帧的位姿
   - `Mode.RELOC`：重定位模式（跟踪丢失时）
3. **后端**（`run_backend` 函数, L45-130）：
   - 通过 `retrieval_database` 查找共视帧
   - `factor_graph.add_factors()` 添加视觉约束
   - `factor_graph.solve_GN_calib()` 使用 LM 求解
4. **滑动窗口**（L400-401）：关键帧数超过 30 时执行 roll_up，保持内存占用可控
5. **输出**：结果写入文本文件，帧数据保存到 HDF5，因子图保存到 pkl

**关键函数**:
- `find_valid_numbers(a, b)` — 从检索结果中过滤有效的回环候选帧
- `run_backend(states, keyframes)` — 执行一次后端优化迭代

---

### 4.2 `mast3r_fusion/global_opt.py` — 因子图与优化

**文件位置**: [global_opt.py](file:///c:/Users/Kwan_/Desktop/MASt3R-Fusion-comment/mast3r_fusion/global_opt.py)

这是系统最核心的模块，实现了因子图管理和位姿优化。

#### `FactorGraph` 类

管理整个因子图的生命周期：

| 方法                  | 作用                                                           |
| --------------------- | -------------------------------------------------------------- |
| `__init__`            | 初始化模型、IMU 参数、预积分参数、滑窗配置                     |
| `add_factors(ii, jj)` | 进行 MASt3R 对称匹配，计算匹配质量，筛选有效边，添加到因子图   |
| `solve_GN_calib()`    | ⭐ 核心求解函数：构建 GTSAM 图、添加 IMU/外参/先验因子、LM 优化 |
| `save_graph()`        | 保存因子图边的 Hessian 矩阵到 pkl（供全局优化使用）            |
| `predict_pose()`      | 利用 IMU 预积分预测下一帧位姿                                  |
| `solve_VI_init()`     | V-I 初始化：在积累 7 帧后估计重力方向、尺度和速度              |

#### `Align2GTSAM_factors()` 函数

将 MASt3R 优化产生的 Hessian 矩阵（7×7，包含 Sim3 的 t,R,s）转换为 GTSAM 兼容的线性容器因子（`LinearContainerFactor`）。涉及从 Sim3 李代数到 GTSAM 的 Pose3 + Scale 参数化的雅可比变换。

#### 滑动窗口与边缘化

当关键帧超过 `window_num` 时，旧的因子和变量被边缘化（`gtsam.marginalizeOut`），产生一个先验因子（Schur 补），保持计算复杂度可控。

---

### 4.3 `mast3r_fusion/tracker.py` — 帧跟踪

**文件位置**: [tracker.py](file:///c:/Users/Kwan_/Desktop/MASt3R-Fusion-comment/mast3r_fusion/tracker.py)

#### `FrameTracker` 类

负责将当前帧注册到最后一个关键帧的坐标系中：

1. **匹配**：`mast3r_match_asymmetric()` 进行非对称匹配（从当前帧到关键帧）
2. **位姿估计**：
   - `opt_pose_ray_dist_sim3()` — 无标定模式：基于射线方向+深度距离的残差
   - `opt_pose_calib_sim3()` — 有标定模式：基于像素重投影+对数深度的残差
3. **关键帧选择**：根据匹配覆盖率（`match_frac_thresh`）决定是否插入新关键帧
4. **关键帧点云更新**：跟踪后利用新的观测更新上一关键帧的点云

**优化过程**：使用 Gauss-Newton 法，通过 Cholesky 分解求解正规方程，支持 Huber 鲁棒核函数。

---

### 4.4 `mast3r_fusion/matching.py` — 密集匹配

**文件位置**: [matching.py](file:///c:/Users/Kwan_/Desktop/MASt3R-Fusion-comment/mast3r_fusion/matching.py)

实现帧间的密集像素匹配：

1. **迭代投影匹配**（`match_iterative_proj`）：
   - 将 3D 点投影到参考帧的射线图中
   - 迭代优化匹配位置（基于射线方向一致性）
   - 调用 C++ 后端 `mast3r_fusion_backends.iter_proj()`
2. **遮挡检测**：基于匹配点的 3D 距离差异过滤遮挡
3. **描述子精细匹配**（`refine_matches`）：在局部邻域内用描述子相似度精细化
4. **亚像素精细化**：支持 2× 和 4× 的亚像素上采样匹配

---

### 4.5 `mast3r_fusion/mast3r_utils.py` — MASt3R 模型封装

**文件位置**: [mast3r_utils.py](file:///c:/Users/Kwan_/Desktop/MASt3R-Fusion-comment/mast3r_fusion/mast3r_utils.py)

封装了对 MASt3R 模型的所有调用：

| 函数                              | 作用                                         |
| --------------------------------- | -------------------------------------------- |
| `load_mast3r()`                   | 加载预训练的 MASt3R 模型                     |
| `load_retriever()`                | 加载图像检索模型                             |
| `mast3r_inference_mono()`         | 单目推理：自身与自身配对，输出 3D 点和置信度 |
| `mast3r_symmetric_inference()`    | 对称推理：两帧互相预测对方的 3D 点           |
| `mast3r_decode_symmetric_batch()` | 批量对称解码                                 |
| `mast3r_match_symmetric()`        | 对称匹配：解码 + 匹配 + 质量评估             |
| `mast3r_match_asymmetric()`       | 非对称匹配（用于跟踪）                       |
| `resize_img()`                    | 图像缩放到模型所需尺寸（512 或 224）         |

**MASt3R 输出**（每对图像）：
- `X` (pts3d): 3D 点云（相机坐标系）
- `C` (conf): 3D 点的置信度
- `D` (desc): 用于匹配的密集描述子
- `Q` (desc_conf): 描述子置信度

---

### 4.6 `mast3r_fusion/frame.py` — 帧与关键帧管理

**文件位置**: [frame.py](file:///c:/Users/Kwan_/Desktop/MASt3R-Fusion-comment/mast3r_fusion/frame.py)

#### `Frame` 数据类

存储一帧的所有信息：

| 属性       | 类型   | 说明                   |
| ---------- | ------ | ---------------------- |
| `frame_id` | int    | 在数据集中的索引       |
| `img`      | Tensor | 归一化后的图像 (3×H×W) |
| `T_WC`     | Sim3   | 世界到相机的 Sim3 变换 |
| `X_canon`  | Tensor | 相机坐标系下的 3D 点   |
| `C`        | Tensor | 置信度图               |
| `feat`     | Tensor | MASt3R 编码器特征      |
| `pos`      | Tensor | patch 位置编码         |

**点云更新策略**（`update_pointmap`）：支持多种融合方式：
- `first`：只保留首次观测
- `recent`：只保留最近观测
- `best_score`：保留置信度最高的
- `indep_conf`：逐像素按置信度选择
- `weighted_pointmap`：加权平均
- `weighted_spherical`：球坐标加权平均

#### `SharedStates` / `SharedKeyframes`

跨进程共享的状态/关键帧缓冲区。使用 `torch.Tensor.share_memory_()` 实现 GPU 数据的进程间共享。

`SharedKeyframes` 维护一个固定大小（默认 64）的滑动缓冲区，通过 `roll_up()` 实现滑动窗口。

---

### 4.7 `mast3r_fusion/vio_utils.py` — V-I 初始化

**文件位置**: [vio_utils.py](file:///c:/Users/Kwan_/Desktop/MASt3R-Fusion-comment/mast3r_fusion/vio_utils.py)

#### `VisualIMUAlignment()` 函数

参考 VINS-Fusion 的 V-I 对齐方法，包含三个阶段：

1. **陀螺仪偏置估计**（`solveGyroscopeBias`）：利用旋转约束，线性求解陀螺仪偏置
2. **线性对齐**（`linearAlignment`）：
   - 联立位置和速度约束
   - 求解每帧速度、重力方向和尺度因子
3. **重力精化**（`RefineGravity`）：
   - 在重力方向的切平面上迭代调整（4 次）
   - 固定重力大小为 9.81 m/s²
4. **坐标系对齐**（`g2R`）：将世界坐标系的 z 轴对齐到重力方向

---

### 4.8 `mast3r_fusion/dataloader.py` — 数据加载

**文件位置**: [dataloader.py](file:///c:/Users/Kwan_/Desktop/MASt3R-Fusion-comment/mast3r_fusion/dataloader.py)

支持多种数据集格式：

| 数据集类             | 适用场景           |
| -------------------- | ------------------ |
| `TUMDataset`         | TUM RGB-D 数据集   |
| `EurocDataset`       | EuRoC MAV 数据集   |
| `ETH3DDataset`       | ETH3D 数据集       |
| `SevenScenesDataset` | 7-Scenes 数据集    |
| `RealsenseDataset`   | RealSense 实时相机 |
| `Webcam`             | 普通网络摄像头     |
| `MP4Dataset`         | MP4 视频文件       |
| `RGBFiles`           | RGB 图片文件夹     |
| `StampedFiles`       | 带时间戳的图片文件 |

#### `Intrinsics` 类

管理相机内参，支持：
- **针孔模型**（pinhole）+ 畸变矫正
- **Mei 全景模型**（mei）+ 全向去畸变
- 自动计算缩放后的内参（适配 MASt3R 的输入尺寸）

---

### 4.9 `mast3r_fusion/geometry.py` — 几何运算

**文件位置**: [geometry.py](file:///c:/Users/Kwan_/Desktop/MASt3R-Fusion-comment/mast3r_fusion/geometry.py)

| 函数                                       | 作用                                     |
| ------------------------------------------ | ---------------------------------------- |
| `skew_sym(x)`                              | 计算向量的反对称矩阵 [x]×                |
| `point_to_dist(X)`                         | 计算点到原点的距离                       |
| `point_to_ray_dist(X)`                     | 将 3D 点分解为射线方向+深度（含雅可比）  |
| `constrain_points_to_ray(img_size, Xs, K)` | 将 3D 点约束到相机射线上（用内参反投影） |
| `act_Sim3(X, pC)`                          | Sim3 变换作用于 3D 点（含雅可比）        |
| `project_calib(P, K, img_size)`            | 标定投影：3D→像素+对数深度（含雅可比）   |
| `backproject(p, z, K)`                     | 反投影：像素+深度→3D 点                  |
| `get_pixel_coords(b, img_size)`            | 生成像素坐标网格                         |

---

### 4.10 `mast3r_fusion/retrieval_database.py` — 图像检索

**文件位置**: [retrieval_database.py](file:///c:/Users/Kwan_/Desktop/MASt3R-Fusion-comment/mast3r_fusion/retrieval_database.py)

基于 MASt3R 的检索模型，实现了增量式倒排文件索引（IVF）：

1. 提取局部特征 → 量化到视觉词典 → 聚合为图像级描述
2. 查询时计算与数据库中所有图像的相似度
3. 返回 Top-K 最相似图像作为回环/共视候选

---

### 4.11 `mast3r_fusion/visualization.py` — 3D 可视化

**文件位置**: [visualization.py](file:///c:/Users/Kwan_/Desktop/MASt3R-Fusion-comment/mast3r_fusion/visualization.py)

基于 ModernGL + ImGui 的实时 3D 可视化窗口：
- 显示关键帧位姿（视锥体）
- 显示点云（surfel / 三角面片 / 点）
- 相机跟随模式
- ImGui 控制面板（暂停、参数调节等）

---

### 4.12 `main_loop.py` — 回环检测

**文件位置**: [main_loop.py](file:///c:/Users/Kwan_/Desktop/MASt3R-Fusion-comment/main_loop.py)

离线回环检测与回环因子计算：

1. 从 HDF5 加载所有关键帧
2. 构建"置信度地图" `gen_conf_map_vec()`：基于关键帧间的里程计不确定性（沿轨迹方向和横向误差积累）估计回环候选的可信度
3. 对每对回环候选：
   - MASt3R 对称匹配
   - `mini_solve()` 优化相对位姿
   - 保存 Hessian 矩阵作为回环因子
4. 输出保存到 `graph_loop.pkl`

---

### 4.13 `main_global_optimization.py` — 全局优化

**文件位置**: [main_global_optimization.py](file:///c:/Users/Kwan_/Desktop/MASt3R-Fusion-comment/main_global_optimization.py)

离线全局位姿图优化：

1. **加载数据**：从 `graph.pkl` 加载视觉因子，从 `graph_loop.pkl` 加载回环因子
2. **GNSS 对齐**（可选）：利用 GNSS 和 IMU 预积分将局部坐标系对齐到 ENU 坐标系
3. **迭代优化**（6 轮）：
   - 视觉 Hessian 因子 → GTSAM 线性容器因子（`Align2GTSAM_factors`）
   - IMU 预积分因子（`CombinedImuFactor`）
   - 外参约束因子（`ExPoseConstraintFactor`）
   - 回环因子（带 Cauchy 鲁棒核，逐步收紧）
   - GNSS 因子（`GPSFactorLever`）
   - LM 优化器求解
4. **输出**：优化后的位姿、偏置、尺度写入结果文件

---

### 4.14 `mast3r_fusion/geoFunc/` — 大地坐标工具库

#### `const_value.py`
WGS-84 椭球参数（`a` = 长半轴, `finv` = 扁率倒数）

#### `trans.py`
坐标变换工具函数：
- `cart2geod()` — ECEF → 大地坐标（纬度、经度、高度）
- `cart2enu()` / `enu2cart()` — ECEF 差分 ↔ ENU（东北天）
- `att2m()` / `m2att()` — 姿态角 ↔ 旋转矩阵
- `q2R()` — 四元数 → 旋转矩阵
- `R2ypr()` / `ypr2R()` — 旋转矩阵 ↔ Yaw/Pitch/Roll
- `FromTwoVectors()` — 从两个向量构造旋转矩阵
- `alignRt()` — ICP 点集配准（SVD 方法）

#### `data_utils.py`

- `IMUPool` — IMU 数据管理：存储时间排序的 IMU 测量，支持按时间区间查询
- `ImageDataset` — 带内参/外参的图像数据集管理
- `loadIE()` / `loadGlobal()` — 加载惯导结果文件（IE 格式 / 自定义格式）

---

### 4.15 其他辅助模块

#### `nonlinear_optimizer.py`
- `check_convergence()` — 检查优化是否收敛（相对误差 + 步长阈值）
- `huber()` / `tukey()` — 鲁棒核函数权重

#### `multiprocess_utils.py`
- `try_get_msg()` — 非阻塞消息获取
- `FakeQueue` — 无可视化时的假队列
- `new_queue()` — 创建消息队列

#### `lietorch_utils.py`
- `as_SE3()` — 从 Sim3 提取 SE3（丢弃尺度信息）

#### `image.py`
- `img_gradient()` — Scharr 算子计算图像 x/y 方向梯度

#### `tictoc.py`
- CUDA 同步计时器

#### `evaluate.py`
- `save_traj()` — 保存轨迹（TUM 格式）
- `save_reconstruction()` — 保存 PLY 点云
- `save_keyframes()` — 保存关键帧图片

---

## 五、关键算法详解

### 5.1 Sim3 视觉 BA → GTSAM 因子转换

MASt3R-Fusion 使用 Sim3（7DoF = 平移3 + 旋转3 + 尺度1）表示位姿。但 GTSAM 原生使用 Pose3（SE3, 6DoF）+ 独立尺度变量。

`Align2GTSAM_factors()` 在两种参数化之间建立桥梁：

1. 在 Sim3 空间中计算 `Hessian (H)` 和 `gradient (v)`
2. 计算从 GTSAM 参数 `[S_i, X_i, S_j, X_j]` 到 Sim3 相对变换的雅可比 `J`
3. 转换：`H_gtsam = J^T * H * J`, `v_gtsam = J^T * v`
4. 封装为 `HessianFactor` → `LinearContainerFactor`

### 5.2 V-I 初始化

参考 VINS-Fusion 的经典方法，分步骤求解：

1. **陀螺仪偏置**：利用旋转预积分与视觉旋转的一致性约束
2. **速度 + 重力 + 尺度**：构建线性系统联立求解
3. **重力精化**：在切平面上迭代修正，保持重力大小约束
4. **坐标系对齐**：旋转世界坐标系使 z 轴与重力对齐

### 5.3 滑动窗口边缘化

当窗口中关键帧数超过 `window_num` 时：

1. 计算离开窗口边界的旧因子的信息矩阵
2. 使用 Schur 补进行边缘化（`gtsam.marginalizeOut`）
3. 产生一个先验因子，保留了旧帧对保留帧的约束信息
4. 对先验因子进行 rekey，使其索引适配新窗口

---

## 六、配置文件说明

以 `config/base_kitti360.yaml` 为例，主要配置项：

```yaml
dataset:
  subsample: 1               # 帧采样间隔
  img_downsample: 1           # 图像降采样因子
  center_principle_point: true # 是否居中主点

tracking:
  max_iters: 10               # 跟踪优化最大迭代次数
  sigma_pixel: 1.0            # 像素误差标准差
  sigma_depth: 0.1            # 深度误差标准差
  match_frac_thresh: 0.4      # 关键帧选择阈值

local_opt:
  max_iters: 3                # 局部优化迭代次数
  C_conf: 1.5                 # 置信度阈值
  Q_conf: 0.1                 # 描述子置信度阈值
  min_match_frac: 0.05        # 最小匹配比例

ms_opt:
  imu_noise: [0.1, 0.01, ...]  # IMU 噪声参数
  window_num: 15               # 滑动窗口大小

retrieval:
  k: 5                        # 检索 Top-K 数量
  min_thresh: 0.0             # 最低相似度阈值
```

---

## 七、运行方式

### 7.1 在线 SLAM

```bash
python main.py \
  --dataset path/to/images \
  --config config/base_kitti360.yaml \
  --calib config/intrinsics_kitti360.yaml \
  --imu_path path/to/imu.txt \
  --result_path result.txt \
  --save_h5
```

### 7.2 回环检测

```bash
python main_loop.py \
  --h5_file data.h5 \
  --config config/base_kitti360.yaml \
  --loop_output graph_loop.pkl
```

### 7.3 全局优化

```bash
python main_global_optimization.py \
  --graph_path graph.pkl \
  --loop_path graph_loop.pkl \
  --config config/base_kitti360.yaml \
  --calib_path config/intrinsics_kitti360.yaml \
  --imu_path path/to/imu.txt \
  --result_path optimized_result.txt
```

---

## 八、建议阅读顺序

1. **`config.py`** → 理解配置加载机制
2. **`frame.py`** → 理解帧的数据结构
3. **`dataloader.py`** → 理解数据如何加载
4. **`mast3r_utils.py`** → 理解 MASt3R 模型如何工作
5. **`matching.py`** → 理解密集匹配流程
6. **`tracker.py`** → 理解帧间位姿估计
7. **`geometry.py`** → 理解投影/反投影等几何运算
8. **`main.py`** → 理解在线 SLAM 主循环
9. **`global_opt.py`** → ⭐ 理解因子图优化核心
10. **`vio_utils.py`** → 理解 V-I 初始化
11. **`retrieval_database.py`** → 理解图像检索
12. **`main_loop.py`** → 理解回环检测
13. **`main_global_optimization.py`** → 理解全局优化