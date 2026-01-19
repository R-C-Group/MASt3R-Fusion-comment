 <h1 align="center"> MASt3R-Fusion测试(基于NAVIDA Thor)
  </h1>


[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center">
  <a href="https://github.com/GREAT-WHU/MASt3R-Fusion">Github</a>
  | <a href="https://arxiv.org/pdf/2509.20757">Paper</a>
  </h3>
  <div align="center"></div>


## 实验配置

* 环境配置

```bash
# rm -rf .git
git clone git@github.com:R-C-Group/MASt3R-Fusion-comment.git --recursive

conda create -n mast3r_fusion python=3.11.9
conda activate mast3r_fusion
# conda remove --name mast3r_fusion --all
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install opencv-python==4.10.0.84 opencv-contrib-python==4.10.0.84
pip install h5py pyparsing
```

* 安装GTSAM，这是作者修改版本的，包含了边缘化以及Sim(3)视觉约束

```bash
conda activate mast3r_fusion

# git clone https://github.com/yuxuanzhou97/gtsam.git
git clone git@github.com:yuxuanzhou97/gtsam.git
cd gtsam
# cd .. && rm -rf build/
mkdir build && cd build
# cmake .. -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.11.9 -DPYTHON_EXECUTABLE=`which python` -Dpybind11_INCLUDE_DIR=$PYBIND11_INCLUDE_DIR
cmake .. \
  -DGTSAM_BUILD_PYTHON=1 \
  -DGTSAM_PYTHON_VERSION=3.11.9 \
  -DPYTHON_EXECUTABLE=`which python` \
  -Dpybind11_INCLUDE_DIR=$PYBIND11_INCLUDE_DIR \
  -DGTSAM_BUILD_WITH_SERIALIZATION=OFF \
  -DGTSAM_PYTHON_BUILD_WITH_SERIALIZATION=OFF \
  -DGTSAM_BUILD_UNSTABLE=OFF \
  -DCMAKE_CXX_STANDARD=17

make python-install -j12
```

* 可能会出现调用`/usr/include/pybind11/`等问题，需要安装最新版的pybind11，然后重新运行。

```bash
pip install --upgrade pybind11

# 获取详细的 pybind11 信息
PYBIND11_INCLUDE_DIR=$(python -c "import pybind11; print(pybind11.get_include())")
PYBIND11_CMAKE_DIR=$(python -c "import pybind11; import os; print(os.path.join(pybind11.__path__[0], 'share', 'cmake', 'pybind11'))")

echo "Pybind11 include: $PYBIND11_INCLUDE_DIR"
echo "Pybind11 cmake: $PYBIND11_CMAKE_DIR"

```

* GCC13编译问题：GCC 13 与 Boost Serialization 库在处理 std::optional 时的兼容性问题。GCC 13 更加严格地执行 C++17 标准，而 GTSAM 中某些序列化的模板推导在遇到 boost::serialization 的前向声明（即报错中的 struct boost::serialization::U）时，无法正确解析。

```bash
sudo apt install gcc-11 g++-11

# 查看gcc版本
gcc --version #出现的是默认版本
gcc-11 --version #出现新安装版本

cmake .. \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DGTSAM_BUILD_PYTHON=1 \
  -DGTSAM_PYTHON_VERSION=3.11.9 \
  -DPYTHON_EXECUTABLE=`which python` \
  -Dpybind11_INCLUDE_DIR=$PYBIND11_INCLUDE_DIR
```


* 工程安装：

```bash
cd {根目录}
conda activate mast3r_fusion
# pip install -e thirdparty/mast3r
pip install --no-build-isolation -e thirdparty/mast3r

pip install -e thirdparty/in3d

# torchcodec的安装
git clone https://github.com/pytorch/torchcodec
cd torchcodec
# 确保你在对应的虚拟环境下
export I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1
pip install . --no-build-isolation
# python -c "import torchcodec; print(f'Version: {torchcodec.__version__}'); from torchcodec import decoders; print('Decoders module: FOUND')"

pip install --no-build-isolation -e .
# pyproject.toml中去掉了pyrealsense2，这是RealSense 相机的依赖
```

* 查看系统架构`python -c "import torch; print(torch.cuda.get_device_capability())"`,Thor为`(11，0)`

* 下载权重文件

```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

* 下载KITTI-360数据集：
  * 首先，需要下载KITTI的`Perspective Images for Train & Val (128G)`。运行`cd dataset && bash download_2d_perspective_unrectified.sh`来下载；
  * 其次，还需下载预备的IMU及GT数据：[Google Drive](https://drive.google.com/file/d/1BO8zGvoey7IdwbWXmAdlhGPr6hiCFJ6Y/view?usp=drive_link)


运行下面测试：

```bash
conda activate mast3r_fusion

bash batch_kitti360_vi.sh # for real-time SLAM
```

测试效果如图所示

<div align="center">
  <img src="assets/Screenshot from 2026-01-19 15-46-54.png" width="90%" />
  <img src="assets/Screenshot from 2026-01-19 15-47-19.png" width="90%" />
<figcaption>  
</figcaption>
</div>
