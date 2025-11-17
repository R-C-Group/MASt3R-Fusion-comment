 <h1 align="center"> MASt3R-Fusion测试
  </h1>


[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center">
  <a href="https://github.com/GREAT-WHU/MASt3R-Fusion">Github</a>
  | <a href="https://arxiv.org/pdf/2509.20757">Paper</a>
  </h3>
  <div align="center"></div>

<br>

## 实验配置


```bash
# rm -rf .git
git clone https://github.com/R-C-Group/MASt3R-Fusion-comment.git --recursive

conda create -n mast3r_fusion python=3.11.9
conda activate mast3r_fusion
# conda remove --name mast3r_fusion --all
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python==4.10.0.84 opencv-contrib-python==4.10.0.84
pip install h5py pyparsing
```

* 安装GTSAM，这是作者修改版本的，包含了边缘化以及Sim(3)视觉约束

```bash
conda activate mast3r_fusion
git clone https://github.com/yuxuanzhou97/gtsam.git
cd gtsam
mkdir build && cd build
cmake .. -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.11.9 -DPYTHON_EXECUTABLE=`which python`
make python-install -j12
```