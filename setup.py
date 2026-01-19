from pathlib import Path
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
has_cuda = torch.cuda.is_available()

include_dirs = [
    os.path.join(ROOT, "mast3r_fusion/backend/include"),
    os.path.join(ROOT, "thirdparty/eigen"),
]

sources = [
    "mast3r_fusion/backend/src/gn.cpp",
]
extra_compile_args = {
    "cores": ["j8"],
    "cxx": ["-O3"],
}

if has_cuda:
    from torch.utils.cpp_extension import CUDAExtension

    sources.append("mast3r_fusion/backend/src/gn_kernels.cu")
    sources.append("mast3r_fusion/backend/src/matching_kernels.cu")
    extra_compile_args["nvcc"] = [
        "-O3",
        # 移除 60, 61, 70, 75 等旧架构，避免 CUDA 13 报错
        # "-gencode=arch=compute_60,code=sm_60",
        # "-gencode=arch=compute_61,code=sm_61",
        # "-gencode=arch=compute_70,code=sm_70",
        # "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        # 显式添加针对 Blackwell (11.0) 的支持
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_110,code=sm_110",
        "-gencode=arch=compute_110,code=compute_110",
    ]
    ext_modules = [
        CUDAExtension(
            "mast3r_fusion_backends",
            include_dirs=include_dirs,
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ]
else:
    print("CUDA not found, cannot compile backend!")

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
