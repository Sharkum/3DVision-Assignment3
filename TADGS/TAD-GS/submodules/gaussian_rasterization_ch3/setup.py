#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
import torch.utils.cpp_extension as cpp_ext
cpp_ext._check_cuda_version = lambda *args, **kwargs: None
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization_ch3",
    packages=['diff_gaussian_rasterization_ch3'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_ch3._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I/tmp/tadgs_env/targets/x86_64-linux/include", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK", "-Xcompiler=-Wno-template-body", "-I/tmp/tadgs_repo/submodules/gaussian_rasterization_ch3/third_party/glm"], "cxx": ["-Wno-template-body"]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
