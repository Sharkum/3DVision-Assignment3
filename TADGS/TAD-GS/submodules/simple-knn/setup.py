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

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I/tmp/tadgs_env/targets/x86_64-linux/include", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK", "-Xcompiler=-Wno-template-body"], "cxx": ["-Wno-template-body"]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
