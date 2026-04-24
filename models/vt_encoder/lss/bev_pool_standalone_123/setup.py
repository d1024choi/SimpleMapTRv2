"""setup.py for the standalone bev_pool CUDA extension.

Builds `bev_pool_ext` – the CUDA kernel used by LSSTransform –
with no dependency on mmdet3d, mmcv, or any OpenMMLab library.

Usage
-----
    cd bev_pool_standalone
    pip install -e .          # editable (recommended during development)
    # or
    python setup.py build_ext --inplace   # in-place build

Requirements
------------
* Python >= 3.8
* PyTorch  >= 1.9  (with matching CUDA toolkit)
* CUDA Toolkit (nvcc must be on PATH or CUDA_HOME must be set)

CUDA toolkit vs. PyTorch wheel
-------------------------------
If ``nvcc`` reports a newer major CUDA than ``torch.version.cuda`` (e.g. 12.x
toolkit with PyTorch cu117), PyTorch's extension builder aborts by default.
This setup relaxes **only** that major-mismatch error for this package and
emits a warning, so the extension can still compile in many setups.

To keep PyTorch's strict check (fail on mismatch), set::

    export BEV_POOL_STRICT_CUDA_VERSION=1
"""

import os
import warnings
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension as _torch_cpp_ext

# PyTorch refuses to build CUDA extensions when the toolkit major != torch.version.cuda
# (e.g. CUDA 12.4 nvcc with torch 1.13+cu117). That is often still buildable; only
# relax for this subprocess — does not modify the installed torch package.
if os.environ.get("BEV_POOL_STRICT_CUDA_VERSION", "").lower() in ("1", "true", "yes", "on"):
    pass
else:
    _orig_cuda_check = _torch_cpp_ext._check_cuda_version

    def _bev_pool_cuda_version_check(compiler_name, compiler_version):
        try:
            _orig_cuda_check(compiler_name, compiler_version)
        except RuntimeError as e:
            if "mismatches the version that was used to compile" in str(e):
                warnings.warn(
                    "bev_pool_standalone: CUDA toolkit version differs from PyTorch's "
                    "CUDA; continuing the build. For strict checking, set "
                    "BEV_POOL_STRICT_CUDA_VERSION=1. Prefer a toolkit matching "
                    "torch.version.cuda when possible.",
                    UserWarning,
                )
                return
            raise

    _torch_cpp_ext._check_cuda_version = _bev_pool_cuda_version_check

# Resolve source paths relative to this setup.py, regardless of the cwd
# from which the build command is invoked.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "src")

setup(
    name="bev_pool_standalone",
    version="1.0.0",
    description="Standalone CUDA BEV-pool kernel (extracted from MapTR/mmdetection3d)",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            # Compiled .so is importable as:  from bev_pool_standalone import bev_pool_ext
            name="bev_pool_standalone.bev_pool_ext",
            sources=[
                os.path.join(SRC, "bev_pool.cpp"),
                os.path.join(SRC, "bev_pool_cuda.cu"),
            ],
            extra_compile_args={
                "cxx":  ["-O3", "-std=c++14"],
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-std=c++14",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch"],
)
