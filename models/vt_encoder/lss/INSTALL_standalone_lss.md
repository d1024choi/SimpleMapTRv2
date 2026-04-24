# Installation Guide for `standalone_lss_transform.py`

`standalone_lss_transform.py` has **no dependency on mmdet3d, mmcv, or any
OpenMMLab library**.  The only non-standard requirement is the self-contained
`bev_pool_standalone` CUDA extension — two source files extracted from the
MapTR mmdetection3d fork, built with a plain `torch.utils.cpp_extension`.

---

## What is inside `bev_pool_standalone/`

```
bev_pool_standalone/          ← Python project root (contains setup.py)
├── setup.py                  ← plain setuptools + CUDAExtension build
├── bev_pool_standalone/      ← Python package
│   ├── __init__.py           ← exposes  bev_pool()
│   └── bev_pool.py           ← QuickCumsumCuda wrapper (pure Python)
└── src/
    ├── bev_pool.cpp          ← pybind11 C++ bindings
    └── bev_pool_cuda.cu      ← CUDA kernel (bev_pool_kernel)
```

After building, a single compiled file is added:
```
bev_pool_standalone/bev_pool_standalone/
    bev_pool_ext.cpython-38-x86_64-linux-gnu.so   ← compiled CUDA extension
```

The `.so` is **machine-specific** (Python version × CUDA version × GPU SM arch)
and must be compiled on the target system.

---

## Dependency overview

| Component | Role | Source |
|---|---|---|
| CUDA Toolkit ≥ 11.1 | Compile the `.so`; run kernels | System / NVIDIA |
| Python 3.8 | Runtime | conda / system |
| PyTorch 1.9.1+cu111 | Deep-learning backend | PyTorch wheel |
| numpy ≥ 1.21 | Array utilities | pip |
| `bev_pool_standalone` | BEV-pool CUDA kernel | Build from `bev_pool_standalone/` |

**No mmdet3d, mmcv, or mmdet packages are required.**

---

## Step 0 – System requirements

```bash
nvcc --version      # CUDA Toolkit must be installed; e.g. "release 11.1"
nvidia-smi          # GPU must be visible
```

Install CUDA 11.1 from:  https://developer.nvidia.com/cuda-11-1-0-download-archive

---

## Step 1 – Create Python 3.8 environment

```bash
conda create -n lss_env python=3.8 -y
conda activate lss_env
```

---

## Step 2 – Install PyTorch (CUDA 11.1 build)

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# → 1.9.1+cu111  True
```

> Other PyTorch + CUDA combinations also work (e.g. torch 2.x + CUDA 12.x),
> as long as `nvcc` matches the torch CUDA version.
> The `.so` will be compiled against whatever torch/CUDA is active.

---

## Step 3 – Build the `bev_pool_standalone` CUDA extension

```bash
cd /path/to/MapTR-maptrv2/bev_pool_standalone
pip install -e .
```

This compiles `src/bev_pool.cpp` and `src/bev_pool_cuda.cu` via nvcc and
places the resulting `.so` inside `bev_pool_standalone/`.

Verify:
```bash
python -c "from bev_pool_standalone import bev_pool; print('bev_pool OK')"
```

---

## Step 4 – Install remaining Python packages

```bash
pip install -r /path/to/MapTR-maptrv2/requirements_standalone_lss.txt
```

---

## Step 5 – Verify the full standalone module

```bash
cd /path/to/MapTR-maptrv2
python - <<'EOF'
from standalone_lss_transform import LSSTransformStandalone
import torch

cfg = dict(
    in_channels=256, out_channels=256, feat_down_sample=32,
    pc_range=[-30., -15., -2., 30., 15., 2.],
    voxel_size=[0.15, 0.15, 4.0], dbound=[2., 58., 0.5],
    downsample=2,
    depthnet_cfg=dict(use_dcn=False, with_cp=False, aspp_mid_channels=96),
    grid_config=dict(depth=[2., 58., 0.5]),
)
model = LSSTransformStandalone(**cfg).cuda().eval()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Ready.")
EOF
```

---

## Dependency graph (standalone)

```
standalone_lss_transform.py
│
├── torch / torch.nn / torch.nn.functional    ← PyTorch
├── numpy                                      ← pip
│
└── bev_pool_standalone                        ← built locally
    ├── bev_pool.py          (pure Python)
    └── bev_pool_ext.*.so    (compiled CUDA kernel)
        ├── bev_pool.cpp     (pybind11 bindings)
        └── bev_pool_cuda.cu (CUDA kernel)
```

---

## Rebuilding after changing torch or CUDA version

If you upgrade PyTorch or switch CUDA versions, recompile the extension:

```bash
cd /path/to/MapTR-maptrv2/bev_pool_standalone
pip install -e .          # recompiles and reinstalls
```

---

## Troubleshooting

| Error | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: bev_pool_standalone` | Extension not built | Run `pip install -e .` in `bev_pool_standalone/` |
| `ImportError: bev_pool_ext` | `.so` missing or wrong Python/CUDA version | Rebuild with `pip install -e .` |
| `CUSOLVER_STATUS_INTERNAL_ERROR` | cuSolver init bug on some CUDA builds | Use CPU-fallback patch from `validate_standalone_lss.py` |
| `IndexError: index -1 is out of bounds` in bev_pool | Zero valid frustum points (all outside BEV range) | Check `pc_range` and camera extrinsics |
| nvcc not found | CUDA Toolkit not on PATH | `export PATH=/usr/local/cuda/bin:$PATH` |
