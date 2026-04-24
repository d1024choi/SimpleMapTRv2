"""bev_pool_standalone – self-contained BEV-pool CUDA extension.

After building with ``pip install -e .`` (inside this directory),
use it as::

    from bev_pool_standalone import bev_pool
    out = bev_pool(feats, coords, B, D, H, W)   # (B, C, D, H, W)

This package has no dependency on mmdet3d, mmcv, or any OpenMMLab library.
"""

from .bev_pool import bev_pool   # noqa: F401

__all__ = ["bev_pool"]
