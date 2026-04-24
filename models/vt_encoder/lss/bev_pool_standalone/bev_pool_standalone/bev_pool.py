"""bev_pool.py – pure-Python wrapper around the bev_pool_ext CUDA kernel.

Verbatim copy of the QuickCumsumCuda logic from:
  MapTR/mmdetection3d/mmdet3d/ops/bev_pool/bev_pool.py

No mmdet3d / mmcv dependencies.
"""

import torch
from . import bev_pool_ext   # the compiled .so

__all__ = ["bev_pool"]


class QuickCumsumCuda(torch.autograd.Function):
    """Sort-then-cumsum BEV pooling, implemented as a differentiable op."""

    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        # Mark interval boundaries (first occurrence of each voxel rank).
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]

        interval_starts  = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1]  = x.shape[0] - interval_starts[-1]

        geom_feats = geom_feats.int()

        out = bev_pool_ext.bev_pool_forward(
            x, geom_feats,
            interval_lengths, interval_starts,
            B, D, H, W,
        )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad, geom_feats,
            interval_lengths, interval_starts,
            B, D, H, W,
        )
        return x_grad, None, None, None, None, None, None


def bev_pool(feats, coords, B, D, H, W):
    """BEV feature pooling via sort + cumulative sum.

    Args:
        feats  (Tensor): ``(N, C)``  per-point features.
        coords (Tensor): ``(N, 4)``  integer voxel coordinates
                         ``[gx, gy, gz, batch_idx]``.
        B (int): batch size.
        D (int): depth  / Z-grid size  (``nx[2]``).
        H (int): height / X-grid size  (``nx[0]``).
        W (int): width  / Y-grid size  (``nx[1]``).

    Returns:
        Tensor: ``(B, C, D, H, W)`` BEV feature volume.
    """
    assert feats.shape[0] == coords.shape[0]

    # Compute a unique linear rank for every point so that identical-voxel
    # points are contiguous after sorting.
    ranks = (
        coords[:, 0] * (W * D * B)
        + coords[:, 1] * (D * B)
        + coords[:, 2] * B
        + coords[:, 3]
    )
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    x = QuickCumsumCuda.apply(feats, coords, ranks, B, D, H, W)
    x = x.permute(0, 4, 1, 2, 3).contiguous()   # (B, D, H, W, C) → (B, C, D, H, W)
    return x
