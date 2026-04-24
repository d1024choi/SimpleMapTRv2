"""
Standalone re-implementation of MapTRDecoder and its sub-modules.
No mmcv / mmdet / mmdetection3d dependency — pure PyTorch only.

Module hierarchy (mirrors the mmcv/mmdet parameter naming so that
`standalone.load_state_dict(original.state_dict())` works out-of-the-box):

  StandaloneMapTRDecoder
    └─ layers: ModuleList[StandaloneDetrTransformerDecoderLayer × 6]
         ├─ attentions[0]: StandaloneMHA          (self-attn, mirrors mmcv.MultiheadAttention)
         ├─ attentions[1]: StandaloneCustomMSDeformableAttention  (cross-attn)
         ├─ ffns[0]:       StandaloneFFN           (mirrors mmcv.FFN)
         └─ norms[0..2]:   nn.LayerNorm × 3
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from models.vt_encoder.bevformer.ops.functions import MSDeformAttnFunction
import copy
import warnings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch multi-scale deformable attention (CPU-compatible).

    Args:
        value:              (bs, num_keys, num_heads, dim_per_head)
        value_spatial_shapes: (num_levels, 2)  — each row is (H, W)
        sampling_locations: (bs, num_queries, num_heads, num_levels, num_points, 2)
        attention_weights:  (bs, num_queries, num_heads, num_levels, num_points)

    Returns:
        (bs, num_queries, embed_dims)
    """
    bs, _, num_heads, dim_per_head = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split(
        [int(H_ * W_) for H_, W_ in value_spatial_shapes], dim=1
    )
    # map [0,1] → [-1,1] for grid_sample
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # (bs, H*W, num_heads, dim_per_head)
        # → (bs*num_heads, dim_per_head, H, W)
        value_l_ = (
            value_list[level]
            .flatten(2)              # (bs, H*W, num_heads*dim_per_head)
            .transpose(1, 2)         # (bs, num_heads*dim_per_head, H*W)
            .reshape(bs * num_heads, dim_per_head, int(H_), int(W_))
        )
        # sampling_grids: (bs, num_queries, num_heads, num_levels, num_points, 2)
        # → pick level → (bs, num_queries, num_heads, num_points, 2)
        # → transpose → (bs, num_heads, num_queries, num_points, 2)
        # → flatten(0,1) → (bs*num_heads, num_queries, num_points, 2)
        sampling_grid_l_ = (
            sampling_grids[:, :, :, level]
            .transpose(1, 2)
            .flatten(0, 1)
        )
        # (bs*num_heads, dim_per_head, num_queries, num_points)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)

    # attention_weights: (bs, num_queries, num_heads, num_levels, num_points)
    # → (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = (
        attention_weights
        .transpose(1, 2)
        .reshape(bs * num_heads, 1, num_queries, num_levels * num_points)
    )
    # stack → (bs*num_heads, dim_per_head, num_queries, num_levels, num_points)
    # flatten(-2) → (bs*num_heads, dim_per_head, num_queries, num_levels*num_points)
    output = (
        torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
    ).sum(-1).view(bs, num_heads * dim_per_head, num_queries)

    return output.transpose(1, 2).contiguous()  # (bs, num_queries, embed_dims)


# ---------------------------------------------------------------------------
# StandaloneMHA — mirrors mmcv.cnn.bricks.transformer.MultiheadAttention
# ---------------------------------------------------------------------------

class StandaloneMHA(nn.Module):
    """Thin wrapper around nn.MultiheadAttention with residual + dropout.

    Parameter names match mmcv's MultiheadAttention exactly:
      self.attn           — nn.MultiheadAttention
      self.proj_drop      — nn.Dropout (proj_drop=0.0)
      self.dropout_layer  — nn.Dropout (drop_prob = `dropout` arg)
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dropout: float = 0.0,
        batch_first: bool = False,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        identity: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        key_pos: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None and query_pos is not None:
            if query_pos.shape == key.shape:
                key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        return identity + self.dropout_layer(self.proj_drop(out))


# ---------------------------------------------------------------------------
# StandaloneCustomMSDeformableAttention
# mirrors bevformer/modules/decoder.py::CustomMSDeformableAttention
# ---------------------------------------------------------------------------

class StandaloneCustomMSDeformableAttention(nn.Module):
    """Multi-scale deformable cross-attention (pure PyTorch).

    Parameter names match CustomMSDeformableAttention exactly:
      self.sampling_offsets  — Linear(embed_dims, heads*levels*points*2)
      self.attention_weights — Linear(embed_dims, heads*levels*points)
      self.value_proj        — Linear(embed_dims, embed_dims)
      self.output_proj       — Linear(embed_dims, embed_dims)
      self.dropout           — nn.Dropout
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        dropout: float = 0.1,
        batch_first: bool = False,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        ).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        identity: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        reference_points: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None, # hidden
        level_start_index: torch.Tensor = None, # hidden
        **kwargs,
    ) -> torch.Tensor:
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # (num_query, bs, embed_dims) → (bs, num_query, embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}"
            )

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


# ---------------------------------------------------------------------------
# CUDAMSDeformableAttention
# CUDA-accelerated drop-in replacement for StandaloneCustomMSDeformableAttention.
# Follows TemporalSelfAttention (deformable_transformer.py) in using
# MSDeformAttnFunction.apply() instead of F.grid_sample, giving a faster
# im2col-based CUDA kernel with identical numerical behaviour.
#
# Design differences vs StandaloneCustomMSDeformableAttention:
#   • _reset_parameters  — uses nn.Parameter wrapper (TemporalSelfAttention style)
#   • im2col_step        — controls CUDA kernel tile size (default 64)
#   • mixed_precision    — optionally upcasts fp16 tensors to float32 before the
#                          CUDA kernel (avoids overflow, mirrors TemporalSelfAttention)
#   • level_start_index  — derived from spatial_shapes when not explicitly supplied
#   • NaN guard          — asserts attention_weights contains no NaN after softmax
#
# Weight-layout is identical to StandaloneCustomMSDeformableAttention so the
# two classes can share state-dicts without any key renaming.
# ---------------------------------------------------------------------------

class CUDAMSDeformableAttention(nn.Module):
    """Multi-scale deformable cross-attention backed by the CUDA im2col kernel.

    Parameter names match :class:`StandaloneCustomMSDeformableAttention` exactly,
    so ``cuda_module.load_state_dict(standalone_module.state_dict())`` works
    without any key renaming.

    The forward interface is identical to ``StandaloneCustomMSDeformableAttention``
    (same positional / keyword arguments, same residual ``dropout(out) + identity``
    return convention).  The only behavioural addition is ``mixed_precision``: when
    ``True``, value / sampling_locations / attention_weights are upcast to
    ``float32`` before the kernel call, matching the flag in ``TemporalSelfAttention``.

    Args:
        embed_dims (int):      Token feature dimension. Default: 256.
        num_heads (int):       Number of attention heads. Default: 8.
        num_levels (int):      Number of feature map levels. Default: 1.
        num_points (int):      Sampling points per head per level. Default: 4.
        dropout (float):       Residual dropout probability. Default: 0.1.
        batch_first (bool):    If False tensors are (seq, batch, dim). Default: False.
        im2col_step (int):     Tile size for the CUDA im2col kernel. Default: 64.
        mixed_precision (bool): Upcast to float32 before the kernel. Default: False.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        dropout: float = 0.1,
        batch_first: bool = False,
        im2col_step: int = 64,
        mixed_precision: bool = False,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.batch_first = batch_first
        self.im2col_step = im2col_step
        self.mixed_precision = mixed_precision
        self.dropout = nn.Dropout(dropout)

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        ).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        identity: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        reference_points: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:


        '''
        (Top-down view grid for bev_embed = value) 

         -15         15
      -30 -------------
          |     |     |
          |     |     |
          -------------> x (100 pels)
          |     |     |   
          |     |     |            
      +30 ------v------
                y
             (200 pels)                     
        '''

        # Save the caller's dtype before any modification.  When mixed_precision
        # is enabled we upcast everything to float32 for the full forward pass
        # (including all linear layers and the CUDA kernel), then cast the final
        # output back so the residual add and downstream ops see the original dtype.
        input_dtype = query.dtype

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        # Upcast query and value to float32 BEFORE any linear layer so that fp32
        # module weights can process fp16 inputs.  All subsequent tensors
        # (offsets, weights, sampling_locations) derive from these two, so no
        # further per-tensor casts are needed inside the method.
        if self.mixed_precision: # NOTE : it must be False!!!
            query = query.float()
            value = value.float()

        if not self.batch_first:
            # (num_query, bs, embed_dims) → (bs, num_query, embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        # Derive level_start_index from spatial_shapes when the caller omits it.
        # MSDeformAttnFunction requires it explicitly; the pure-PyTorch fallback does not.
        if level_start_index is None:
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)),  spatial_shapes.prod(1).cumsum(0)[:-1]))

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, self.embed_dims // self.num_heads)

        sampling_offsets = self.sampling_offsets(query).view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, num_query, self.num_heads, self.num_levels, self.num_points)
        assert torch.count_nonzero(torch.isnan(attention_weights)) == 0

        # NOTE : At this point, the last dimension of reference_points is (x, y) = (W, H) = (100, 200)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1) # (W, H) = (x, y) = (100, 200)
            sampling_locations = (reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :])
        elif reference_points.shape[-1] == 4:
            sampling_locations = (reference_points[:, :, None, :, None, :2] + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}")

        # CUDA kernel only supports float32; under AMP, value/sampling_locations/attention_weights may be Half
        if self.mixed_precision:
            value = value.float()
            sampling_locations = sampling_locations.float()
            attention_weights = attention_weights.float()

        output = MSDeformAttnFunction.apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        # Restore caller's dtype so the residual add and any downstream ops
        # see the same dtype as the original input (e.g. fp16 → fp32 → fp16).
        if self.mixed_precision:
            output = output.to(input_dtype)

        return self.dropout(output) + identity


# ---------------------------------------------------------------------------
# StandaloneFFN — mirrors mmcv.cnn.bricks.transformer.FFN
# ---------------------------------------------------------------------------

class StandaloneFFN(nn.Module):
    """Two-layer feed-forward network with residual connection.

    self.layers structure matches mmcv FFN (num_fcs=2) exactly:
      self.layers = Sequential(
          Sequential(Linear(dim, ffn_dim), ReLU(inplace=True), Dropout),  # [0]
          Linear(ffn_dim, dim),                                            # [1]
          Dropout,                                                          # [2]
      )
    This layout means state-dict keys are:
      layers.0.0.weight / bias  — first Linear
      layers.1.weight / bias    — second Linear
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_channels: int = 512,
        ffn_drop: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(embed_dims, feedforward_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(ffn_drop),
            ),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(ffn_drop),
        )
        self.dropout_layer = nn.Identity()
        self.add_identity = True

    def forward(self, x: torch.Tensor, identity: torch.Tensor = None) -> torch.Tensor:
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


# ---------------------------------------------------------------------------
# StandaloneDetrTransformerDecoderLayer
# mirrors mmdet.models.utils.transformer.DetrTransformerDecoderLayer
#       + mmcv.cnn.bricks.transformer.BaseTransformerLayer (forward logic)
# ---------------------------------------------------------------------------

class StandaloneDetrTransformerDecoderLayer(nn.Module):
    """One decoder layer: self-attn → norm → cross-attn → norm → ffn → norm.

    Parameter layout (matches original state-dict):
      attentions[0]  — StandaloneMHA
      attentions[1]  — StandaloneCustomMSDeformableAttention
      ffns[0]        — StandaloneFFN
      norms[0..2]    — nn.LayerNorm
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        ffn_channels: int = 512,
        ffn_drop: float = 0.1,
        attn_drop: float = 0.1,
        num_levels: int = 1,
        num_points: int = 4,
        mixed_precision: bool = False,
        num_vec: int = 50,
        num_pts_per_vec: int = 20,
    ):
        super().__init__()
        self.operation_order = (
            "self_attn", "norm", "self_attn", "norm", "cross_attn", "norm", "ffn", "norm"
        )
        self.pre_norm = False
        self.num_attn = 2
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec

        self.attentions = nn.ModuleList([
            StandaloneMHA(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=0.0,
                dropout=attn_drop,
                batch_first=False,
            ),
            StandaloneMHA(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=0.0,
                dropout=attn_drop,
                batch_first=False,
            ),
            # StandaloneCustomMSDeformableAttention(
            CUDAMSDeformableAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                dropout=attn_drop,
                batch_first=False,
                mixed_precision=mixed_precision,  # kernel is FP32-only; required when training uses AMP
            ),
        ])
        self.ffns = nn.ModuleList([
            StandaloneFFN(embed_dims, ffn_channels, ffn_drop)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dims),
            nn.LayerNorm(embed_dims),
            nn.LayerNorm(embed_dims),
            nn.LayerNorm(embed_dims),
        ])

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        key_pos: torch.Tensor = None,
        attn_masks: torch.Tensor = None,
        query_key_padding_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        **kwargs, # spatial_shapes, level_start_index
    ) -> torch.Tensor:
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        # if attn_masks is None:
        #     attn_masks = [None] * self.num_attn
        # elif isinstance(attn_masks, torch.Tensor):
        #     attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
        # else:
        #     raise ValueError(f'Invalid attn_masks: {attn_masks}')
        if (attn_masks is not None and isinstance(attn_masks, torch.Tensor) == False):
            raise ValueError(f'Invalid attn_masks: {attn_masks}')

        for op in self.operation_order:
            if op == "self_attn":
                if (attn_index == 0):
                    # (num_vec, num_pts_per_vec, n_batch, n_dim) -> (num_vec, num_pts_per_vec*n_batch, n_dim)
                    n_pts, n_batch, n_dim = query.shape
                    query = query.view(self.num_vec, self.num_pts_per_vec, n_batch, n_dim).flatten(1,2)
                    query_pos = query_pos.view(self.num_vec, self.num_pts_per_vec, n_batch, n_dim).flatten(1,2)                    
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query, temp_key, temp_value,
                        identity=None,          # pre_norm=False → pass None, residual handled inside
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks,
                        key_padding_mask=query_key_padding_mask,
                        **kwargs,
                    )
                    query = query.view(self.num_vec, self.num_pts_per_vec, n_batch, n_dim).flatten(0,1)
                    query_pos = query_pos.view(self.num_vec, self.num_pts_per_vec, n_batch, n_dim).flatten(0,1)                    
                    attn_index += 1
                    identity = query
                else:
                    # (num_vec, num_pts_per_vec, n_batch, n_dim) -> (num_pts_per_vec, num_vec*n_batch, n_dim)
                    n_pts, n_batch, n_dim = query.shape
                    query = query.view(self.num_vec, self.num_pts_per_vec, n_batch, n_dim).permute(1,0,2,3).contiguous().flatten(1,2)
                    query_pos = query_pos.view(self.num_vec, self.num_pts_per_vec, n_batch, n_dim).permute(1,0,2,3).contiguous().flatten(1,2)                    
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query, temp_key, temp_value,
                        identity=None,          # pre_norm=False → pass None, residual handled inside
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=None,
                        key_padding_mask=query_key_padding_mask,
                        **kwargs,
                    )
                    query = query.view(self.num_pts_per_vec, self.num_vec, n_batch, n_dim).permute(1,0,2,3).contiguous().flatten(0,1)
                    query_pos = query_pos.view(self.num_pts_per_vec, self.num_vec, n_batch, n_dim).permute(1,0,2,3).contiguous().flatten(0,1)                    
                    attn_index += 1
                    identity = query

            elif op == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif op == "cross_attn":
                query = self.attentions[attn_index](
                    query, key, value, # (dec_queries, None, bev_featurs)
                    identity=None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    # attn_mask=attn_masks[attn_index],
                    attn_mask=None,
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif op == "ffn":
                query = self.ffns[ffn_index](query, identity=None)
                ffn_index += 1

        return query


# ---------------------------------------------------------------------------
# MapTRDecoder — mirrors decoder.py::MapTRDecoder
# ---------------------------------------------------------------------------

class MapTRDecoder(nn.Module):
    """Standalone MapTR decoder (no mmcv/mmdet dependency).

    Identical forward logic to MapTRDecoder.  Parameter layout mirrors the
    original so ``standalone.load_state_dict(original.state_dict())`` works.

    Args:
        num_layers:          Number of decoder layers (default 6).
        embed_dims:          Feature dimension (default 256).
        num_heads:           Attention heads (default 8).
        ffn_channels:        FFN hidden dim (default 512).
        ffn_drop:            FFN dropout probability (default 0.1).
        attn_drop:           Attention dropout probability (default 0.1).
        num_levels:          Number of BEV feature levels (default 1).
        num_points:          Deformable sampling points per head (default 4).
        return_intermediate: Return all layer outputs (default True).
    """

    def __init__(
        self,
        num_layers: int = 6,
        embed_dims: int = 256,
        num_heads: int = 8,
        ffn_channels: int = 512,
        ffn_drop: float = 0.1,
        attn_drop: float = 0.1,
        num_levels: int = 1,
        num_points: int = 4,
        return_intermediate: bool = True,
        mixed_precision: bool = False,
        **kwargs
    ):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_vec = kwargs['num_vec'] # one2one + one2many
        self.num_pts_per_vec = kwargs['num_pts_per_vec']
        self.num_vec_one2one = kwargs['num_vec_one2one'] # one2one
        self.isTrain = kwargs['isTrain']
        num_vec = kwargs['num_vec_one2one'] # one2one
        if self.isTrain:
            num_vec = kwargs['num_vec'] # one2one + one2many

        self.layers = nn.ModuleList([
            StandaloneDetrTransformerDecoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                ffn_channels=ffn_channels,
                ffn_drop=ffn_drop,
                attn_drop=attn_drop,
                num_levels=num_levels,
                num_points=num_points,
                mixed_precision=mixed_precision,
                num_vec=num_vec,
                num_pts_per_vec=self.num_pts_per_vec,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        reference_points: torch.Tensor = None,
        reg_branches: nn.ModuleList = None,
        key_padding_mask: torch.Tensor = None,
        attn_masks: torch.Tensor = None,
        **kwargs, # spatial_shapes, level_start_index
    ):
        """
        Args:
            query:            (num_query, bs, embed_dims) 
            key:              None (not used; kept for API compatibility) -> None
            value:            (num_bev, bs, embed_dims) — BEV feature map
            query_pos:        (num_query, bs, embed_dims) — learnable positional embedding
            reference_points: (bs, num_query, 2) — normalised 2D points - current setting : okay
            reg_branches:     nn.ModuleList or None — iterative refinement heads : okay
            key_padding_mask: (bs, num_bev) bool mask or None - current setting : None
            **kwargs:         spatial_shapes, level_start_index, …


        (Top-down view grid for bev_embed) 

         -15         15
      -30 -------------
          |     |     |
          |     |     |
          -------------> x (100 pels)
          |     |     |   
          |     |     |            
      +30 ------v------
                y
             (200 pels)                
        
        ** How pipeline goes in the original implementation **
        (1) class MapTR() in ...maptr/detectors/maptr.py
        (2) class MapTRHead() in ...maptr/dense_heads/maptr_head.py
        (3) class MapTRPerceptaionTransformer() in ...maptr/modules/transformer.py
        (4) class MapTRDecoder() in ...maptr/modules/decoder.py


        Returns:
            If return_intermediate:
                (intermediate_outputs, intermediate_reference_points)
                each with shape (num_layers, num_query, bs, embed_dims) /
                               (num_layers, bs, num_query, 2)
            Else:
                (output, reference_points)
        """
        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(2)

            output = layer(output,                                   # query
                           value=value,                              # bev_embed
                           query_pos=query_pos,                      # query_pos
                           attn_masks=attn_masks,
                           reference_points=reference_points_input,
                           key_padding_mask=key_padding_mask,
                           **kwargs)
            output = output.permute(1, 0, 2)  # (bs, num_query, embed_dims)

            if reg_branches is not None:
                # Logit space
                offset_preds = reg_branches[lid](output)
                assert reference_points.shape[-1] == 2
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = (offset_preds[..., :2] + inverse_sigmoid(reference_points[..., :2]))

                # Sigmoid space
                new_reference_points = new_reference_points.sigmoid() # to Sigmoid space
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)  # back to (num_query, bs, embed_dims)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points
