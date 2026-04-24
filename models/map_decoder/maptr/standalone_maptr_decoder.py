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
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
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
    ):
        super().__init__()
        self.operation_order = (
            "self_attn", "norm", "cross_attn", "norm", "ffn", "norm"
        )
        self.pre_norm = False
        self.num_attn = 2

        self.attentions = nn.ModuleList([
            StandaloneMHA(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=0.0,
                dropout=attn_drop,
                batch_first=False,
            ),
            StandaloneCustomMSDeformableAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                dropout=attn_drop,
                batch_first=False,
            ),
        ])
        self.ffns = nn.ModuleList([
            StandaloneFFN(embed_dims, ffn_channels, ffn_drop)
        ])
        self.norms = nn.ModuleList([
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
        attn_masks=None,
        query_key_padding_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        if attn_masks is None:
            attn_masks = [None] * self.num_attn

        for op in self.operation_order:
            if op == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query, temp_key, temp_value,
                    identity=None,          # pre_norm=False → pass None, residual handled inside
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif op == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif op == "cross_attn":
                query = self.attentions[attn_index](
                    query, key, value,
                    identity=None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
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
# StandaloneMapTRDecoder — mirrors decoder.py::MapTRDecoder
# ---------------------------------------------------------------------------

class StandaloneMapTRDecoder(nn.Module):
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
    ):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

        self.layers = nn.ModuleList([
            StandaloneDetrTransformerDecoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                ffn_channels=ffn_channels,
                ffn_drop=ffn_drop,
                attn_drop=attn_drop,
                num_levels=num_levels,
                num_points=num_points,
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
        **kwargs,
    ):
        """
        Args:
            query:            (num_query, bs, embed_dims)
            key:              None (not used; kept for API compatibility)
            value:            (num_bev, bs, embed_dims) — BEV feature map
            query_pos:        (num_query, bs, embed_dims) — positional encoding
            reference_points: (bs, num_query, 2) — normalised 2D points
            reg_branches:     nn.ModuleList or None — iterative refinement heads
            key_padding_mask: (bs, num_bev) bool mask or None
            **kwargs:         spatial_shapes, level_start_index, …

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
            # (bs, num_query, 2) → (bs, num_query, num_levels=1, 2)
            reference_points_input = reference_points[..., :2].unsqueeze(2)

            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            output = output.permute(1, 0, 2)  # (bs, num_query, embed_dims)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 2
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = (
                    tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                )
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)  # back to (num_query, bs, embed_dims)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points
