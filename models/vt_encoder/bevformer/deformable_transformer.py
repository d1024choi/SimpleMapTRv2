import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from einops import rearrange, repeat
# from util.misc import inverse_sigmoid
from models.vt_encoder.bevformer.ops.modules import MSDeformAttn, MSDeformAttn3D
from models.vt_encoder.bevformer.ops.functions import MSDeformAttnFunction

import warnings
warnings.filterwarnings(action='ignore')

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TemporalSelfAttention(nn.Module):
    def __init__(self, h=200, w=200, d_model=256, n_levels=1, n_heads=8, n_points=4, mixed_precision=False, m=None, **kwargs):
        super().__init__()

        self.h, self.w = h, w
        self.m = m if m is not None else h  # Default to h if not provided

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.mixed_precision = mixed_precision

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        # OK, 230605 ---
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self,
                query,
                query_pos,
                reference_points,
                input_flatten,
                input_padding_mask=None):

        """
        query : b num_queries dim
        query_pos : b num_queries dim
        reference_points : b num_queries 1 2
        input_flatten : b num_inputs dim (=query)
        input_padding_mask : b num_inputs

        Note : the difference from the original implementation is that the original uses [query, query] as value
        """

        if (query_pos is not None):
            query = query + query_pos

        # Handle both spatial shapes: (h, w) for original queries and (m, m) for global queries
        if self.m is not None and self.m != self.h:
            # Two spatial levels: original (h*w) and global (m*m)
            input_spatial_shapes = torch.as_tensor([(self.h, self.w), (self.m, self.m)], dtype=torch.long, device=query.device)  # 2 levels, 2 dims
            input_level_start_index = torch.cat((input_spatial_shapes.new_zeros((1,)),
                                                 input_spatial_shapes.prod(1).cumsum(0)[:-1]))  # [0, h*w]
        else:
            # Single spatial level: only (h, w)
            input_spatial_shapes = torch.as_tensor([(self.h, self.w)], dtype=torch.long, device=query.device)  # 1 level, 2 dims
            input_level_start_index = torch.cat((input_spatial_shapes.new_zeros((1,)),
                                                 input_spatial_shapes.prod(1).cumsum(0)[:-1]))  # [0]

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        assert torch.count_nonzero(torch.isnan(attention_weights)) == 0

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        else:
            raise ValueError(
                'Last dim of reference_points must be 2, but get {} instead.'.format(reference_points.shape[-1]))

        if (self.mixed_precision):
            value = value.float()
            sampling_locations = sampling_locations.float()
            attention_weights = attention_weights.float()

        output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations,
                                            attention_weights, self.im2col_step)

        output = self.output_proj(output)
        return output


class TemporalSelfAttention_update(nn.Module):
    """
    Updated TemporalSelfAttention that mirrors the original BEVFormer logic:

    - Uses a BEV queue of length `num_bev_queue` (default 2).
    - Builds offsets/weights from concatenated [history_bev, current_bev] features.
    - Runs deformable attention on the queued BEVs and fuses queue outputs by mean.

    Note: This module includes the residual connection internally (like the original MMCV
    implementation). If you wire it into `DeformableTransformerEncoderLayer`, avoid adding
    another residual outside this module.
    """

    def __init__(
        self,
        h: int = 200,
        w: int = 200,
        d_model: int = 256,
        n_levels: int = 1,
        n_heads: int = 8,
        n_points: int = 4,
        num_bev_queue: int = 2,
        im2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = True,
        mixed_precision: bool = False,
        **kwargs,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}"
            )

        self.h = h
        self.w = w
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.num_bev_queue = num_bev_queue
        self.im2col_step = im2col_step
        self.batch_first = batch_first
        self.mixed_precision = mixed_precision

        self.dropout = nn.Dropout(dropout)

        self.sampling_offsets = nn.Linear(
            d_model * num_bev_queue,
            num_bev_queue * n_heads * n_levels * n_points * 2,
        )
        self.attention_weights = nn.Linear(
            d_model * num_bev_queue,
            num_bev_queue * n_heads * n_levels * n_points,
        )
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    @staticmethod
    def _is_power_of_2(value: int) -> bool:
        if (not isinstance(value, int)) or value < 0:
            raise ValueError(
                f"invalid input for _is_power_of_2: {value} (type: {type(value)})"
            )
        return (value & (value - 1) == 0) and value != 0

    def _reset_parameters(self) -> None:
        constant_(self.sampling_offsets.weight.data, 0.0)

        dim_per_head = self.d_model // self.n_heads
        if not self._is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set d_model so that each attention head dimension "
                "is a power of 2 for better CUDA efficiency."
            )

        thetas = (
            torch.arange(self.n_heads, dtype=torch.float32)
            * (2.0 * math.pi / self.n_heads)
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            grid_init
            / grid_init.abs().max(-1, keepdim=True)[0]
        ).view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels * self.num_bev_queue, self.n_points, 1
        )

        for point_index in range(self.n_points):
            grid_init[:, :, point_index, :] *= point_index + 1

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
        query: Tensor,
        query_pos: Optional[Tensor],
        reference_points: Tensor,
        input_flatten: Optional[Tensor] = None,
        input_padding_mask: Optional[Tensor] = None,
        identity: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            query: (bs, num_query, d_model) if batch_first else (num_query, bs, d_model)
            query_pos: positional encoding with same shape as query (optional)
            reference_points: (bs, num_query, n_levels, 2) normalized in [0,1]
            input_flatten: (bs, num_value, d_model) tokens to be used as BEV value
            input_padding_mask: (bs, num_value) optional mask
            identity: residual tensor; defaults to original `query`
            value: optional queued BEV value tensor. If None, will build a
                   queue by duplicating current value/query.
        Returns:
            Tensor: (bs, num_query, d_model) if batch_first else (num_query, bs, d_model)
        """

        if input_flatten is None:
            input_flatten = query

        if identity is None:
            identity = query

        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            identity = identity.permute(1, 0, 2)
            input_flatten = input_flatten.permute(1, 0, 2)
            if value is not None:
                value = value.permute(1, 0, 2)

        bs, num_query, embed_dims = query.shape

        if value is None:
            if self.num_bev_queue != 2:
                raise ValueError(
                    f"Default queue construction is only implemented for num_bev_queue=2, "
                    f"but got {self.num_bev_queue}"
                )
            _, num_value, _ = input_flatten.shape
            value = (torch.stack([input_flatten, input_flatten], dim=1).reshape(bs * self.num_bev_queue, num_value, embed_dims))
        else:
            _, num_value, _ = value.shape

        # BEVFormer duplicates ref_2d across the BEV queue dimension (bs -> bs*num_bev_queue),
        # so once we reshape offsets/weights to (bs*num_bev_queue, ...) the shapes still align.
        if reference_points.size(0) == bs:
            reference_points = (
                torch.stack([reference_points] * self.num_bev_queue, dim=1)
                .reshape(bs * self.num_bev_queue, *reference_points.shape[1:])
            )

        if input_padding_mask is not None and input_padding_mask.size(0) == bs:
            input_padding_mask = (
                torch.stack([input_padding_mask] * self.num_bev_queue, dim=1)
                .reshape(bs * self.num_bev_queue, *input_padding_mask.shape[1:])
            )

        input_spatial_shapes = torch.as_tensor([(self.h, self.w)], dtype=torch.long, device=query.device)
        level_start_index = torch.cat(
            (
                input_spatial_shapes.new_zeros((1,)),
                input_spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == num_value

        history_bev = value[:bs]
        query_with_history = torch.cat([history_bev, query], dim=-1)

        projected_value = self.value_proj(value)
        if input_padding_mask is not None:
            projected_value = projected_value.masked_fill(input_padding_mask[..., None], float(0.0))

        projected_value = projected_value.view(
            bs * self.num_bev_queue,
            num_value,
            self.n_heads,
            self.d_model // self.n_heads,
        )

        sampling_offsets = self.sampling_offsets(query_with_history).view(
            bs,
            num_query,
            self.n_heads,
            self.num_bev_queue,
            self.n_levels,
            self.n_points,
            2,
        )
        attention_weights = self.attention_weights(query_with_history).view(
            bs,
            num_query,
            self.n_heads,
            self.num_bev_queue,
            self.n_levels * self.n_points,
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs,
            num_query,
            self.n_heads,
            self.num_bev_queue,
            self.n_levels,
            self.n_points,
        )

        attention_weights = (
            attention_weights.permute(0, 3, 1, 2, 4, 5)
            .reshape(
                bs * self.num_bev_queue,
                num_query,
                self.n_heads,
                self.n_levels,
                self.n_points,
            )
            .contiguous()
        )
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
            bs * self.num_bev_queue,
            num_query,
            self.n_heads,
            self.n_levels,
            self.n_points,
            2,
        )

        if reference_points.shape[-1] != 2:
            raise ValueError(
                f"Last dim of reference_points must be 2, but got {reference_points.shape[-1]}"
            )

        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        sampling_locations = (reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :])

        if self.mixed_precision:
            projected_value = projected_value.float()
            sampling_locations = sampling_locations.float()
            attention_weights = attention_weights.float()

        output = MSDeformAttnFunction.apply(
            projected_value,
            input_spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )

        output = output.permute(1, 2, 0)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue).mean(-1)
        output = output.permute(2, 0, 1)
        output = self.output_proj(output)

        # ----------------------------------------------------------
        # NOTE : this part will be performed outside this module.
        # output = self.dropout(output) + identity
        # if not self.batch_first:
        #     output = output.permute(1, 0, 2)

        return output


class SpatialCrossAttention(nn.Module):
    def __init__(self, h=200, w=200, num_points_in_pillar=4, d_model=256, n_levels=1, n_heads=8, n_points=4, mixed_precision=False, scale=1, **kwargs):
        super().__init__()

        self.h, self.w = h, w
        self.num_points_in_pillar = num_points_in_pillar

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.mixed_precision = mixed_precision

        self.cross_attn = MSDeformAttn3D(d_model, n_levels, n_heads, 2 * n_points, mixed_precision)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, queries, embeds, features, reference_points_3D, bev_mask):
        '''
        queries : b n q_len d (where q_len can be (h*w) or ((h*w)+(m*m)) when global queries are present)
        features : [(b*n c h' w')]
        embeds : [pos_emb, lvl_emb, cam_emb]
           pos_emb : b n q_len d
           lvl_emb : lvl d
           cam_emb : n_cam d
        reference_points_3d : b n q_len D 2
        bev_mask : b n q_len D 2
        '''

        b, n, q_len, dim = queries.size()
        pos_emb, lvl_emb, cam_emb = embeds[0], embeds[1], embeds[2],

        # reshape input features
        input_flatten, mask_flatten, spatial_shapes = [], [], []
        for l in range(len(features)):

            h, w = features[l].size(-2), features[l].size(-1)
            spatial_shapes.append((h, w))

            # (b n) c h w -> b n c (h w)
            input = rearrange(features[l], '(b n) c h w -> b n c (h w)', b=b, n=n)

            # note : according to original implementation
            if (lvl_emb is not None):
                input = input + lvl_emb[l][None, None, :, None]

            # note : according to original implementation
            if (cam_emb is not None):
                input = input + cam_emb[None, :, :, None]

            # b n c (h w) -> (b n) (h w) c
            input = rearrange(input, 'b n c l -> (b n) l c')
            input_flatten.append(input)

            # (b n) (h w) 1
            mask = torch.zeros(size=(b*n, h*w)).bool().to(input.device)
            mask_flatten.append(mask)

        input_flatten = torch.cat(input_flatten, dim=1)     # (b n) (h' w') c
        mask_flatten = torch.cat(mask_flatten, 1)           # (b n) (h' w')
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=queries.device)      # lvl 2
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # lvl


        # Note, 230605 : assume that batch_size=1
        indexes = []
        for i in range(bev_mask.size(1)): # b n q_len D 2
            # mask for i-th camera image
            mask_per_img = bev_mask[:, i, ..., 0]  # b q_len D

            # if at least one depth falls into the image, the corresponding query is used.
            # This works for both original queries and concatenated queries (with global queries)
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)  # b q_len D -> q_len D -> q_len
            indexes.append(index_query_per_img)
        max_q_len = max([len(each) for each in indexes])

        # create new queries and reference_points of size max_q_len
        # Note, 230605 : it seems that 'pos_emb' is not used in the original implementation
        queries_pos = queries
        if (pos_emb is not None): queries_pos = queries_pos + pos_emb
        queries_rebatch = queries.new_zeros([b, n, max_q_len, self.d_model])  # b n l dim
        reference_points_rebatch = reference_points_3D.new_zeros([b, n, max_q_len, self.num_points_in_pillar, 2])  # b n l D 2
        for j in range(b):
            for i in range(n):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = queries_pos[j, i, index_query_per_img]

                reference_points_per_img = reference_points_3D[j, i]  # q_len D 2
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[
                    index_query_per_img]

        queries_rebatch = rearrange(queries_rebatch, 'b n l d -> (b n) l d')
        reference_points_rebatch = rearrange(reference_points_rebatch, 'b n l d c -> (b n) l d c')
        queries_attn = self.cross_attn(queries_rebatch, reference_points_rebatch, input_flatten, spatial_shapes,
                                 level_start_index, mask_flatten)
        queries_attn = rearrange(queries_attn, '(b n) l d -> b n l d', b=b, n=n)

        # re-store
        slots = torch.zeros_like(queries)  # b n q_len d       
        for b_idx in range(b):
            for c_idx, index_query_per_img in enumerate(indexes):
                slots[b_idx, c_idx, index_query_per_img] = queries_attn[b_idx, c_idx, :len(index_query_per_img)]
        slots = slots.sum(dim=1)    # b q_len d

        # if at least one depth falls into the image, the corresponding query is used.
        # ex) a query has D depths, and all pulled values from the D depths are weighted averaged by attention weights.
        # Therefore, (bev_mask[..., 0].sum(-1) > 0)
        # from (b, n_cam, q_len, D) to (b, q_len)
        count = (bev_mask[..., 0].sum(-1) > 0).sum(1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None] # (b, q_len, d) / (b, q_len, cnt)

        return self.output_proj(slots)


class DeformableTransformerEncoderLayer(nn.Module):
    '''
    (260320) This version is exactly the same as the official implementation.
    '''
    def __init__(self, cfg):
        super().__init__()

        self.n_cam, self.z_candi = cfg['n_cam'], cfg['z_candi']
        self.h, self.w = cfg['h'], cfg['w']
        dropout, d_model, d_ffn, activation = cfg['dropout'], cfg['d_model'], cfg['d_ffn'], cfg['activation']

        # -----------------------------------------------------
        # self attention
        
        # The original implementation that reflects temporal context
        self.self_attn = TemporalSelfAttention_update(**cfg) 
        
        # # A re-implementation that does not reflect temporal context
        # # Set n_levels to 2 if we have global queries (m != h), otherwise 1
        # m = cfg.get('m', None)
        # n_levels_self_attn = 2 if (m is not None and m != self.h) else 1
        # cfg.update({"n_levels": n_levels_self_attn})        
        # self.debug_temp_attn_type = cfg.get('debug_temp_attn_type', 'ori')
        # if self.debug_temp_attn_type == 'my':
        #     self.self_attn = TemporalSelfAttention(**cfg) 

        # -----------------------------------------------------
        # cross attention
        self.cross_attn = SpatialCrossAttention(**cfg)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.dropout0 = nn.Dropout(dropout)
        self.norm0 = nn.LayerNorm(d_model)

        # -----------------------------------------------------
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight.data)
        constant_(self.linear1.bias.data, 0.)
        xavier_uniform_(self.linear2.weight.data)
        constant_(self.linear2.bias.data, 0.)


    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, queries, embeds, reference_points_cam, reference_points_bev, features, bev_mask):
        '''
        queries : b (h w) d
        features : [(b*n c h' w')]
        embeds : [pos_emb, level_emb, cam_emb]
           pos_emb : b (h w) d
           level_emb : lvl d
           cam_emb : n_cam d
        reference_points_cam : b n (h w) D 2
        bev_mask : b n (h w) D 2
        reference_points_bev : b (h w) 2
        '''

        b, q_len, d = queries.size()
        pos_emb, lvl_emb, cam_emb = embeds[0], embeds[1], embeds[2]

        # self-attention + (add, norm)
        reference_points_bev = rearrange(reference_points_bev, 'b l d -> b l 1 d')
        queries_out = self.self_attn(query=queries,
                                     query_pos=pos_emb,
                                     reference_points=reference_points_bev,
                                     input_flatten=queries)
        queries = queries + self.dropout0(queries_out) # refine queries with attention result
        queries = self.norm0(queries) # b q_len d


        # cross-attention + (proj, add, norm)
        # NOTE : the original implementation does not use pos_emb
        queries_repeat = queries[:, None].repeat(1, self.n_cam, 1, 1) # b n (h w) d
        queries_out = self.cross_attn(queries_repeat, [None, lvl_emb, cam_emb],
                                      features, reference_points_cam, bev_mask)
        queries = queries + self.dropout1(queries_out) # refine queries with attention result
        queries = self.norm1(queries) # b q_len d

        # ffn (linear, add, norm)
        return self.forward_ffn(queries.view(-1, d)).view(b, q_len, d)


def main():
    N, M, D = 1, 2, 2
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
    level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    S = sum([(H * W).item() for H, W in shapes])

    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2


if __name__ == '__main__':
    main()