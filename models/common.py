import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List
import warnings
import torch.distributed as dist


def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor

def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

class LearnedPositionalEncoding2D(nn.Module):
    """Learned 2D positional encoding for BEV grids.

    This closely follows the common DETR / BEVFormer pattern:

      - Learn a row embedding:  row_embed[y] ∈ R^{num_feats},   y ∈ [0, H-1]
      - Learn a column embedding: col_embed[x] ∈ R^{num_feats}, x ∈ [0, W-1]

      For each spatial location (y, x) we build a 2*num_feats vector:

          pos[y, x] = concat(col_embed[x], row_embed[y])

      and then broadcast it across the batch and reshape to (B, C, H, W).
    """

    def __init__(
        self,
        num_feats: int,
        row_num_embed: int,
        col_num_embed: int,
    ) -> None:
        super().__init__()
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)

        # Match typical DETR initialisation (small uniform range).
        nn.init.uniform_(self.row_embed.weight, -0.1, 0.1)
        nn.init.uniform_(self.col_embed.weight, -0.1, 0.1)

    def forward(self, bev_mask: torch.Tensor) -> torch.Tensor:
        """Compute BEV positional encoding.

        Args:
            bev_mask: (B, H, W) tensor; only its spatial shape is used.

        Returns:
            pos: (B, 2*num_feats, H, W) tensor.
        """
        assert bev_mask.dim() == 3, f"bev_mask must be (B,H,W), got {bev_mask.shape}"
        B, H, W = bev_mask.shape
        device = bev_mask.device

        if H > self.row_num_embed or W > self.col_num_embed:
            raise ValueError(
                f"Mask size {(H, W)} exceeds embedding size "
                f"(row_num_embed={self.row_num_embed}, col_num_embed={self.col_num_embed})"
            )

        # Indices 0..H-1 and 0..W-1
        y = torch.arange(H, device=device)
        x = torch.arange(W, device=device)

        # (H, num_feats) and (W, num_feats)
        row = self.row_embed(y)  # row[y]
        col = self.col_embed(x)  # col[x]

        # Broadcast to (H, W, 2*num_feats):
        #   first half from columns, second half from rows
        #   pos[y, x, :F]   = col[x]
        #   pos[y, x, F:]   = row[y]
        F_dim = self.num_feats
        pos = torch.zeros(H, W, 2 * F_dim, device=device, dtype=row.dtype)
        pos[:, :, :F_dim] = col.unsqueeze(0).expand(H, W, F_dim)
        pos[:, :, F_dim:] = row.unsqueeze(1).expand(H, W, F_dim)

        pos = rearrange(pos, 'h w f -> 1 f h w').repeat(B, 1, 1, 1)

        # # (H, W, 2F) -> (B, 2F, H, W)
        # pos = pos.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        # return pos.contiguous()
        return pos

class Normalize(nn.Module):
    def __init__(self, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

   

# Debugged, (260327)
def bbox_xyxy_to_cxcywh(bbox):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    # x = Width, y = Height
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)
   


def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)] # (x1, y1, x2, y2)
    return torch.cat(bbox_new, dim=-1)

# Debugged, (260327)
def normalize_2d_bbox(bboxes, pc_range):
    '''
    ** bboxes come in (xyxy) format. **
    ** points come in (x, y) = (W, H) = (100, 200) format. **
    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]  
    '''

    patch_h = pc_range[4] - pc_range[1] 
    patch_w = pc_range[3] - pc_range[0] 
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes) # Convert (xyxy) to (cx, cy, w, h)
    cxcywh_bboxes[...,0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0] # x -> width
    cxcywh_bboxes[...,1:2] = cxcywh_bboxes[...,1:2] - pc_range[1] # y -> height
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h]) # (w, h, w, h)

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def denormalize_2d_bbox(bboxes, pc_range):

    '''
    ** bboxes come in (cxcywh) format. **
    bboxes : num_bboxes 4
    pc_range :  [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
                [-H/2,  -W/2,  -z,   H/2,  W/2,    z]
    
    '''

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2]*(pc_range[4] - pc_range[1]) + pc_range[1]) # x -> width
    bboxes[..., 1::2] = (bboxes[..., 1::2]*(pc_range[3] - pc_range[0]) + pc_range[0]) # y -> height

    return bboxes

# Debugged, (260327)
def normalize_2d_pts(pts, pc_range):
    '''
    ** points come in (x, y) = (W, H) = (100, 200) format. **
    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]  
    '''

    patch_h = pc_range[4] - pc_range[1] 
    patch_w = pc_range[3] - pc_range[0] 
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0] # x -> width 
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]  # y -> height
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

# Debugged, (260327)
def denormalize_2d_pts(pts, pc_range):
    '''
    ** points come in (x, y) = (W, H) = (100, 200) format. **
    pc_range : [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]      
    '''

    patch_h = pc_range[4] - pc_range[1] 
    patch_w = pc_range[3] - pc_range[0] 

    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*patch_w + pc_range[0]) # x -> width
    new_pts[...,1:2] = (pts[...,1:2]*patch_h + pc_range[1]) # y -> height
    return new_pts


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords