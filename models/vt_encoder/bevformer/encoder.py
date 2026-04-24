import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from models.vt_encoder.bevformer.ops.modules import MSDeformAttn, MSDeformAttn3D
from models.vt_encoder.bevformer.deformable_transformer import DeformableTransformerEncoderLayer
from models.common import LearnedPositionalEncoding2D, xavier_init

def generate_grid(height: int, width: int):
    '''
    F.pad : to pad the last 3 dimensions, use (left, right, top, bottom, front, back)
    For example,
       x = torch.zeros(size=(2, 3, 4))
       x = F.pad(x, (a, b, c, d, e, f), value=1)
       x.size() # 2+e+f x 3+c+d x 4+a+b

    ------------> x (width)
    |
    |
    v
    y (height)

    '''
    # note : original paper implementation
    xs = (torch.linspace(0.5, width - 0.5, width) / width)
    ys = (torch.linspace(0.5, height - 0.5, height) / height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class GenerateGrid(nn.Module):
    '''
    Generate 2D and 3D grid points in the ego centric coordinate systems
    '''
    def __init__(
        self,
        dim,
        sigma,
        bev_height,
        bev_width,
        h_meters,
        w_meters,
        offset,
        pc_range,
        num_points_in_pillar,
        z_candi=[0.0, 1.0, 2.0, 3.0, 4.0],
        **kwargs
    ):
        super().__init__()

        '''
        BEV grid points (top-down view)
          |
        --------------> x (width)
          |
          |
          v
        y (height)
        '''

        # the size of the BEV grid
        h, w = bev_height, bev_width


        # Original reference_points_cam calculation ----------------------------------
        z = pc_range[5] - pc_range[2]
        zs = torch.linspace(0.5, z - 0.5, num_points_in_pillar).view(-1, 1, 1).expand(num_points_in_pillar, h, w) / z
        xs = torch.linspace(0.5, w - 0.5, w).view(1, 1, w).expand(num_points_in_pillar, h, w) / w
        ys = torch.linspace(0.5, h - 0.5, h).view(1, h, 1).expand(num_points_in_pillar, h, w) / h
        ref_3d = torch.stack((xs, ys, zs), -1) # (num_points_in_pillar, H, W, 3)
        ref_3d = rearrange(ref_3d, 'd h w c -> d c h w') # (num_points_in_pillar, 3, H, W)        
        
        # ref_3d[:, 0:1] = ref_3d[:, 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0] # width (-15~15)
        # ref_3d[:, 1:2] = ref_3d[:, 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1] # height (30~30)
        # ref_3d[:, 2:3] = ref_3d[:, 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2] # up (-2~2)
                 
        # reference_points = rearrange(ref_3d, 'd h w c -> d c h w') # (num_points_in_pillar, 3, H, W)
        # reference_points[:, 0:1] = reference_points[:, 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0] # width (-15~15)
        # reference_points[:, 1:2] = reference_points[:, 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1] # height (30~30)
        # reference_points[:, 2:3] = reference_points[:, 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2] # up (-2~2)

        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, h - 0.5, h), torch.linspace(0.5, w - 0.5, w))
        ref_x = ref_x.reshape(-1) / w
        ref_y = ref_y.reshape(-1) / h
        ref_2d = torch.stack((ref_x, ref_y), -1).view(h, w, 2) # (h*w, 2) -> (h, w, 2)
        ref_2d = rearrange(ref_2d, 'h w c -> c h w') # (2, h, w)

        self.register_buffer('grid_2d_normalized', ref_2d, persistent=False)              # 2 h w
        self.register_buffer('grid_3d_normalized', ref_3d, persistent=False)              # D 3 h w        
        # ----------------------------------------------------------------------------


        # New reference_points calculation ----------------------------------------
        # ** This doesn't match the official implementation **
        if False:
            # the BEV grid points in the pixel coordinate system
            grid_2D = generate_grid(h, w).squeeze(0) # 3 h w (values from 0 to 1), ** This part matches the official implementation, ref_2d **
            grid_2D[0] = bev_width * grid_2D[0] 
            grid_2D[1] = bev_height * grid_2D[1]     # (width, height, 1) x h x w

            # the BEV grid points in the ego centric coordinate system 
            V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
            V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
            grid_2D = V_inv @ rearrange(grid_2D, 'd h w -> d (h w)')                # 3 (h w)
            grid_2D = rearrange(grid_2D, 'd (h w) -> d h w', h=h, w=w)              # 3 h w -> (height, width, 1) x h x w

            # the 3D grid points in the ego centric coordinate system (add z_candi to the 2D grid points)
            z = torch.as_tensor(z_candi, dtype=grid_2D.dtype, device=grid_2D.device)  # D
            grid_xy = grid_2D[:2].unsqueeze(0).expand(z.numel(), -1, -1, -1)          # D 2 h w
            z_axis = z.view(-1, 1, 1, 1).expand(-1, 1, h, w)                          # D 1 h w
            grid_3D = torch.cat((grid_xy, z_axis), dim=1)                             # D 3 h w -> D x (height, width, up) x h x w

            self.register_buffer('grid_2d', grid_2D, persistent=False)              # 3 h w
            self.register_buffer('grid_3d', grid_3D, persistent=False)              # D 3 h w


class ReferencePoints(nn.Module):
    def __init__(self, image_h, image_w, pc_range, depth_thr=1.0, **kwargs):
        super().__init__()

        self.h, self.w = image_h, image_w
        self.depth_thr = depth_thr
        self.pc_range = pc_range

    def __call__(self, grid_3d_normalized, I, E, bev_aug, normalize=True):
        '''
        grid_3d_normalized : D 3 h w -> D x (width, height, up) x h x w
        I : b n 3 3
        E : b n 4 4
        bev_aug : b n 4 4

        image_2D (mask) : b n D 2 (h w)

        '''

        b, n, _, _ = I.size()

        # Original reference_points_cam calculation 
        lidar2img = I @ E[:, :, :3, :]
        bottom = lidar2img.new_tensor([0.0, 0.0, 0.0, 1.0]).view(1, 1, 1, 4).expand(b, n, 1, 4)
        lidar2img = torch.cat([lidar2img, bottom], dim=2) # b n 4 4

        reference_points = grid_3d_normalized.clone()
        reference_points[:, 0:1] = reference_points[:, 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0] # width (-15~15)
        reference_points[:, 1:2] = reference_points[:, 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1] # height (30~30)
        reference_points[:, 2:3] = reference_points[:, 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2] # up (-2~2)

        reference_points = rearrange(reference_points, 'd c h w -> d (h w) c')[None].repeat(b, 1, 1, 1)
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= self.w
        reference_points_cam[..., 1] /= self.h

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        try:
            bev_mask = torch.nan_to_num(bev_mask)
        except:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = rearrange(reference_points_cam, 'd b c l p -> b c l d p')            # (B, n_cam, num_q, D, 2)
        bev_mask = rearrange(bev_mask, 'd b c l p -> b c l d p').expand(-1, -1, -1, -1, 2)          # (B, n_cam, num_q, D, 2)


        # Visualization --------------------------------
        # Compare lidar2img path (reference_points_cam, 800×480 norm) vs E@I path (ref_pts_3D, self.w×self.h norm).
        # Batch index 0 only. Enable with:  export BEV_REF_VIZ=1
        # Optional output path:  export BEV_REF_VIZ_OUT=/tmp/bev_ref_compare.png
        if False:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            b0 = 0
            n_cam = reference_points_cam.size(1)
            num_q = reference_points_cam.size(2)
            num_d = reference_points_cam.size(3)
            stride = max(1, num_q // 12000)

            rpc = reference_points_cam[b0].detach().float().cpu().numpy()
            rp3 = reference_points_cam[b0].detach().float().cpu().numpy()
            m_cam = (bev_mask[b0].detach().float().cpu().numpy() > 0.5)
            m_r = (bev_mask[b0].detach().float().cpu().numpy() > 0.5)

            fig_h = max(4.0, 2.6 * n_cam)
            fig, axes = plt.subplots(n_cam, 2, figsize=(10.0, fig_h), squeeze=False)
            wh_rpc = (800.0, 480.0)
            wh_r3 = (float(self.w), float(self.h))

            for ci in range(n_cam):
                ax_l, ax_r = axes[ci, 0], axes[ci, 1]
                idx = np.arange(0, num_q, stride)
                for d in range(num_d):
                    col = plt.cm.tab10(d % 10)
                    u0 = rpc[ci, idx, d, 0] * wh_rpc[0]
                    v0 = rpc[ci, idx, d, 1] * wh_rpc[1]
                    ok0 = m_cam[ci, idx, d, 0]
                    u1 = rp3[ci, idx, d, 0] * wh_r3[0]
                    v1 = rp3[ci, idx, d, 1] * wh_r3[1]
                    ok1 = m_r[ci, idx, d, 0]
                    ax_l.scatter(u0[ok0], v0[ok0], s=2, alpha=0.4, color=col, label=f'z{d}' if ci == 0 else None)
                    ax_r.scatter(u1[ok1], v1[ok1], s=2, alpha=0.4, color=col, label=f'z{d}' if ci == 0 else None)

                ax_l.set_xlim(0, wh_rpc[0])
                ax_l.set_ylim(wh_rpc[1], 0)
                ax_l.set_aspect('equal')
                ax_l.set_xlabel('u (px)')
                ax_l.set_ylabel('v (px)')
                ax_l.set_title(f'cam {ci}  reference_points_cam\n(lidar2img, norm ÷ {int(wh_rpc[0])}×{int(wh_rpc[1])})')

                ax_r.set_xlim(0, wh_r3[0])
                ax_r.set_ylim(wh_r3[1], 0)
                ax_r.set_aspect('equal')
                ax_r.set_xlabel('u (px)')
                ax_r.set_ylabel('v (px)')
                ax_r.set_title(f'cam {ci}  ref_pts_3D\n(E @ I, norm ÷ {int(wh_r3[0])}×{int(wh_r3[1])})')

            axes[0, 0].legend(loc='upper right', fontsize=7, markerscale=3)
            fig.suptitle('BEV ref. projection compare (batch item 0)', fontsize=11)
            plt.tight_layout()
            out_path = os.environ.get('BEV_REF_VIZ_OUT', './bev_ref_compare_b0.png')
            plt.savefig(out_path, dpi=130, bbox_inches='tight')
            plt.close()
        # Visualization --------------------------------


        return reference_points_cam, bev_mask



class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, h, w, device=None):

        mask = torch.ones(size=(1, h, w)).to(device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale # 0 to 2*pi
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t # sin(2 * pi / T)
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class BEVFormerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # self.cfg = cfg
        # cfg_bf, cfg_bev = cfg[model_name], cfg['BEV']

        self.bs = cfg['bs']
        self.n_cam = cfg['n_cam']        
        self.h_dim = cfg['dim']
        self.n_lvl = cfg['n_lvl']
        # self.z_candi = cfg['z_candi']
        self.h = cfg['h']
        self.w = cfg['w']
        self.m = None
        self.mixed_precision = cfg.get('mixed_precision', False)  # True so AMP (Half) inputs are cast before CUDA kernel
        self.len_can_bus = cfg.get('len_can_bus', 18)
        self.use_can_bus = cfg.get('use_can_bus', True)
        self.position_embedding_type = cfg.get('position_embedding_type', 'learned')

        # (** confirmed **) Embeddings 
        self.queries = nn.Embedding(self.h * self.w, self.h_dim)

        # (** confirmed **) Can Bus MLP
        self.can_bus_mlp = None
        if self.use_can_bus:
            self.can_bus_mlp = nn.Sequential(
                nn.Linear(self.len_can_bus, self.h_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.h_dim // 2, self.h_dim),
                nn.ReLU(inplace=True),
            )
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.h_dim))

        # Generate 2D and 3D grid points in the ego centric coordinate systems
        # (** confirmed **) grid_2D and grid_3D generation process matche the official implementation
        self.grids = GenerateGrid(**cfg)
        
        # 3D Reference points projected into camera images
        # Confirmed : the same as the official implementation
        self.GetReferencePoints = ReferencePoints(**cfg)

        # Query Position Embedding
        if self.position_embedding_type == 'sine':
            self.position_embedding = PositionEmbeddingSine(self.h_dim // 2, normalize=True)
        elif self.position_embedding_type == 'learned':
            # Confirmed : the same as the official implementation
            self.position_embedding = LearnedPositionalEncoding2D(self.h_dim // 2, self.h, self.w)
        else:
            raise ValueError(f"Invalid position embedding type: {self.position_embedding_type}")
        
        # Feature level and Camera Embeddings
        self.level_embed = nn.Parameter(torch.Tensor(self.n_lvl, self.h_dim)) # multi-scale feat. map levels
        self.cams_embed = nn.Parameter(torch.Tensor(self.n_cam, self.h_dim)) # multiple cameras
        
        n_heads = 8
        n_points = 4
        cfg.update({"d_model": self.h_dim, "d_ffn": 1024, "dropout": 0.1, "activation": "relu",
                            "n_levels": self.n_lvl, "n_heads": n_heads, "n_points": n_points,
                            "mixed_precision": self.mixed_precision, "m": self.m})
        encoder = DeformableTransformerEncoderLayer(cfg)
        self.DeformAttnEnc = nn.ModuleList([copy.deepcopy(encoder) for _ in range(cfg['repeat'])])

        # Initialize
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize the transformer weights."""
        # TODO : Check if this is necessary
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        # for m in self.modules():
        #     if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) or isinstance(m, CustomMSDeformableAttention):
        #         try:
        #             m.init_weight()
        #         except AttributeError:
        #             m.init_weights()

        xavier_uniform_(self.queries.weight)
        normal_(self.level_embed)
        normal_(self.cams_embed)        
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    def forward(self, features, intrinsics, extrinsics, bev_aug, can_bus):
        '''
        features : [((b n) c h' w')]
        intrinsics : b n 3 3
        extrinsics : b n 4 4
        bev_aug : b n 4 4
        can_bus : b can_bus_len
        '''

        b = intrinsics.size(0)

        queries = self.queries.weight.unsqueeze(0).repeat(b, 1, 1)                         # b (h*w) d
        queries_g = None


        # Can Bus MLP
        if self.can_bus_mlp is not None:
            can_bus = self.can_bus_mlp(can_bus)[:, None, :]
            queries = queries + can_bus

        # reference points in BEV and Image (bev_mask denotes the ref_pts that is outside the image area)    
        reference_points_bev = rearrange(self.grids.grid_2d_normalized, 'c h w -> 1 (h w) c') # 1 (h w) 2
        reference_points_cam, bev_mask = self.GetReferencePoints(self.grids.grid_3d_normalized, intrinsics, extrinsics, bev_aug) # b n (h w) D 2, b n (h w) D 2

        # positional embeddings for queries
        if self.position_embedding_type == 'sine':
            pos_emb = self.position_embedding(h=self.h, w=self.w, device=queries.device).repeat(b, 1, 1, 1)  # b d h w
        elif self.position_embedding_type == 'learned':
            pos_emb = self.position_embedding(torch.zeros(b, self.h, self.w).to(queries.device))             # b d h w
        pos_emb = rearrange(pos_emb, 'b c h w -> b (h w) c')   
        

        # global queries, not supported yet         
        if queries_g is not None:
            # # Create global queries and uniformly sample from spatial tensors
            # h_indices = torch.linspace(0, self.h - 1, self.m, dtype=torch.long, device=queries.device)
            # w_indices = torch.linspace(0, self.w - 1, self.m, dtype=torch.long, device=queries.device)
            # h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')  # m m
            # flat_indices = (h_grid * self.w + w_grid).flatten()  # (m*m)
        
            # pos_emb_g = pos_emb[:, flat_indices, :]  # b (m*m) d
            # reference_points_2d_g = reference_points_2d[:, flat_indices, :]  # b (m*m) 2
            # reference_points_3d_g = reference_points_3d[:, :, flat_indices, :, :]  # b n (m*m) D 2
            # bev_mask_g = bev_mask[:, :, flat_indices, :, :]  # b n (m*m) D 2
            
            # # Concatenate original and global tensors
            # queries = torch.cat([queries, queries_g], dim=1)  # b ((h*w)+(m*m)) d
            # pos_emb = torch.cat([pos_emb, pos_emb_g], dim=1)  # b ((h*w)+(m*m)) d
            # reference_points_2d = torch.cat([reference_points_2d, reference_points_2d_g], dim=1)  # b ((h*w)+(m*m)) 2
            # reference_points_3d = torch.cat([reference_points_3d, reference_points_3d_g], dim=2)  # b n ((h*w)+(m*m)) D 2
            # bev_mask = torch.cat([bev_mask, bev_mask_g], dim=2)  # b n ((h*w)+(m*m)) D 2
            NotImplementedError("global queries are not supported yet")
       
        embeds = [pos_emb, self.level_embed, self.cams_embed] 
        for _, layer in enumerate(self.DeformAttnEnc):
            queries = layer(queries, embeds, reference_points_cam, reference_points_bev, features, bev_mask)


        hw_size = self.h * self.w
        if queries_g is not None:
            raise NotImplementedError("global queries are not supported yet")
        else:
            return queries[:, :hw_size, :]
