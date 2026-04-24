"""standalone_lss_transform.py
Standalone, pure-PyTorch implementation of LSSTransform.

Removes every dependency on mmdet3d / mmcv while preserving the exact same
architecture so that weights can be transferred with::

    standalone.load_state_dict(original.state_dict())

Only restriction: ``depthnet_cfg`` must have ``use_dcn=False`` (the default in
maptrv2_nusc_r50_24ep.py), because the standalone version does not wrap
mmcv's DeformConv.

Corresponding original:
  projects/mmdet3d_plugin/maptr/modules/encoder.py  ·  LSSTransform (line 1065)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import wraps
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Shim: replaces @force_fp32() / @force_fp32 from mmcv.runner
# With fp32 inputs the decorator is a no-op, so a passthrough is correct.
# ─────────────────────────────────────────────────────────────────────────────
def force_fp32(func=None, **_kw):
    """No-op shim for mmcv's ``@force_fp32()`` / ``@force_fp32``."""
    if callable(func):          # @force_fp32 (no parentheses)
        return func

    def decorator(f):           # @force_fp32()  or  @force_fp32(apply_to=...)
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Grid helpers  (verbatim from encoder.py)
# ─────────────────────────────────────────────────────────────────────────────
def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor(
        [int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


# ─────────────────────────────────────────────────────────────────────────────
# Pure-PyTorch BEV pool
# Replaces the CUDA kernel  mmdet3d.ops.bev_pool  with scatter_add.
# Both operations compute the same sum; small fp differences may arise from
# CUDA atomic-add ordering, which is why the validation uses a tolerance.
# ─────────────────────────────────────────────────────────────────────────────
def bev_pool_pytorch(feats: torch.Tensor, coords: torch.Tensor,
                     B: int, Z: int, X: int, Y: int) -> torch.Tensor:
    """Scatter-add replacement for ``mmdet3d.ops.bev_pool``.

    Args:
        feats  (Tensor): ``(N, C)``  per-point features.
        coords (Tensor): ``(N, 4)``  integer voxel coords ``[gx, gy, gz, batch]``.
        B, Z, X, Y (int): output grid dimensions.

    Returns:
        Tensor: ``(B, C, Z, X, Y)`` BEV feature volume.
    """
    N, C = feats.shape
    gx = coords[:, 0].long()
    gy = coords[:, 1].long()
    gz = coords[:, 2].long()
    bi = coords[:, 3].long()

    # Flat index into a (B, Z, X, Y) volume for each point.
    bev_idx = bi * (Z * X * Y) + gz * (X * Y) + gx * Y + gy  # (N,)

    # Accumulate: out[bev_idx[n], c] += feats[n, c]
    out = feats.new_zeros(B * Z * X * Y, C)
    out.scatter_add_(0, bev_idx.unsqueeze(1).expand(-1, C), feats)

    # Reshape (B*Z*X*Y, C)  →  (B, Z, X, Y, C)  →  (B, C, Z, X, Y)
    return out.view(B, Z, X, Y, C).permute(0, 4, 1, 2, 3).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# BasicBlock  –  standard ResNet block matching mmdet's layer names
# (conv1, bn1, relu, conv2, bn2, downsample)
# ─────────────────────────────────────────────────────────────────────────────
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, groups=1, base_width=64,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(
            inplanes, planes, 3,
            stride=stride, padding=dilation, dilation=dilation,
            groups=groups, bias=False,
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, 3,
            padding=1, groups=groups, bias=False,
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ─────────────────────────────────────────────────────────────────────────────
# ASPP helpers  (verbatim from encoder.py lines 1366-1470)
# ─────────────────────────────────────────────────────────────────────────────
class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes, planes,
            kernel_size=kernel_size, stride=1,
            padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(inplanes, mid_channels, 1,
                                 padding=0,           dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, mid_channels, 3,
                                 padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, mid_channels, 3,
                                 padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, mid_channels, 3,
                                 padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# ─────────────────────────────────────────────────────────────────────────────
# DepthNet helpers  (verbatim from encoder.py lines 747-864)
# ─────────────────────────────────────────────────────────────────────────────
class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.ReLU, drop=0.0):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1    = nn.Linear(in_features,     hidden_features)
        self.act    = act_layer()
        self.drop1  = nn.Dropout(drop)
        self.fc2    = nn.Linear(hidden_features, out_features)
        self.drop2  = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x);   x = self.act(x);   x = self.drop1(x)
        x = self.fc2(x);                       x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1        = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate        = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):

    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels, use_dcn=True, use_aspp=True,
                 with_cp=False, aspp_mid_channels=-1, only_depth=False):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.only_depth = only_depth or context_channels == 0
        if not self.only_depth:
            self.context_conv = nn.Conv2d(
                mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
            self.context_mlp = Mlp(22, mid_channels, mid_channels)
            self.context_se  = SELayer(mid_channels)
        self.bn        = nn.BatchNorm1d(22)
        self.depth_mlp = Mlp(22, mid_channels, mid_channels)
        self.depth_se  = SELayer(mid_channels)

        depth_conv_list = [
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            if aspp_mid_channels < 0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        if use_dcn:
            raise ValueError(
                "use_dcn=True is not supported in the standalone class. "
                "Set use_dcn=False in depthnet_cfg (the default in maptrv2_nusc_r50_24ep.py)."
            )
        depth_conv_list.append(
            nn.Conv2d(mid_channels, depth_channels,
                      kernel_size=1, stride=1, padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.with_cp    = with_cp

    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        if not self.only_depth:
            context_se = self.context_mlp(mlp_input)[..., None, None]
            context    = self.context_se(x, context_se)
            context    = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth    = self.depth_se(x, depth_se)
        if self.with_cp:
            depth = checkpoint(self.depth_conv, depth)
        else:
            depth = self.depth_conv(depth)
        if not self.only_depth:
            return torch.cat([depth, context], dim=1)
        else:
            return depth


# ─────────────────────────────────────────────────────────────────────────────
# BaseTransformStandalone
# Mirrors BaseTransform (encoder.py lines 28-320) without mmcv/mmdet3d.
# The only external replacement is bev_pool  →  bev_pool_pytorch.
# ─────────────────────────────────────────────────────────────────────────────
class BaseTransformStandalone(nn.Module):

    def __init__(self, in_channels, out_channels, feat_down_sample,
                 pc_range, voxel_size, dbound):
        super().__init__()
        self.in_channels    = in_channels
        self.feat_down_sample = feat_down_sample
        self.xbound = [pc_range[0], pc_range[3], voxel_size[0]]
        self.ybound = [pc_range[1], pc_range[4], voxel_size[1]]
        self.zbound = [pc_range[2], pc_range[5], voxel_size[2]]
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C        = out_channels
        self.frustum  = None
        self.D        = int((dbound[1] - dbound[0]) / dbound[2])
        self.fp16_enabled = False

    @force_fp32()
    def create_frustum(self, fH, fW, img_metas):
        # img_metas[0]['img_shape'][0] is (H, W, 3) for the first camera.
        iH = img_metas[0]['img_shape'][0][0]
        iW = img_metas[0]['img_shape'][0][1]
        assert iH // self.feat_down_sample == fH, \
            f"img_shape H={iH} / feat_down_sample={self.feat_down_sample} != fH={fH}"

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape
        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )
        return torch.stack((xs, ys, ds), -1)

    @force_fp32()
    def get_geometry_v1(self, fH, fW, rots, trans, intrins,
                        post_rots, post_trans,
                        lidar2ego_rots, lidar2ego_trans,
                        img_metas, **kwargs):
        B, N, _ = trans.shape
        device   = trans.device

        if self.frustum is None:
            self.frustum = self.create_frustum(fH, fW, img_metas).to(device)

        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)

        # Use CPU fallback for torch.inverse on builds where cuSPARSE init fails.
        try:
            post_rots_inv = torch.inverse(post_rots)
        except RuntimeError as e:
            if 'cusparseCreate' in str(e) or 'operation not supported' in str(e):
                post_rots_inv = torch.inverse(
                    post_rots.detach().cpu()).to(device=device, dtype=post_rots.dtype)
            else:
                raise
        points = post_rots_inv.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:3]),
            5,
        )

        try:
            intrins_inv = torch.inverse(intrins)
        except RuntimeError as e:
            if 'cusparseCreate' in str(e) or 'operation not supported' in str(e):
                intrins_inv = torch.inverse(
                    intrins.detach().cpu()).to(device=device, dtype=intrins.dtype)
            else:
                raise
        combine = rots.matmul(intrins_inv)
        points  = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        points -= lidar2ego_trans.view(B, 1, 1, 1, 1, 3)
        try:
            lidar2ego_rots_inv = torch.inverse(lidar2ego_rots)
        except RuntimeError as e:
            if 'cusparseCreate' in str(e) or 'operation not supported' in str(e):
                lidar2ego_rots_inv = torch.inverse(
                    lidar2ego_rots.detach().cpu()).to(device=device, dtype=lidar2ego_rots.dtype)
            else:
                raise
        points = (
            lidar2ego_rots_inv.view(B, 1, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
            .squeeze(-1)
        )

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_mlp_input(self, sensor2ego, intrin, post_rot, post_tran):
        raise NotImplementedError

    def get_cam_feats(self, x, mlp_input):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        """Pure-PyTorch drop-in for BaseTransform.bev_pool."""
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        x = x.reshape(Nprime, C)

        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        kept = (
            (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        )
        x          = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool_pytorch(
            x, geom_feats,
            B,
            int(self.nx[2]),   # Z
            int(self.nx[0]),   # X
            int(self.nx[1]),   # Y
        )
        return torch.cat(x.unbind(dim=2), 1)   # collapse Z → (B, C*Z, X, Y)

    @force_fp32()
    def forward(self, images, img_metas):
        B, N, C, fH, fW = images.shape

        lidar2img, camera2ego, camera_intrinsics, img_aug_matrix, lidar2ego = \
            [], [], [], [], []
        for m in img_metas:
            lidar2img.append(m['lidar2img'])
            camera2ego.append(m['camera2ego'])
            camera_intrinsics.append(m['camera_intrinsics'])
            img_aug_matrix.append(m['img_aug_matrix'])
            lidar2ego.append(m['lidar2ego'])

        lidar2img          = images.new_tensor(np.asarray(lidar2img))
        camera2ego         = images.new_tensor(np.asarray(camera2ego))
        camera_intrinsics  = images.new_tensor(np.asarray(camera_intrinsics))
        img_aug_matrix     = images.new_tensor(np.asarray(img_aug_matrix))
        lidar2ego          = images.new_tensor(np.asarray(lidar2ego))

        rots             = camera2ego[..., :3, :3]
        trans            = camera2ego[..., :3, 3]
        intrins          = camera_intrinsics[..., :3, :3]
        post_rots        = img_aug_matrix[..., :3, :3]
        post_trans       = img_aug_matrix[..., :3, 3]
        lidar2ego_rots   = lidar2ego[..., :3, :3]
        lidar2ego_trans  = lidar2ego[..., :3, 3]

        geom = self.get_geometry_v1(
            fH, fW, rots, trans, intrins,
            post_rots, post_trans,
            lidar2ego_rots, lidar2ego_trans,
            img_metas,
        )
        mlp_input = self.get_mlp_input(
            camera2ego, camera_intrinsics, post_rots, post_trans)
        x, depth = self.get_cam_feats(images, mlp_input)
        x = self.bev_pool(geom, x)
        x = x.permute(0, 1, 3, 2).contiguous()   # (B, C*Z, X, Y) → (B, C*Z, Y, X)
        return x, depth


# ─────────────────────────────────────────────────────────────────────────────
# LSSTransformStandalone
# Architecture-identical clone of LSSTransform (encoder.py lines 1065-1212).
# ─────────────────────────────────────────────────────────────────────────────
class LSSTransformStandalone(BaseTransformStandalone):

    def __init__(
        self,
        in_channels,
        out_channels,
        feat_down_sample,
        pc_range,
        voxel_size,
        dbound,
        downsample=1,
        loss_depth_weight=3.0,
        depthnet_cfg=None,
        grid_config=None,
    ):
        if depthnet_cfg is None:
            depthnet_cfg = {}
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feat_down_sample=feat_down_sample,
            pc_range=pc_range,
            voxel_size=voxel_size,
            dbound=dbound,
        )
        self.loss_depth_weight = loss_depth_weight
        self.grid_config       = grid_config
        self.depth_net = DepthNet(
            in_channels, in_channels, self.C, self.D, **depthnet_cfg)

        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3,
                          stride=downsample, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    # ------------------------------------------------------------------
    # get_cam_feats  –  fuses depth & context; returns (x_3d, depth)
    # ------------------------------------------------------------------
    @force_fp32()
    def get_cam_feats(self, x, mlp_input):
        B, N, C, fH, fW = x.shape
        x = x.view(B * N, C, fH, fW)
        x = self.depth_net(x, mlp_input)

        depth = x[:, :self.D].softmax(dim=1)
        # Outer product of depth distribution and context features:
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        # Reshape to (B, N, D, H, W, C) for bev_pool
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        depth = depth.view(B, N, self.D, fH, fW)
        return x, depth

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, images, img_metas):
        x, depth = super().forward(images, img_metas)
        x = self.downsample(x)
        return dict(bev=x, depth=depth)

    # ------------------------------------------------------------------
    # Depth supervision helpers  (identical to original)
    # ------------------------------------------------------------------
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:  ``(B, N, H, W)``  raw lidar depth maps.
        Output: ``(B*N*h*w, D)``  one-hot depth labels at feature resolution.
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.feat_down_sample, self.feat_down_sample,
            W // self.feat_down_sample, self.feat_down_sample,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.feat_down_sample * self.feat_down_sample)

        # Replace zero (invalid) depth with a large value before taking min.
        gt_depths_tmp = torch.where(
            gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(
            B * N, H // self.feat_down_sample, W // self.feat_down_sample)

        gt_depths = (
            gt_depths
            - (self.grid_config['depth'][0] - self.grid_config['depth'][2])
        ) / self.grid_config['depth'][2]
        gt_depths = torch.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1
        ).view(-1, self.D + 1)[:, 1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        if depth_preds is None:
            return 0
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds  = depth_preds.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.D)
        fg_mask      = depth_labels > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds  = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds, depth_labels, reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss

    # ------------------------------------------------------------------
    # Camera-aware MLP input builder  (identical to original)
    # ------------------------------------------------------------------
    def get_mlp_input(self, sensor2ego, intrin, post_rot, post_tran):
        B, N, _, _ = sensor2ego.shape
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
        ], dim=-1)
        sensor2ego = sensor2ego[:, :, :3, :].reshape(B, N, -1)
        mlp_input  = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input
