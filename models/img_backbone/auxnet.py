import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Type
from einops import rearrange
from fvcore.nn import sigmoid_focal_loss

from utils.loss import SimpleLoss

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.ReLU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SELayer(nn.Module):
    def __init__(
        self,
        channels: int,
        act_layer: Type[nn.Module] = nn.ReLU,
        gate_layer: Type[nn.Module] = nn.Sigmoid,
    ):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x: torch.Tensor, x_se: torch.Tensor) -> torch.Tensor:
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class _ASPPModule(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        batch_norm: Type[nn.Module],
    ):
        super().__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = batch_norm(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(
        self,
        inplanes: int,
        mid_channels: int = 256,
        batch_norm: Type[nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes, mid_channels, 1, padding=0, dilation=dilations[0], batch_norm=batch_norm
        )
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            batch_norm=batch_norm,
        )
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            batch_norm=batch_norm,
        )
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            batch_norm=batch_norm,
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            batch_norm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = batch_norm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class BasicBlock(nn.Module):
    """
    Torch-only ResNet BasicBlock compatible with MMDetection's state_dict keys.

    This is intentionally minimal: it supports the exact usage in `DepthNet`
    (no downsample, stride=1), but keeps canonical attribute names (`conv1`,
    `bn1`, `conv2`, `bn2`, `downsample`, `relu`) so weights can be copied 1:1.
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out

class DepthNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        context_channels: int,
        depth_channels: int,
        use_dcn: bool = True,
        use_aspp: bool = True,
        with_cp: bool = False,
        aspp_mid_channels: int = -1,
        only_depth: bool = False,
        feat_down_sample: int = 32,
        grid_config: dict = None,
        loss_depth_weight: float = 1.0,
        mixed_precision: bool = False,
        **kwargs,
    ):
        super().__init__()
        if with_cp:
            raise ValueError("DepthNetStandalone does not support checkpointing (with_cp=True).")
        if use_dcn:
            raise ValueError("DepthNetStandalone does not support DCN (use_dcn=True).")

        self.mixed_precision = mixed_precision
        self.feat_down_sample = feat_down_sample
        self.grid_config = grid_config
        self.D = int((grid_config['depth'][1] - grid_config['depth'][0]) / grid_config['depth'][2])
        self.loss_depth_weight = loss_depth_weight

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.only_depth = only_depth or context_channels == 0
        if not self.only_depth:
            self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
            self.context_mlp = Mlp(22, mid_channels, mid_channels)
            self.context_se = SELayer(mid_channels)

        self.bn = nn.BatchNorm1d(22)
        self.depth_mlp = Mlp(22, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)

        depth_conv_list = [
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            if aspp_mid_channels < 0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        depth_conv_list.append(
            nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)
        )
        self.depth_conv = nn.Sequential(*depth_conv_list)


    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape

        gt_depths_copy = gt_depths.clone()  # full [B,N,H,W] before tile min-pool

        # Split H,W into feat_down_sample x feat_down_sample tiles (match feature stride).
        gt_depths = gt_depths.view(B * N, H // self.feat_down_sample,
                                   self.feat_down_sample, W // self.feat_down_sample,
                                   self.feat_down_sample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        # Flatten each tile to one row so we can min-pool over its pixels.
        gt_depths = gt_depths.view(-1, self.feat_down_sample * self.feat_down_sample)

        # 0 = no depth; treat as large value so min() picks the smallest valid depth in the tile.
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.feat_down_sample, W // self.feat_down_sample)

        # Optional: side-by-side PNG of input vs min-pooled map (set ONLINE3DHDMAP_VIZ_GT_DEPTH=1).
        if False:
            
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np

            d_min, d_max = self.grid_config['depth'][0], self.grid_config['depth'][1]
            raw = gt_depths_copy[0, 0].detach().float().cpu().numpy()
            pooled = gt_depths[0].detach().float().cpu().numpy()
            raw_viz = np.where(raw > 0, raw, np.nan)
            pooled_viz = np.where(pooled < 1e4, pooled, np.nan)  # tiles with no valid depth stayed ~1e5

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            titles = (
                'gt_depths_copy [B=0,N=0] (H×W)',
                f'after min-pool (H/{self.feat_down_sample}×W/{self.feat_down_sample})',
            )
            for ax, arr, title in zip(axes, (raw_viz, pooled_viz), titles):
                im = ax.imshow(arr, cmap='turbo', vmin=d_min, vmax=d_max, aspect='auto', interpolation='nearest')
                ax.set_title(title)
                fig.colorbar(im, ax=ax, fraction=0.046, label='depth (m)')
            plt.tight_layout()
            out = os.environ.get('ONLINE3DHDMAP_VIZ_GT_DEPTH_PATH', './debug_gt_depth_before_after.png')
            plt.savefig(out, dpi=150)
            plt.close(fig)

        # Map depth to discrete bin index (grid_config['depth']: min, max, step).
        # depth = (depth - (min_depth - step)) / step = (depth - min_depth) / step + 1 -> from 1 to 
        gt_depths = (gt_depths - (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))

        # One-hot over D bins; drop class 0 (invalid / out-of-range) for BCE targets.
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        return gt_depths.float()

    def get_depth_loss(self, depth_labels, depth_preds):
        
        if depth_preds is None:
            return 0
        
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.D)

        fg_mask = depth_labels > 0.0 
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        
        depth_loss = F.binary_cross_entropy_with_logits(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())

        return self.loss_depth_weight * depth_loss

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
        sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

    def forward(self, x, sensor2ego, intrin, post_rot, post_tran, depth_labels=None):
        '''
        x : (B, N, C, H, W)
        sensor2ego : (B, N, 4, 4) = camera2ego
        intrin : (B, N, 4, 4) = camera intrinsic
        post_rot : (B, N, 3, 3) = camera post rotation
        post_tran : (B, N, 3) = camera post translation
        '''

        B, N, _, _ = sensor2ego.shape

        mlp_input = self.get_mlp_input(sensor2ego, intrin, post_rot, post_tran)
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)

        if not self.only_depth:
            context_se = self.context_mlp(mlp_input)[..., None, None]
            context = self.context_se(x, context_se)
            context = self.context_conv(context)

        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        depth = rearrange(depth, '(b n) c h w -> b n h w c', b=B, n=N)

        output = {'depth': depth, 'depth_loss': 0}
        if depth_labels is not None:
            if self.mixed_precision:
                depth = depth.float()
                depth_labels = depth_labels.float()
            with torch.amp.autocast("cuda", enabled=False):            
                depth_loss = self.get_depth_loss(depth_labels, depth)
                output['depth_loss'] = depth_loss
        return output

class BEVSegNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mixed_precision: bool = False,
        pos_weight: float = 1.0,
        weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.alpha = -1.0
        self.gamma = 2.0
        self.reduction = 'none'
        self.head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
                )
        
        self.mixed_precision = mixed_precision
        self.weight = weight
        self.loss_fn = SimpleLoss(pos_weight=pos_weight, loss_weight=weight)

    def forward(self, x, label=None):
        '''
        x : (B, C, H, W)
        label : (B, 1, H, W)
        '''

        output_seg = self.head(x) # (B, 1, H, W)
        output = {'bevseg': output_seg, 'bevseg_loss': 0}
        if label is not None:
            if self.mixed_precision:
                output_seg = output_seg.float()
                label = label.float()
            with torch.amp.autocast("cuda", enabled=False):            
                output['bevseg_loss'] = self.loss_fn(output_seg, label)
        return output

class PVSegNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mixed_precision: bool = False,
        pos_weight: float = 1.0,
        weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.alpha = -1.0
        self.gamma = 2.0
        self.reduction = 'none'
        self.head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
                )
        
        self.mixed_precision = mixed_precision
        self.weight = weight
        self.loss_fn = SimpleLoss(pos_weight=pos_weight, loss_weight=weight)

    def forward(self, x, label=None):
        '''
        x : (B*N, C, H, W)
        label : (B, N, H, W)
        '''

        output_seg = self.head(x) 
        output = {'pvseg': output_seg, 'pvseg_loss': 0}
        if label is not None:
            label = rearrange(label, 'b n h w -> (b n) 1 h w')
            if self.mixed_precision:
                output_seg = output_seg.float()
                label = label.float()
            with torch.amp.autocast("cuda", enabled=False):            
                output['pvseg_loss'] = self.loss_fn(output_seg, label)
        return output        