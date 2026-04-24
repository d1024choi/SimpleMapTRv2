"""resnet_new.py – ResNet-50/101 backbone + MapTR-faithful FPN neck.

Faithfully mirrors the original MapTR configuration:

    img_backbone=dict(
        type='ResNet', depth=50, num_stages=4,
        out_indices=(3,),                          # only C4 (2048-ch) output
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,                        # 256
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,                     # 1
        relu_before_extra_convs=True),

Key differences from resnet.py:
  1. FPNNeck uses bare Conv1×1 lateral convs and bare Conv3×3 output convs –
     NO BatchNorm, NO ReLU inside the neck (faithful to mmcv FPN).
  2. Top-down path uses nearest-neighbour interpolation (mmcv default).
  3. Supports add_extra_convs='on_output'/'on_input'/'on_lateral' and
     relu_before_extra_convs (ReLU inserted before each extra-level conv).
  4. All conv weights initialised with Xavier-uniform (mmcv FPN default).
  5. Default target_layers=['reduction_4'] matches out_indices=(3,).
  6. Return format: dict  {layer_name: tensor}  – preserves compatibility
     with Scratch.py which calls  list(features.values()).
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

try:
    from torchvision.models import ResNet50_Weights, ResNet101_Weights
    _TORCHVISION_NEW_API = True
except ImportError:
    _TORCHVISION_NEW_API = False

# ---------------------------------------------------------------------------
# Pre-computed output shapes per image resolution
# ---------------------------------------------------------------------------
FEATURE_SIZE_224x480 = {
    'resnet50': {
        'reduction_1': torch.Size((1, 256, 56, 120)),
        'reduction_2': torch.Size((1, 512, 28, 60)),
        'reduction_3': torch.Size((1, 1024, 14, 30)),
    }
}

FEATURE_SIZE_448x960 = {
    'resnet50': {
        'reduction_1': torch.Size((1, 256, 112, 240)),
        'reduction_2': torch.Size((1, 512, 56, 120)),
        'reduction_3': torch.Size((1, 1024, 28, 60)),
        'reduction_4': torch.Size((1, 2048, 14, 30)),
    }
}

FEATURE_SIZE_450x800 = {
    'resnet50': {
        'reduction_1': torch.Size((1, 256, 113, 200)),
        'reduction_2': torch.Size((1, 512, 57, 100)),
        'reduction_3': torch.Size((1, 1024, 29, 50)),
        'reduction_4': torch.Size((1, 2048, 15, 25)),
    }
}

FEATURE_SIZE_900x1600 = {
    'resnet50': {
        'reduction_1': torch.Size((1, 256, 225, 400)),
        'reduction_2': torch.Size((1, 512, 113, 200)),
        'reduction_3': torch.Size((1, 1024, 57, 100)),
        'reduction_4': torch.Size((1, 2048, 29, 50)),
    },
    'resnet101': {
        'reduction_1': torch.Size((1, 256, 225, 400)),
        'reduction_2': torch.Size((1, 512, 113, 200)),
        'reduction_3': torch.Size((1, 1024, 57, 100)),
        'reduction_4': torch.Size((1, 2048, 29, 50)),
    },
}


# ---------------------------------------------------------------------------
# FPNNeck – faithful port of mmcv FPN
# ---------------------------------------------------------------------------

class FPNNeck(nn.Module):
    """Image neck that faithfully mirrors mmcv's Feature Pyramid Network (FPN).

    Architecture for the default MapTR-tiny config
    (in_channels=[2048], out_channels=256, num_outs=1):

        C4 (2048) ──► Conv1×1 ──► Conv3×3 ──► out[0] (256)
                     (lateral)  (fpn_conv)

    No BN or ReLU inside lateral/fpn convs; weights initialised with
    Xavier-uniform (identical to mmcv FPN).

    When num_outs > len(in_channels), extra output levels are appended via
    stride-2 Conv3×3, optionally preceded by ReLU when
    relu_before_extra_convs=True.

    Args:
        in_channels (list[int]): Number of channels for each backbone input.
        out_channels (int):      Output channel count for every FPN level.
        num_outs (int):          Total number of output levels.
        start_level (int):       Index of first backbone level to use.
        add_extra_convs (str):   Source for extra levels beyond backbone:
                                   'on_input'   – last backbone input feature
                                   'on_lateral' – last lateral conv output
                                   'on_output'  – last FPN output  (mmcv default)
        relu_before_extra_convs (bool): Insert ReLU before each extra-level conv.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        add_extra_convs: str = 'on_output',
        relu_before_extra_convs: bool = True,
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, (list, tuple))
        assert add_extra_convs in ('on_input', 'on_lateral', 'on_output'), (
            f"add_extra_convs must be one of 'on_input', 'on_lateral', "
            f"'on_output'; got '{add_extra_convs}'"
        )

        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs

        # Number of backbone levels actually consumed
        self.num_backbone_outs = self.num_ins - start_level

        # ── Lateral 1×1 convs (no BN, no ReLU – faithful to mmcv FPN) ─────
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels[start_level:]
        )

        # ── FPN output 3×3 convs (no BN, no ReLU) ──────────────────────────
        used_backbone_levels = min(num_outs, self.num_backbone_outs)
        extra_levels = num_outs - used_backbone_levels

        self.fpn_convs = nn.ModuleList(
            # regular levels from backbone
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
             for _ in range(used_backbone_levels)]
            +
            # extra levels (stride-2 to halve spatial resolution each step)
            [nn.Conv2d(out_channels, out_channels, kernel_size=3,
                       stride=2, padding=1)
             for _ in range(extra_levels)]
        )

        self.used_backbone_levels = used_backbone_levels
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for all conv weights (mmcv default)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # ------------------------------------------------------------------
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            inputs: Ordered list of backbone feature maps,
                    inputs[0] = largest spatial size, inputs[-1] = smallest.
                    len(inputs) must equal self.num_ins.

        Returns:
            List of ``num_outs`` output tensors, same order as inputs.
        """
        assert len(inputs) == self.num_ins, (
            f"FPNNeck expects {self.num_ins} inputs, got {len(inputs)}"
        )

        # 1. Build lateral features for the used backbone levels
        laterals: List[torch.Tensor] = [
            self.lateral_convs[i](inputs[i + self.start_level])
            for i in range(self.num_backbone_outs)
        ]

        # 2. Top-down path: nearest-neighbour upsample + element-wise add
        #    (start from highest level, walk toward lowest)
        for i in range(self.used_backbone_levels - 1, 0, -1):
            _, _, H, W = laterals[i - 1].shape
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=(H, W), mode='nearest'
            )

        # 3. Apply FPN output convs to backbone levels
        outs: List[torch.Tensor] = [
            self.fpn_convs[i](laterals[i])
            for i in range(self.used_backbone_levels)
        ]

        # 4. Extra output levels beyond the backbone (e.g. P6, P7)
        if self.num_outs > len(outs):
            if self.add_extra_convs == 'on_output':
                extra_src = outs[-1]
            elif self.add_extra_convs == 'on_input':
                extra_src = inputs[self.num_ins - 1]
            else:  # 'on_lateral'
                extra_src = laterals[-1]

            for i in range(self.used_backbone_levels, self.num_outs):
                if self.relu_before_extra_convs:
                    extra_src = F.relu(extra_src)
                outs.append(self.fpn_convs[i](extra_src))
                extra_src = outs[-1]

        return outs


# ---------------------------------------------------------------------------
# ResNet – backbone + FPNNeck, faithful to MapTR config
# ---------------------------------------------------------------------------

class ResNet(nn.Module):
    """ResNet-50/101 backbone with a MapTR-faithful FPNNeck.

    Stage layout (matches MMDetection / MapTR convention):

        conv1 = stem (conv+bn+relu+maxpool) + layer1
        conv2 = layer2   (stride 8 from input)
        conv3 = layer3   (stride 16)
        conv4 = layer4   (stride 32) → 'reduction_4', 2048 channels

    Freezing semantics (frozen_stages):

        0  →  nothing frozen
        1  →  conv1  (stem + layer1)
        2  →  conv1 + conv2
        3  →  conv1 + conv2 + conv3
        4  →  all stages

    norm_cfg.requires_grad=False  →  BN γ/β frozen across the whole backbone.
    norm_eval=True               →  ALL BN layers kept in eval() during training
                                    (running mean/var never updated by batches).

    Args:
        input_size (tuple[int,int]): (H, W) of input images.
        fpn_dim (int):               FPN output channels (= _dim_ = 256).
        resnet_model (str):          'resnet50' or 'resnet101'.
        target_layers (list[str]):   Which reduction stages to feed into FPN.
                                     Defaults to ['reduction_4'] matching
                                     out_indices=(3,) in the MapTR config.
        skip_fpn (bool):             If True, skip the neck and return raw
                                     backbone features.
        frozen_stages (int):         Number of stages to freeze (default 1).
        norm_cfg (dict):             {'type': 'BN', 'requires_grad': bool}.
        norm_eval (bool):            Keep BN in eval mode during training.
        num_outs (int):              Number of FPN output levels (default 1).
        add_extra_convs (str):       FPN extra-level source (default 'on_output').
        relu_before_extra_convs (bool): ReLU before extra FPN convs (default True).
    """

    # Canonical channel counts per reduction stage (torchvision ResNet-50/101)
    _STAGE_CHANNELS = {
        'reduction_1': 256,
        'reduction_2': 512,
        'reduction_3': 1024,
        'reduction_4': 2048,
    }

    def __init__(
        self,
        input_size: tuple,
        fpn_dim: int,
        resnet_model: str = 'resnet50',
        target_layers: List[str] = None,
        skip_fpn: bool = False,
        frozen_stages: int = 1,
        norm_cfg: dict = None,
        norm_eval: bool = True,
        num_outs: int = 1,
        add_extra_convs: str = 'on_output',
        relu_before_extra_convs: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        # Defaults that match the MapTR-tiny config
        if target_layers is None:
            target_layers = ['reduction_4']
        if norm_cfg is None:
            norm_cfg = {'type': 'BN', 'requires_grad': False}

        self.skip_fpn = skip_fpn
        self.target_layers = target_layers
        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        # ------------------------------------------------------------------
        # Backbone: torchvision ResNet-50 / ResNet-101 with ImageNet weights
        # Supports both torchvision >= 0.13 (Weights API) and < 0.13 (pretrained=True)
        # ------------------------------------------------------------------
        if _TORCHVISION_NEW_API:
            _w50  = ResNet50_Weights.IMAGENET1K_V1
            _w101 = ResNet101_Weights.IMAGENET1K_V1
            if resnet_model == 'resnet50':
                children = list(models.resnet50(weights=_w50).children())
            elif resnet_model == 'resnet101':
                children = list(models.resnet101(weights=_w101).children())
            else:
                sys.exit(f">> [Error] Unsupported resnet model: {resnet_model}")
        else:
            if resnet_model == 'resnet50':
                children = list(models.resnet50(pretrained=True).children())
            elif resnet_model == 'resnet101':
                children = list(models.resnet101(pretrained=True).children())
            else:
                sys.exit(f">> [Error] Unsupported resnet model: {resnet_model}")

        # Assemble stages exactly as in resnet.py so frozen_stages semantics
        # are identical (stage-index 0 = conv1 = stem + layer1).
        self.conv1 = nn.Sequential(
            children[0], children[1], children[2], children[3],  # stem
            *children[4]                                          # layer1
        )
        self.conv2 = children[5]   # layer2
        self.conv3 = children[6]   # layer3
        self.conv4 = children[7]   # layer4

        # Apply freezing
        self._freeze_stages()
        if not self.norm_cfg.get('requires_grad', True):
            self._freeze_bn_params()

        # ------------------------------------------------------------------
        # Output-shape lookup (used for downstream bookkeeping)
        # ------------------------------------------------------------------
        img_H, img_W = input_size
        size_map = {
            (900, 1600): FEATURE_SIZE_900x1600,
            (450, 800):  FEATURE_SIZE_450x800,
            (448, 960):  FEATURE_SIZE_448x960,
            (224, 480):  FEATURE_SIZE_224x480,
        }
        if (img_H, img_W) not in size_map:
            sys.exit(f">> [Error] Unsupported image size: {img_H}×{img_W}")
        FEATURE_SIZE = size_map[(img_H, img_W)][resnet_model]

        self.output_shapes: Dict[str, torch.Size] = {
            k: FEATURE_SIZE[k] for k in target_layers
        }

        # ------------------------------------------------------------------
        # Neck: FPNNeck (only built when skip_fpn=False)
        # ------------------------------------------------------------------
        self.FPN: Optional[FPNNeck] = None
        if not skip_fpn:
            in_channels = [
                self._STAGE_CHANNELS[layer] for layer in target_layers
            ]
            self.FPN = FPNNeck(
                in_channels=in_channels,
                out_channels=fpn_dim,
                num_outs=num_outs,
                start_level=0,
                add_extra_convs=add_extra_convs,
                relu_before_extra_convs=relu_before_extra_convs,
            )
            # Update output_shapes to reflect FPN output channels
            for i, layer in enumerate(target_layers):
                _, _, h, w = self.output_shapes[layer]
                self.output_shapes[layer] = torch.Size([1, fpn_dim, h, w])

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    def _freeze_stages(self) -> None:
        """Freeze params and set BN to eval() for all frozen stages.

        Called both in __init__ and in train() because model.train()
        recursively resets every submodule back to training mode.
        """
        stage_modules = [self.conv1, self.conv2, self.conv3, self.conv4]
        for idx in range(min(self.frozen_stages, len(stage_modules))):
            stage = stage_modules[idx]
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def _freeze_bn_params(self) -> None:
        """Set requires_grad=False for γ (weight) and β (bias) of every BN."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)

    # ------------------------------------------------------------------
    # train() override – re-apply freezing after model.train() resets children
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> 'ResNet':
        """Keep frozen stages and BN layers in eval during training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Image tensor (B, 3, H, W), already normalised.

        Returns:
            dict {layer_name: feature_tensor}.
            With the default config (target_layers=['reduction_4'],
            skip_fpn=False) the dict has exactly one key ('reduction_4')
            whose value has shape (B, fpn_dim, H/32, W/32).
        """
        x1 = self.conv1(x)   # 1/4  of input resolution
        x2 = self.conv2(x1)  # 1/8
        x3 = self.conv3(x2)  # 1/16
        x4 = self.conv4(x3)  # 1/32

        backbone_feats = {
            'reduction_1': x1,
            'reduction_2': x2,
            'reduction_3': x3,
            'reduction_4': x4,
        }

        selected = [backbone_feats[k] for k in self.target_layers]

        if not self.skip_fpn and self.FPN is not None:
            neck_outs = self.FPN(selected)
            return {k: v for k, v in zip(self.target_layers, neck_outs)}

        return {k: v for k, v in zip(self.target_layers, selected)}
