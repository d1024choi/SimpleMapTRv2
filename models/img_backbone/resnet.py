from collections import OrderedDict
import sys
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet101_Weights
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

FEATURE_SIZE_224x480 = {
    'resnet50': {
        'reduction_1': torch.Size((1, 256, 56, 120)),
        'reduction_2': torch.Size((1, 512, 28, 60)),
        'reduction_3': torch.Size((1, 1024, 14, 30))
    }
}

FEATURE_SIZE_448x960 = {
    'resnet50': {
        'reduction_1': torch.Size((1, 256, 112, 240)),
        'reduction_2': torch.Size((1, 512, 56, 120)),
        'reduction_3': torch.Size((1, 1024, 28, 60)),
        'reduction_4': torch.Size((1, 2048, 14, 30))
    }
}

FEATURE_SIZE_450x800 = {
    'resnet50': {
        'reduction_1': torch.Size((1, 256, 113, 200)),
        'reduction_2': torch.Size((1, 512, 57, 100)),
        'reduction_3': torch.Size((1, 1024, 29, 50)),
        'reduction_4': torch.Size((1, 2048, 15, 25))
    }
}


FEATURE_SIZE_480x800 = {
    'resnet50': {
        'reduction_1': torch.Size((1, 256, 120, 200)),
        'reduction_2': torch.Size((1, 512, 60, 100)),
        'reduction_3': torch.Size((1, 1024, 30, 50)),
        'reduction_4': torch.Size((1, 2048, 15, 25))
    }
}

FEATURE_SIZE_900x1600 = {
    'resnet50': {
        'reduction_1': torch.Size((1, 256, 225, 400)),
        'reduction_2': torch.Size((1, 512, 113, 200)),
        'reduction_3': torch.Size((1, 1024, 57, 100)),
        'reduction_4': torch.Size((1, 2048, 29, 50))
    },
    
    'resnet101': {
        'reduction_1': torch.Size((1, 256, 225, 400)),
        'reduction_2': torch.Size((1, 512, 113, 200)),
        'reduction_3': torch.Size((1, 1024, 57, 100)),
        'reduction_4': torch.Size((1, 2048, 29, 50))
    }
}

class FPN_OLD(nn.Module):

    def __init__(self, dim, target_layers, input_shapes):
        '''
        dim : target dimension
        target_layers : List
        input_shapes : Dict
        '''
        super(FPN_OLD, self).__init__()

        self.target_layers, self.target_bottom_layer = target_layers, target_layers[0] # [1,2,3,4], 2
        _input_layers = [name for name, shape in input_shapes.items()][::-1] # [4,3,2,1]
        self.input_layers = [] # 4, 3, 2
        for name in _input_layers: # 4->3->2->1
            self.input_layers.append(name)
            if (name == self.target_bottom_layer): # 2 == 2
                break

        # feature map projection
        self.output_shapes, self.proj = [], nn.ModuleDict()
        for name, shape in input_shapes.items():
            b, in_ch, h, w = shape

            if (name in self.input_layers):
                # self.proj[name] = nn.Sequential(nn.Conv2d(in_ch, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(dim))
                self.proj[name] = nn.Conv2d(in_ch, dim, 1)

            # if (name in target_layers):
            self.output_shapes.append(torch.Size([1, dim, h, w]))

        # upsample and merge
        self.bottom_layer_name, self.top_layer_name = self.input_layers[-1], self.input_layers[0]
        self.upsample, self.merge = nn.ModuleDict(), nn.ModuleDict()
        for name in self.input_layers:
            if (name is self.top_layer_name): continue
            # self.upsample[name] = nn.Upsample(scale_factor=2, mode='nearest')
            self.merge[name] = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                                             nn.BatchNorm2d(dim),
                                             nn.ReLU(dim))
            

    def forward(self, feats):
        '''
        feats : Dicts
        '''

        # feat projection
        proj = {}
        for name in self.input_layers:
            proj[name] = self.proj[name](feats[name])

        merge = {}
        for _, bot_name in enumerate(self.input_layers):
            if (bot_name is self.top_layer_name): merge[bot_name] = proj[bot_name]
            else:
                top_name = self.input_layers[_-1]
                # sum_feat = self.upsample[bot_name](merge[top_name]) + proj[bot_name]
                _, _, H, W = proj[bot_name].size()
                sum_feat = F.interpolate(merge[top_name], size=(H, W), mode='bilinear', align_corners=True) + proj[bot_name]
                merge[bot_name] = self.merge[bot_name](sum_feat)

        # return merge[self.target_bottom_layer]
        return {k: v for k, v in reversed(list(merge.items()))}

class FPN(nn.Module):
    """FPN aligned with MMDetection `mmdet.models.necks.FPN` (nearest TD + post 3x3; optional extras)."""
    def __init__(
        self,
        dim,
        level_names,
        in_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
        upsample_mode="nearest",
    ):
        """
        level_names: e.g. ['1','2','3','4'] fine → coarse (matches one in_channels entry each).
        in_channels: same length as level_names; channels for each level.
        num_outs: total outputs (>= number of used backbone levels).
        add_extra_convs: False | True | 'on_input' | 'on_lateral' | 'on_output' (MMDet semantics).
        """
        super().__init__()
        assert len(level_names) == len(in_channels)
        self.level_names = list(level_names)
        self.dim = dim
        self.num_ins = len(level_names)
        self.num_outs = num_outs
        self.start_level = start_level
        self.relu_before_extra_convs = relu_before_extra_convs
        self.upsample_mode = upsample_mode
        # Which backbone levels feed the neck (slice of level_names / in_channels).
        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        # Extra pyramid levels: off, bool→'on_input', or explicit MMDet string.
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ("on_input", "on_lateral", "on_output")
            self.add_extra_convs = add_extra_convs
        elif add_extra_convs:
            self.add_extra_convs = "on_input"
        else:
            self.add_extra_convs = False
        used = self.level_names[start_level : self.backbone_end_level]
        used_in_ch = in_channels[start_level : self.backbone_end_level]
        self.used_level_names = used
        # 1×1: map each used level to dim.  3×3: refine after top-down merge.
        self.lateral_convs = nn.ModuleDict(
            {n: nn.Conv2d(c, dim, 1) for n, c in zip(used, used_in_ch)}
        )
        self.fpn_convs = nn.ModuleDict({n: nn.Conv2d(dim, dim, 3, padding=1) for n in used})
        # Optional stride-2 convs to add more outputs when num_outs > #used levels.
        extra_levels = num_outs - self.backbone_end_level + start_level
        self.extra_level_names = []
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == "on_input":
                    src_ch = in_channels[self.backbone_end_level - 1]
                else:
                    src_ch = dim
                name = f"_extra_{i}"
                self.extra_level_names.append(name)
                self.fpn_convs[name] = nn.Conv2d(src_ch, dim, 3, stride=2, padding=1)



    def forward(self, feats: dict):

        # Project backbone feats to a common channel (dim).
        laterals = []
        for n in self.used_level_names:
            laterals.append(self.lateral_convs[n](feats[n]))
        # Top-down: upsample coarser level to finer resolution, add, repeat.
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            target_h, target_w = laterals[i - 1].shape[2], laterals[i - 1].shape[3]
            if self.upsample_mode == "nearest":
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=(target_h, target_w), mode="nearest"
                )
            else:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i],
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
        # 3×3 on each merged map (MMDet FPN output per used level).
        outs_list = []
        name_by_idx = list(self.used_level_names)
        for i in range(used_backbone_levels):
            outs_list.append(self.fpn_convs[name_by_idx[i]](laterals[i]))
        # More outputs than backbone levels: max-pool chain or extra convs.
        if self.num_outs > len(outs_list):
            if not self.add_extra_convs:
                out = outs_list[-1]
                for _ in range(self.num_outs - used_backbone_levels):
                    out = F.max_pool2d(out, 1, stride=2)
                    outs_list.append(out)
            else:
                if self.add_extra_convs == "on_input":
                    extra_source = feats[self.level_names[self.backbone_end_level - 1]]
                elif self.add_extra_convs == "on_lateral":
                    extra_source = laterals[-1]
                elif self.add_extra_convs == "on_output":
                    extra_source = outs_list[-1]
                else:
                    raise NotImplementedError
                outs_list.append(self.fpn_convs[self.extra_level_names[0]](extra_source))
                for j in range(1, self.num_outs - used_backbone_levels):
                    prev = outs_list[-1]
                    if self.relu_before_extra_convs:
                        prev = F.relu(prev)
                    outs_list.append(self.fpn_convs[self.extra_level_names[j]](prev))

        # Order: finest used level first, then coarser, then any extras (MMDet-style tuple).
        return tuple(outs_list)

class ResNet(nn.Module):
    """ResNet backbone with MapTR-style frozen_stages / norm_cfg / norm_eval support.

    Stage layout (matches MMDetection convention):
      frozen_stages=0  →  nothing frozen
      frozen_stages=1  →  stem + layer1  (conv1 in this codebase)
      frozen_stages=2  →  + layer2 (conv2)
      frozen_stages=3  →  + layer3 (conv3)
      frozen_stages=4  →  + layer4 (conv4)

    norm_cfg : dict
        type         – BN type string (currently only 'BN' is used)
        requires_grad – if False, γ/β of every BN in the backbone are frozen

    norm_eval : bool
        When True, ALL BN layers remain in eval() mode during training so that
        running-mean/var are never updated by batches (identical to MapTR behaviour).
    """

    def __init__(
        self,
        input_size,
        fpn_dim,
        resnet_model='resnet50',
        target_layers=['reduction_2', 'reduction_3', 'reduction_4'],
        skip_fpn=False,
        frozen_stages=1, # 0
        norm_cfg=dict(type='BN', requires_grad=False), # True
        norm_eval=True, # False
        **kwargs,
    ):
        super(ResNet, self).__init__()

        self.skip_fpn = skip_fpn
        self.target_layers = target_layers
        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        # torchvision 0.13+: use weights=... instead of deprecated pretrained=...
        use_pretrained = True
        weights_50 = ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
        weights_101 = ResNet101_Weights.IMAGENET1K_V1 if use_pretrained else None
        if (resnet_model == 'resnet50'):
            model_list = list(models.resnet50(weights=weights_50).children())
        elif (resnet_model == 'resnet101'):
            model_list = list(models.resnet101(weights=weights_101).children())
        else:
            sys.exit(f">> [Error] Unsupported resnet model: {resnet_model}")

        # conv1 = stem (conv+bn+relu+maxpool) + layer1  → frozen when frozen_stages >= 1
        conv0 = nn.Sequential(model_list[0], model_list[1], model_list[2], model_list[3])
        layer1 = model_list[4]
        self.conv1 = nn.Sequential(*conv0, *layer1)
        self.conv2 = model_list[5]   # layer2  (frozen when frozen_stages >= 2)
        self.conv3 = model_list[6]   # layer3  (frozen when frozen_stages >= 3)
        self.conv4 = model_list[7]   # layer4  (frozen when frozen_stages >= 4)

        # Freeze stages whose index <= frozen_stages
        self._freeze_stages()

        # Freeze BN gamma/beta across the whole backbone when requires_grad=False
        if not self.norm_cfg.get('requires_grad', True):
            self._freeze_bn_params()

        # output shapes
        img_H, img_W = input_size
        feature_size_by_input = {
            (900, 1600): FEATURE_SIZE_900x1600,
            (450, 800): FEATURE_SIZE_450x800,
            (480, 800): FEATURE_SIZE_480x800,
            (448, 960): FEATURE_SIZE_448x960,
            (224, 480): FEATURE_SIZE_224x480,
        }
        feature_size_table = feature_size_by_input.get((img_H, img_W))
        if feature_size_table is None:
            sys.exit(f">> [Error] Unsupported image size: {img_H}x{img_W}")
        if resnet_model not in feature_size_table:
            sys.exit(f">> [Error] Unsupported resnet model: {resnet_model}")
        FEATURE_SIZE = feature_size_table[resnet_model]

        self.output_shapes = {}
        for key in target_layers:
            self.output_shapes[key] = FEATURE_SIZE[key]

        self.FPN = None
        if not self.skip_fpn:
            level_names = list(self.target_layers)
            in_channels = [int(self.output_shapes[name][1]) for name in level_names]
            num_outs = len(level_names)
            self.FPN = FPN(
                dim=fpn_dim,
                level_names=level_names,
                in_channels=in_channels,
                num_outs=num_outs,
                start_level=0,
                end_level=-1,
                add_extra_convs='on_output',
                upsample_mode="nearest",
                relu_before_extra_convs=True,
            )
            for k in level_names:
                _, _, h, w = self.output_shapes[k]
                self.output_shapes[k] = torch.Size((1, fpn_dim, h, w))

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    def _freeze_stages(self):
        """Freeze parameters and put BN into eval for all frozen stages.

        Called both in __init__ and train() because model.train() recursively
        resets every submodule back to training mode.
        """
        # Stage index → submodule.  frozen_stages=1 freezes stages 0+1 (conv1).
        stage_modules = [self.conv1, self.conv2, self.conv3, self.conv4]
        for stage_idx in range(min(self.frozen_stages, len(stage_modules))):
            stage = stage_modules[stage_idx]
            # Put the entire stage (including its BN layers) into eval mode
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def _freeze_bn_params(self):
        """Set requires_grad=False for γ (weight) and β (bias) of every BN layer."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)

    # ------------------------------------------------------------------
    # train() override – re-apply freezing after model.train() resets children
    # ------------------------------------------------------------------

    def train(self, mode=True):
        """Override to keep frozen stages and BN layers in eval during training."""
        super(ResNet, self).train(mode)
        # Re-freeze stages (super().train() would have re-enabled them)
        self._freeze_stages()
        # Keep ALL BN layers in eval mode (use running stats, no batch updates)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):

        x1 = self.conv1(x)   # x 1/4
        x2 = self.conv2(x1)  # x 1/2
        x3 = self.conv3(x2)  # x 1/2
        x4 = self.conv4(x3)  # x 1/2

        features = {
            'reduction_1': x1,
            'reduction_2': x2,
            'reduction_3': x3,
            'reduction_4': x4,
        }

        # select target layers
        output = {}
        for layer in self.target_layers:
            output[layer] = features[layer]

        if not self.skip_fpn:
            fpn_out = self.FPN(output)
            return OrderedDict(zip(self.target_layers, fpn_out))
        return output

