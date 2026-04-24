from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, cfg, backbone, train_backbone, return_interm_layers=True):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        _return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        _strides, _num_channels = [4, 8, 16, 32], [256, 512, 1024, 2048]

        return_layers = {}
        self.stride, self.num_channels = [], []
        for _, (key, value) in enumerate(_return_layers.items()):
            if (_ in cfg['encoder']['feat_levels']):
                return_layers[key] = value
                self.stride.append(_strides[_])
                self.num_channels.append(_num_channels[_])

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        hidden_dim = cfg['encoder']['dim']
        input_proj_list = []
        for _, in_channels in enumerate(self.num_channels):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
        self.input_proj = nn.ModuleList(input_proj_list)

    def forward(self, x):
        x_intermediate = self.body(x)

        x_out = []
        for l in range(len(self.num_channels)):
            x_out.append(self.input_proj[l](x_intermediate[str(l)]))

        return x_out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, cfg, name='resnet50', train_backbone=True, return_interm_layers=True, dilation=False):

        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(cfg, backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2
