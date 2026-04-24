import sys
from matplotlib.pyplot import xkcd
import torch
import torch.nn.functional as F
from torchvision import models
from torch import nn
from einops import rearrange

class ImgBackbone(nn.Module):
    def __init__(self, cfg, model_name='Scratch', isTrain=False):
        super().__init__()

        from models.common import Normalize
        self.norm = Normalize()

        cfg_model = cfg[model_name]
        cfg_img_bb = cfg['Img_Backbone']
        cfg_ns = cfg['nuscenes']
        cfg_img = cfg_ns['image']
        target_h, target_w = self._target_hw(cfg_img['target_size']['h'], cfg_img['target_size']['w'], cfg_img['size_divisor'])

        self.grid_mask = None
        if cfg_model['use_grid_mask']:
            from utils.augmentation import GridMask
            self.grid_mask = GridMask(use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)


        img_bb_type = cfg_model['img_backbone_type']
        h_dim = cfg_model['h_dim']
        if ('resnet' in img_bb_type):
            from models.img_backbone.resnet import ResNet
            self.backbone = ResNet(**{'backbone_type': img_bb_type, 
                                      'target_layers': cfg_img_bb[img_bb_type]['target_layers'], 
                                      'skip_fpn': cfg_img_bb['skip_fpn'],
                                      'input_size': (target_h, target_w),
                                      'fpn_dim': h_dim,
                                      'frozen_stages': cfg_img_bb[img_bb_type]['frozen_stages'],
                                      'norm_eval': cfg_img_bb[img_bb_type]['norm_eval'],
                                      'norm_cfg': cfg_img_bb[img_bb_type]['norm_cfg']                                                 
                                      })
                           
        else:
            sys.exit(f">> [Error] _{img_bb_type}_ is not supported as a backbone!!")    


        # # Auxiliary tasks
        # from models.aux_tasks.aux_tasks import Aux_tasks
        # self.aux_tasks = Aux_tasks(cfg, model_name=model_name, isTrain=isTrain)


    def _target_hw(self, h: int, w: int, size_divisor: int):
        pad_h = (size_divisor - (h % size_divisor)) % size_divisor
        pad_w = (size_divisor - (w % size_divisor)) % size_divisor
        return h + pad_h, w + pad_w



    def __call__(self, x, isTrain=False):
        '''
        x : (B, N, C, H, W) or (B*N, C, H, W)
        '''

        # --------------------------------------------------
        # Image Backbone
        if x.dim() == 5:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)       
        x = self.norm(x)
        if self.grid_mask is not None and isTrain:
            x = self.grid_mask(x)
        x = self.backbone(x)   
        features = list(x.values())



        return features