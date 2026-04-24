import sys
from matplotlib.pyplot import xkcd
import torch
import torch.nn.functional as F
from torchvision import models
from torch import nn
from einops import rearrange

class Aux_tasks(nn.Module):
    def __init__(self, cfg, model_name='Scratch', isTrain=False):
        super().__init__()

        cfg_model = cfg[model_name]
        cfg_img_bb = cfg['Img_Backbone']
        cfg_ns = cfg['nuscenes']
        cfg_img = cfg_ns['image']
        cfg_aux = cfg['Aux_tasks']
        h_dim = cfg_model['h_dim']
        self.bev_h = cfg_ns['bev']['bev_h']
        self.bev_w = cfg_ns['bev']['bev_w']
        
        self.depthnet = None
        if cfg_model['aux_tasks']['depth'] and cfg_model['vt_encoder_type'] != 'lss': # depthnet is only supported for BEVFormer
            depthnet_type = cfg_model['depthnet_type']
            if depthnet_type == 'depthnet':
                from models.img_backbone.auxnet import DepthNet
                dbound = cfg_ns['grid_config']['depth']
                D = int((dbound[1] - dbound[0]) / dbound[2])
                self.depthnet = DepthNet(in_channels=h_dim, 
                                        mid_channels=h_dim, 
                                        context_channels=0, 
                                        depth_channels=D, 
                                        # feat_down_sample=cfg_img_bb[depthnet_type]['feat_down_sample'],
                                        grid_config=cfg_ns['grid_config'],
                                        mixed_precision=cfg['train']['bool_mixed_precision'],
                                        **cfg_aux[depthnet_type])
            else:
                sys.exit(f">> [Error] {depthnet_type} is not supported as a depthnet!!")

        self.pvsegnet = None
        if cfg_model['aux_tasks']['pv_seg']:
            from models.img_backbone.auxnet import PVSegNet
            self.pvsegnet = PVSegNet(in_channels=h_dim, 
                                     out_channels=cfg_model['aux_tasks']['seg_classes'], 
                                     mixed_precision=cfg['train']['bool_mixed_precision'], 
                                     pos_weight=cfg_aux['pvsegnet']['pos_weight'],
                                     weight=cfg_aux['pvsegnet']['loss_pvseg_weight'])

        self.bevsegnet = None
        if cfg_model['aux_tasks']['bev_seg']:
            from models.img_backbone.auxnet import BEVSegNet
            self.bevsegnet = BEVSegNet(in_channels=h_dim, 
                                       out_channels=cfg_model['aux_tasks']['seg_classes'], 
                                       mixed_precision=cfg['train']['bool_mixed_precision'], 
                                       pos_weight=cfg_aux['bevsegnet']['pos_weight'],
                                       weight=cfg_aux['bevsegnet']['loss_bevseg_weight'])


    def forward(self, x, bev_embed, cam2egos, intrin, post_rot=None, post_tran=None, aux_labels=None, isTrain=False):
        '''
        x : (B*N, C, H, W)
        bev_embed : (B, D, H, W)
        cam2egos : (B, N, 4, 4) = camera2ego
        intrin : (B, N, 3, 3) = camera intrinsic
        post_rot : (B, N, 3, 3) = camera post rotation
        post_tran : (B, N, 3) = camera post translation
        aux_labels : (depth, pv_seg, bev_seg)        
        '''

        gt_depths, gt_pv_semantic_masks, gt_semantic_masks = aux_labels
        depth, pv_seg, bev_seg = None, None, None
        if self.depthnet is not None and isTrain:
            B, N, _, _ = cam2egos.size()
            if intrin.shape[-2:] == (3, 3):
                z = torch.zeros((*intrin.shape[:-1], 1), dtype=intrin.dtype, device=intrin.device)
                bottom = torch.zeros((*intrin.shape[:-2], 1, 4), dtype=intrin.dtype, device=intrin.device)
                bottom[..., 0, 3] = 1
                intrin = torch.cat([torch.cat([intrin, z], dim=-1), bottom], dim=-2)

            if post_rot is None:
                post_rot = torch.zeros((B, N, 3, 3), dtype=intrin.dtype, device=intrin.device)

            if post_tran is None:
                post_tran = torch.zeros((B, N, 3), dtype=intrin.dtype, device=intrin.device)

            depth = self.depthnet(x, cam2egos, intrin, post_rot, post_tran, depth_labels=gt_depths)
        
        
        if self.pvsegnet is not None and isTrain:
            pv_seg = self.pvsegnet(x, label=gt_pv_semantic_masks)
        
        if self.bevsegnet is not None and isTrain:
            if bev_embed.dim() == 3:
                bev_embed = rearrange(bev_embed, 'b (h w) d -> b d h w', h=self.bev_h, w=self.bev_w)
            bev_seg = self.bevsegnet(bev_embed, label=gt_semantic_masks)

        return (depth, pv_seg, bev_seg)