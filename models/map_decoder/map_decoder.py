import sys
import torch
import torch.nn.functional as F
from torchvision import models
from torch import nn
from models.common import xavier_init
import copy    
from einops import rearrange

# NOTE : This part should correspond to class MapTRHead() (/project/mmdet3d_plugin/maptr/dense_heads/maptr_head.py)
class MapDecoder(nn.Module):
    def __init__(self, cfg, model_name='Scratch', dataset_name='nuscenes', isTrain=False):
        super().__init__()

        # cfg initialization
        cfg_model = cfg[model_name]
        cfg_ds = cfg[dataset_name]
        self.map_decoder_type = cfg_model['map_decoder_type']

        if ('maptr' in self.map_decoder_type):
            from models.map_decoder.maptr.maptr import MapTR
            self.maptr = MapTR(cfg, model_name=model_name, dataset_name=dataset_name, isTrain=isTrain)
        else:
            sys.exit(f">> [Error] _{self.map_decoder_type}_ is not supported as a VT encoder!!")    


    def forward(self, bev_embed, gt_label, isTrain=False):
        '''
        bev_embed : b (bev_h*bev_w) dim

          (Top-down view grid for bev_embed) 

                |
                |
          -------------> x (100 pels)
                |         
                |                 
                v 
                y
             (200 pels)        
        '''

        if bev_embed.dim() == 4:
            bev_embed = rearrange(bev_embed, 'b d h w -> b (h w) d')

        batch_size = bev_embed.size(0)

        if ('maptr' in self.map_decoder_type):
            with torch.amp.autocast("cuda", enabled=False):
                pred = self.maptr(bev_embed, isTrain)
                if (isTrain):
                    loss = self.maptr.loss(pred['outs_one2one'], gt_label, batch_size)
                    pred.update(loss)
                    if (pred['outs_one2many'] is not None):
                        loss = self.maptr.loss(pred['outs_one2many'], gt_label, batch_size)
                        loss_key_updated = {}
                        for key in loss.keys():
                            loss_key_updated[f'{key}_one2many'] = loss[key]
                        pred.update(loss_key_updated)
                return pred
        else:
            NotImplementedError(f">> [Error] _{self.map_decoder_type}_ is not supported as a MapDecoder!!")
