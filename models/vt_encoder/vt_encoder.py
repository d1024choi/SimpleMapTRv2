import sys
import torch
import torch.nn.functional as F
from torchvision import models
from torch import nn

class VTEncoder(nn.Module):
    def __init__(self, cfg, model_name='Scratch', isTrain=False):
        super().__init__()

        # cfg initialization
        cfg_model = cfg[model_name]
        cfg_ns = cfg['nuscenes']
        cfg_img = cfg_ns['image']
        cfg_bev = cfg_ns['bev']
        cfg_opt = cfg['Optimizer']
        cfg_aux = cfg['Aux_tasks']
        self.vt_encoder_type = cfg_model['vt_encoder_type']
        cfg_vt = cfg['VTEncoder'][self.vt_encoder_type]        
        target_h, target_w = self._target_hw(cfg_img['target_size']['h'], cfg_img['target_size']['w'], cfg_img['size_divisor'])        

        # bev settings, updated 260325
        pc_range = cfg_ns['pc_range'] # [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        w_meters = pc_range[3] - pc_range[0]
        h_meters = pc_range[4] - pc_range[1]
        z_meters = pc_range[5] - pc_range[2]
        
        self.mix_precision = cfg['train']['bool_mixed_precision']
        if ('bevformer' in self.vt_encoder_type):
            from models.vt_encoder.bevformer.encoder import BEVFormerEncoder
            n_lvl = len(cfg['Img_Backbone'][cfg_model['img_backbone_type']]['target_layers'])
            z_candi = (torch.linspace(0.5, z_meters-0.5, cfg_vt['num_z_candi']) + pc_range[2]).numpy().tolist() # -1.5 -0.5 0.5 1.5
            cfg_bevformer = {"bs" : cfg_opt['bs'],
                            "n_cam" : cfg_opt['n_cam'],
                            "dim" : cfg_model['h_dim'],
                            "z_candi" : z_candi,
                            "pc_range" : pc_range,
                            "num_points_in_pillar" : cfg_vt['num_z_candi'],
                            "h" : cfg_bev['bev_h'],
                            "w" : cfg_bev['bev_w'],
                            "sigma" : 1.0,
                            "bev_height" : cfg_bev['bev_h'],
                            "bev_width" : cfg_bev['bev_w'],
                            "h_meters" : h_meters,
                            "w_meters" : w_meters,
                            "offset" : 0.0,
                            "depth_thr" : 1.0,
                            "image_h" : target_h,
                            "image_w" : target_w,
                            "n_lvl" : n_lvl,
                            "repeat" : cfg_vt['repeat'],
                            "mixed_precision": self.mix_precision,
                            "len_can_bus": cfg_ns['can_bus_len'],
                            "use_can_bus": cfg['VTEncoder']['use_can_bus'],
                            "position_embedding_type": cfg['VTEncoder']['position_embedding_type']
                            }

            self.encoder = BEVFormerEncoder(cfg_bevformer)
        elif ('lss' in self.vt_encoder_type):
            from models.vt_encoder.lss.standalone_lss_transform import LSSTransformStandalone 
            cfg_lss = dict(
                in_channels      = cfg_model['h_dim'],
                out_channels     = cfg_model['h_dim'],
                feat_down_sample = cfg_aux['depthnet']['feat_down_sample'],
                pc_range         = pc_range,
                voxel_size       = cfg_ns['voxel_size'],
                dbound           = cfg_ns['grid_config']['depth'],
                downsample       = 2,
                loss_depth_weight= cfg_aux['depthnet']['loss_depth_weight'],
                depthnet_cfg     = dict(use_dcn=False, with_cp=False, aspp_mid_channels=96),
                grid_config      = cfg_ns['grid_config'])

            self.encoder = LSSTransformStandalone(**cfg_lss)
        else:
            sys.exit(f">> [Error] _{self.vt_encoder_type}_ is not supported as a VT encoder!!")    

    def _target_hw(self, h: int, w: int, size_divisor: int):
        pad_h = (size_divisor - (h % size_divisor)) % size_divisor
        pad_w = (size_divisor - (w % size_divisor)) % size_divisor
        return h + pad_h, w + pad_w

    def __call__(self, inputs: dict):
        '''
        features : [((b n) c h' w')]
        intrinsics : b n 3 3
        cam2egos : b n 4 4
        lidar2cams : b n 4 4
        lidar2egos : b n 4 4
        bev_aug_mat : b n 4 4
        can_bus : b can_bus_len

          (Top-down view grid) 

                |
                |
          -------------> x (100 pels)
                |         
                |                 
                v 
                y
             (200 pels)
        '''


        features = inputs['features']
        if self.mix_precision:
            if isinstance(features, (list, tuple)):
                features = [f.float() for f in features]
            else:
                features = features.float()

        with torch.amp.autocast("cuda", enabled=False):
            if self.vt_encoder_type == 'lss':
                out = self.encoder(features[-1], inputs['cam2egos'], inputs['intrinsics'], None, inputs['lidar2egos'])
                out['depth_loss'] = self.encoder.get_depth_loss(inputs['gt_depth'], out['depth'])
                return out
            elif self.vt_encoder_type == 'bevformer':
                bev = self.encoder(features, inputs['intrinsics'], inputs['lidar2cams'], inputs['bev_aug_mat'], inputs['can_bus'])
                return {'bev': bev}

