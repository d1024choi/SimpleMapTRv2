from utils.functions import *
import torch
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from models.img_backbone.img_backbone import ImgBackbone
from models.vt_encoder.vt_encoder import VTEncoder
from models.map_decoder.map_decoder import MapDecoder

class Scratch(nn.Module):
    def __init__(self, cfg, rank=0, isTrain=False):
        super().__init__()

        
        self.cfg = cfg
        model_name = 'Scratch'
        dataset_name = cfg['dataset_name']
        self.cfg_model = self.cfg[model_name]
        self.vt_encoder_type = self.cfg_model['vt_encoder_type']

        # --------------------------------------------------
        # Image Backbone
        self.img_backbone = ImgBackbone(cfg, model_name=model_name)


        # --------------------------------------------------
        # VT Encoder
        self.vt_encoder = VTEncoder(cfg, model_name=model_name)


        # --------------------------------------------------
        # Map Decoder
        self.map_decoder = MapDecoder(cfg, model_name=model_name, dataset_name=dataset_name, isTrain=isTrain)


        # --------------------------------------------------
        # Auxiliary tasks
        from models.aux_tasks.aux_tasks import Aux_tasks
        self.aux_tasks = Aux_tasks(cfg, model_name=model_name, isTrain=isTrain)


        # Model size analysis
        if rank == 0:
            self._print_model_summary()


    def _print_model_summary(self):
        '''Print model modules with their names and parameter counts (in millions).'''
        print("\n" + "=" * 60)
        print(f"{'Module Name':<40} {'Params (M)':>10} {'Train (M)':>10}")
        print("=" * 60)

        total_params, trainable_params = 0, 0
        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += params
            trainable_params += trainable
            print(f"{name:<40} {params/1e6:>10.3f} {trainable/1e6:>10.3f}")

        print("-" * 60)
        print(f"{'TOTAL':<40} {total_params/1e6:>10.3f} {trainable_params/1e6:>10.3f}")
        print("=" * 60 + "\n")

    def _update_target_variable_list(self, inputs):
        if self.cfg_model['aux_tasks']['depth']:
            inputs.append('gt_depths')
        if self.cfg_model['aux_tasks']['pv_seg']:
            inputs.append('gt_pv_semantic_masks')
        if self.cfg_model['aux_tasks']['bev_seg']:
            inputs.append('gt_semantic_masks')
        return inputs

    def _return_aux_labels(self, seq_batch):
        aux_labels = [None, None, None]
        if self.cfg_model['aux_tasks']['depth']:
            aux_labels[0] = seq_batch[-1]['gt_depths']
        if self.cfg_model['aux_tasks']['pv_seg']:
            aux_labels[1] = seq_batch[-1]['gt_pv_semantic_masks']
        if self.cfg_model['aux_tasks']['bev_seg']:
            aux_labels[2] = seq_batch[-1]['gt_semantic_masks']
        return tuple(aux_labels)

    def forward(self, batch, dtype, isTrain=False):

        # -------------------------------------------
        # Data preparation (debug)
        inputs = ['images', 'intrinsics', 'lidar2cams', 'bev_aug_mat', 'can_bus', 'cam2egos', 'lidar2egos']
        inputs = self._update_target_variable_list(inputs)

        # conversion to sequential batch
        b, s = batch['images'].size(0), batch['images'].size(1)
        seq_batch = [{} for _ in range(s)]
        for t in range(s):
            for key in inputs:
                seq_batch[t][key] = batch[key][:, t].type(dtype).cuda()


        # gather input to model
        b, n, _, _, _ = seq_batch[-1]['images'].shape
        images = seq_batch[-1]['images'].flatten(0, 1)                                 # (b n) c h w
        intrinsics = seq_batch[-1]['intrinsics']                                       # b n 3 3
        lidar2cams = seq_batch[-1]['lidar2cams']                                       # b n 4 4
        bev_aug_mat = seq_batch[-1]['bev_aug_mat']                                     # b n 4 4
        can_bus = seq_batch[-1]['can_bus']                                             # b n can_bus_len
        cam2egos = seq_batch[-1]['cam2egos']                                          
        lidar2egos = seq_batch[-1]['lidar2egos']                                       # b n 4 4
        aux_labels = self._return_aux_labels(seq_batch)


        # -------------------------------------------
        # Extract features from images
        features = self.img_backbone(images, isTrain=isTrain)    

        # -------------------------------------------
        # From PV features to BEV features
        inputs_to_vt_encoder = {
            'features': features,
            'intrinsics': intrinsics,
            'cam2egos': cam2egos,
            'lidar2cams': lidar2cams,
            'lidar2egos': lidar2egos,
            'bev_aug_mat': bev_aug_mat,
            'can_bus': can_bus,
            'gt_depth': aux_labels[0]
        }
        # vt_enc_output = self.vt_encoder(**inputs_to_vt_encoder)
        vt_enc_output = self.vt_encoder(inputs_to_vt_encoder)


        # -------------------------------------------
        # Auxiliary tasks
        depth_output, pv_seg_output, bev_seg_output = self.aux_tasks(features[-1], vt_enc_output['bev'], cam2egos, intrinsics, aux_labels=aux_labels, isTrain=isTrain)


        # -------------------------------------------
        # From BEV features to map instances
        output = self.map_decoder(vt_enc_output['bev'], batch['polylines'], isTrain=isTrain)        


        # -------------------------------------------
        # Auxiliary Loss
        if (isTrain):
            if self.cfg_model['vt_encoder_type'] in ['lss']:
                output['depth_loss'] = vt_enc_output['depth_loss']
            else:
                if (isinstance(depth_output, dict)):
                    output['depth_loss'] = depth_output['depth_loss']

            if (isinstance(pv_seg_output, dict)):
                output['pvseg_loss'] = pv_seg_output['pvseg_loss']

            if (isinstance(bev_seg_output, dict)):
                output['bevseg_loss'] = bev_seg_output['bevseg_loss']

        return output
