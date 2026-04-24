"""
Unofficial implementation of BEVFormer (ECCV22)
https://arxiv.org/abs/2203.17270
"""

from utils.functions import *
import torch
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from models.vt_encoder.bevformer.encoder import BEVFormerEncoder


class BEVformer(nn.Module):
    def __init__(self, cfg, rank=0, isTrain=False):
        super().__init__()

        self.cfg = cfg
        cfg_bf, cfg_bev = cfg['BEVFormer'], cfg['BEV']
        
        # Training setting
        self.cfg, self.rank = cfg, rank
        self.h, self.w = cfg_bf['encoder']['init_qmap_size'], cfg_bf['encoder']['init_qmap_size']

        # Embeddings (initialized before backbone to be available in load_pretrained)
        self.bev_queries = nn.Embedding(self.h * self.w, cfg_bf['encoder']['dim'])

        # ---------------------------------------------------
        # Backbone
        from models.common import Normalize
        self.norm = Normalize()
        
        if ('resnet' in cfg_bf['backbone_type']):
            from models.backbone.resnet import ResNet
            self.backbone = ResNet(cfg=cfg, resnet_model=cfg_bf['backbone_type'], target_layers=cfg_bf[cfg_bf['backbone_type']]['target_layers'], skip_fpn=False)
        else:
            sys.exit(f">> [Error] _{cfg_bf['backbone_type']}_ is not supported as a backbone!!")        
        
        
        # ---------------------------------------------------
        # Encoder
        self.BEVFormerEncoder = BEVFormerEncoder(cfg=cfg) 


        # ---------------------------------------------------
        # Decoder
        from models.decoder import CVTDecoder
        cfg_dec = {'input_channels' : cfg_bf['encoder']['dim'], 
                   'out_dim' : 1,
                   'decoder_dim' : cfg_bf['decoder']['dim'], 
                   'num_decoder_blocks' :  int(math.log2(float(cfg_bev['bev']['h'] / cfg_bf['encoder']['init_qmap_size']))), 
                   'target_list' : cfg_bev['target']}
        self.CVTDecoder = CVTDecoder(**cfg_dec)

        # Initialize
        self._reset_parameters()
        
        # Model size analysis
        if rank == 0:
            self.print_model_summary()

    def print_model_summary(self):
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

    def _reset_parameters(self):
        xavier_uniform_(self.bev_queries.weight)

    def forward(self, batch, dtype, isTrain=True):

        # -------------------------------------------
        # Data preparation (debug)
        inputs = ['image', 'intrinsics', 'extrinsics', 'bev_aug_mat', 'bev_no_aug', 'bev']
        output = {}

        # conversion to sequential batch
        b, s = batch['image'].size(0), batch['image'].size(1)
        seq_batch = [{} for _ in range(s)]
        for t in range(s):
            for key in inputs:
                seq_batch[t][key] = batch[key][:, t].type(dtype).cuda()


        # gather input to model
        b, n, _, _, _ = seq_batch[-1]['image'].shape
        image = seq_batch[-1]['image'].flatten(0, 1)                                # (b n) c h w
        I = seq_batch[-1]['intrinsics']                                             # b n 3 3
        E = seq_batch[-1]['extrinsics']                                             # b n 4 4
        bev_aug_mat = seq_batch[-1]['bev_aug_mat']                                           # b n 4 4

        # -------------------------------------------
        # Extract features from images
        if (self.cfg['BEVFormer']['img_norm']): image = self.norm(image)
        features = self.backbone(image)    
        features = list(features.values())


        # -------------------------------------------
        # bevformer encoding
        
        # bev_queries, (h w) d -> b (h w) d
        bev_queries = self.bev_queries.weight[None].repeat(b, 1, 1)                         # b (h*w) d

        x = self.BEVFormerEncoder(bev_queries, None, features, I, E, bev_aug_mat)  # b (h*w) d
        x = rearrange(x, 'b (h w) d -> b d h w', h=self.h, w=self.w)

 
        # decoding
        output.update(self.CVTDecoder(x))
        
        return output
