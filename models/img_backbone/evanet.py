
import torch
import torch.nn as nn
import numpy as np
from models.backbone.eva_vit import EVAViT


# Default img_backbone configuration (used if not provided)
_default_img_backbone = dict(
        img_size=640, 
        patch_size=16,
        window_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4*2/3,
        window_block_indexes = (
        list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + list(range(12, 14)) + list(range(15, 17)) + list(range(18, 20)) + list(range(21, 23))
        ),
        qkv_bias=True,
        drop_path_rate=0.3,
        flash_attn=True,
        with_cp=True, 
        frozen=False)


class EvaNet(EVAViT):

    def __init__(self, cfg, img_backbone_cfg=None, isNormImg=True, isTrain=False):
        """
        Initialize EvaNet backbone.
        
        Args:
            cfg: Configuration object containing 'eva_vit' dict with architecture params.
                 If cfg['eva_vit'] contains architecture params (img_size, embed_dim, etc.),
                 they will be used. Otherwise, defaults are used.
            img_backbone_cfg: Dictionary of backbone configuration parameters.
                        If provided, this takes precedence over cfg.
                        If None, tries to extract from cfg['eva_vit'], else uses default.
                        Example: dict(img_size=640, patch_size=16, ...)
            isNormImg: Whether to apply ImageNet normalization in forward()
            isTrain: Whether in training mode
        """
        
        # Determine img_backbone config: priority: provided > cfg > default
        if img_backbone_cfg is None:
            
            # Try to extract architecture params from cfg['eva_vit']
            if cfg is not None and 'eva_vit' in cfg:
                eva_cfg = cfg['eva_vit']
                # Check if eva_cfg contains architecture parameters (not just metadata)
                architecture_keys = ['img_size', 'patch_size', 'window_size', 'in_chans', 
                                   'embed_dim', 'depth', 'num_heads', 'mlp_ratio', 
                                   'window_block_indexes', 'qkv_bias', 'drop_path_rate', 
                                   'flash_attn', 'with_cp']
                if any(key in eva_cfg for key in architecture_keys):
                    # Extract architecture params from cfg, excluding metadata
                    img_backbone_cfg = {k: v for k, v in eva_cfg.items() if k in architecture_keys}
                    # Also preserve frozen flag from cfg (not in architecture_keys but needed)
                    if 'frozen' in eva_cfg:
                        img_backbone_cfg['frozen'] = eva_cfg['frozen']
                    # Convert window_block_indexes from list to tuple if needed (for compatibility)
                    if 'window_block_indexes' in img_backbone_cfg and isinstance(img_backbone_cfg['window_block_indexes'], list):
                        img_backbone_cfg['window_block_indexes'] = tuple(img_backbone_cfg['window_block_indexes'])
                    # Use defaults for missing keys
                    for key, default_val in _default_img_backbone.items():
                        if key not in img_backbone_cfg:
                            img_backbone_cfg[key] = default_val
                    print(f">> Using EVAViT architecture parameters from config: {list(img_backbone_cfg.keys())}")
                else:
                    # No architecture params in cfg, use defaults
                    img_backbone_cfg = _default_img_backbone.copy()
                    print(f">> Using default EVAViT architecture parameters (cfg['eva_vit'] contains only metadata)")
            
            else:
                # No cfg or no eva_vit in cfg, use defaults
                img_backbone_cfg = _default_img_backbone.copy()
                print(f">> Using default EVAViT architecture parameters (no config provided)")
        
        else:
            print(f">> Using provided img_backbone_cfg configuration")
        
        super(EvaNet, self).__init__(**img_backbone_cfg)

        self.isTrain = isTrain
        self.isNormImg = isNormImg

        self.register_buffer('norm_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('norm_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Freeze all EVAViT parameters (backbone is untrainable)
        # Get frozen flag from cfg if not in img_backbone_cfg
        frozen_flag = img_backbone_cfg.get('frozen', cfg.get('eva_vit', {}).get('frozen', False) if cfg else False)
        if frozen_flag:
            for param in super(EvaNet, self).parameters():
                param.requires_grad = False
            # Set EVAViT to eval mode for memory efficiency (prevents gradient computation)
            super(EvaNet, self).eval()
            print(f">> EVAViT backbone is frozen (requires_grad=False, eval mode)")
        
        # Build projection layers (these will ALWAYS be trainable, regardless of frozen flag)
        encoder_dim = img_backbone_cfg['embed_dim']
        proj_dim = cfg['encoder']['dim']
        proj_layers = [nn.Conv2d(encoder_dim, encoder_dim//2, kernel_size=1, bias=True)]
        proj_layers.append(nn.GELU())  # GELU is smoother than ReLU and preserves more information
        proj_layers.append(nn.Conv2d(encoder_dim//2, proj_dim, kernel_size=1, bias=True))
        self.proj_layers = nn.Sequential(*proj_layers)
        
        # Ensure projection layers are ALWAYS trainable (regardless of frozen or isTrain)
        if isTrain:
            for param in self.proj_layers.parameters():
                param.requires_grad = True

        
    def forward(self, x):
        
        if (self.isNormImg):
            x = (x - self.norm_mean) / self.norm_std
        
        x = super(EvaNet, self).forward(x)     
        
        return {'reduction_1': self.proj_layers(x[0])}
