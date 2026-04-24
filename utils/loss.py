import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
import numpy as np
import torch.optim as optim
# from models.backbone.transform import *
import os
from typing import Optional
from fvcore.nn import sigmoid_focal_loss
import sys
from scipy.optimize import linear_sum_assignment
from models.common import normalize_2d_bbox, denormalize_2d_bbox, normalize_2d_pts, denormalize_2d_pts, reduce_mean

# Reduce noisy matplotlib debug logs (backend + font manager) when saving LR scheduler visuals.
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

DYNAMIC = ['pedestrian', 'vehicle']
STATIC_LARGE = ['drivable', 'walkway', 'carpark_area']
STATIC_SMALL = ['stop_line', 'ped_crossing', 'divider']


# --------------------------------
# Common
class Optimizers(nn.Module):
    """Optimizer wrapper supporting multiple optimizer types."""
    
    # Optimizer type mapping
    _OPTIMIZER_MAP = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
    }
    
    def __init__(self, model, optimizer_type, learning_rate, weight_decay, config=None):
        super().__init__()
        
        optimizer_type = optimizer_type.lower()
        if optimizer_type not in self._OPTIMIZER_MAP:
            sys.exit(f">> Optimizer {optimizer_type} is not supported! Available: {list(self._OPTIMIZER_MAP.keys())}")
        
        optimizer_class = self._OPTIMIZER_MAP[optimizer_type]

        # config['lr_mult'] = {'submodule_name': multiplier} enables per-module LR.
        # e.g. {'img_backbone': 0.1} → backbone LR = learning_rate * 0.1
        # Under DDP, model is wrapped so named_modules() has 'module.img_backbone'; use inner module for lookup.
        lr_mult = (config or {}).get('lr_mult', {})
        if lr_mult:
            base = model.module if hasattr(model, 'module') else model
            named_modules = dict(base.named_modules())
            grouped_ids = set()
            param_groups = []
            for mod_name, mult in lr_mult.items():
                mod = named_modules.get(mod_name)
                if mod is None:
                    sys.exit(f">> lr_mult: submodule '{mod_name}' not found in model.")
                mod_params = list(mod.parameters())
                param_groups.append({'params': mod_params, 'lr': learning_rate * mult})
                grouped_ids.update(id(p) for p in mod_params)
            remaining = [p for p in model.parameters() if id(p) not in grouped_ids]
            if remaining:
                param_groups.append({'params': remaining})
            self.opt = optimizer_class(param_groups, lr=learning_rate, weight_decay=weight_decay)
        else:
            self.opt = optimizer_class(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


class _WarmupCosineAnnealingLR_deprecated(optim.lr_scheduler._LRScheduler):
    """Linear warmup then cosine annealing, changed per-epoch.

    The scheduler may be stepped per-iteration, but the computed learning rate
    only updates at the beginning of each epoch and stays constant for all
    iterations inside that epoch.
    """

    def __init__(
        self,
        optimizer_,
        total_iters: int,
        warmup_iters: int = 0,
        warmup_ratio: float = 1.0 / 3,
        max_lr: float = 0.0005,
        min_lr_ratio: float = 1e-3,
        num_epochs: int = 100,
        iters_per_epoch: Optional[int] = None,
        last_epoch: int = -1,
    ):
        self.total_iters = int(total_iters)
        self.num_epochs = int(num_epochs)
        self.warmup_iters = int(max(0, warmup_iters))
        self.warmup_ratio = float(warmup_ratio)
        self.base_lr = float(max_lr)
        self.min_lr_ratio = float(min_lr_ratio)
        if self.total_iters <= 0:
            raise ValueError("total_iters must be > 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if iters_per_epoch is None:
            self.iters_per_epoch = int(np.ceil(self.total_iters / float(self.num_epochs)))
        else:
            self.iters_per_epoch = int(iters_per_epoch)
        if self.iters_per_epoch <= 0:
            raise ValueError("iters_per_epoch must be > 0")
        super().__init__(optimizer_, last_epoch=last_epoch)

    def get_lr(self):
        t = self.last_epoch  # iteration index if stepped per-iter; epoch index if stepped per-epoch
        base_lrs = self.base_lrs
        if t < 0:
            t = 0
        if t > self.total_iters:
            t = self.total_iters

        epoch = t // self.iters_per_epoch
        if epoch < 0:
            epoch = 0
        if epoch >= self.num_epochs:
            epoch = self.num_epochs - 1

        warmup_epochs = min(self.warmup_iters, self.num_epochs)
        if warmup_epochs > 0 and epoch < warmup_epochs:
            p = (epoch + 1) / float(warmup_epochs)
            scale = self.warmup_ratio + (1.0 - self.warmup_ratio) * p
            return [lr * scale for lr in base_lrs]

        remain_epochs = max(1, self.num_epochs - warmup_epochs)
        tt = epoch - warmup_epochs
        if tt < 0:
            tt = 0
        if tt > remain_epochs:
            tt = remain_epochs
        cos = 0.5 * (1.0 + np.cos(np.pi * tt / float(remain_epochs)))
        return [
            lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos)
            for lr in base_lrs
        ]

class _WarmupCosineAnnealingLR(optim.lr_scheduler._LRScheduler):
    """Linear warmup then cosine annealing.

    Stepping modes
    --------------
    * ``per_epoch=False`` (default, MapTR-style): stepped per iteration.
      ``total_iters`` / ``warmup_iters`` are in iteration units.
    * ``per_epoch=True``: LR is held constant within each epoch and only
      changes at epoch boundaries. Two usages are supported:
        - Step per iteration (keep caller unchanged): pass the actual
          ``steps_per_epoch`` and ``total_iters = steps_per_epoch * num_epochs``.
          ``last_epoch`` is internally quantized to the nearest epoch start.
        - Step per epoch: pass ``steps_per_epoch=1`` and
          ``total_iters = num_epochs``; ``warmup_iters`` is then in epoch units.
    """

    def __init__(
        self,
        optimizer_,
        total_iters: int,
        warmup_iters: int = 0,
        warmup_ratio: float = 1.0 / 3,
        max_lr: float = 0.0005,
        min_lr_ratio: float = 1e-3,
        num_epochs: int = 100,
        per_epoch: bool = False,
        steps_per_epoch: int = 1,
        last_epoch: int = -1,
    ):
        self.total_iters = int(total_iters)
        self.num_epochs = int(num_epochs)
        self.warmup_iters = int(max(0, warmup_iters))
        self.warmup_ratio = float(warmup_ratio)
        self.base_lr = float(max_lr)
        self.min_lr_ratio = float(min_lr_ratio)
        self.per_epoch = bool(per_epoch)
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        if self.total_iters <= 0:
            raise ValueError("total_iters must be > 0")
        super().__init__(optimizer_, last_epoch=last_epoch)

    def get_lr(self):
        t = self.last_epoch
        base_lrs = self.base_lrs
        if t < 0:
            t = 0
        if t > self.total_iters:
            t = self.total_iters
        if self.per_epoch and self.steps_per_epoch > 1:
            # Quantize down to the most recent epoch boundary so the LR
            # returned is constant across all iterations within an epoch.
            t = (t // self.steps_per_epoch) * self.steps_per_epoch
        if self.warmup_iters > 0 and t < self.warmup_iters:
            p = (t + 1) / float(self.warmup_iters)
            scale = self.warmup_ratio + (1.0 - self.warmup_ratio) * p
            return [lr * scale for lr in base_lrs]
        remain = max(1, self.total_iters - self.warmup_iters)
        tt = t - self.warmup_iters
        if tt < 0:
            tt = 0
        if tt > remain:
            tt = remain
        cos = 0.5 * (1.0 + np.cos(np.pi * tt / float(remain)))
        return [
            lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos)
            for lr in base_lrs
        ]


class LRScheduler(nn.Module):
    def __init__(self, optimizer, config=None, save_dir=None):
        super(LRScheduler, self).__init__()
        """ Required config keys
        1) StepLR : step_size, gamma
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html

        2) ExponentialLR : gamma
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR

        3) OneCycleLR : max_lr, div_factor, final_div_factor, pct_start, steps_per_epoch, epochs, cycle_momentum
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html

        4) CosineAnnealing (MapTR-style warmup + cosine)
           Required keys:
             - steps_per_epoch, epochs   (or alternatively total_iters)
             - warmup (None or 'linear')
             - warmup_iters (int)
             - warmup_ratio (float)  (initial_lr = base_lr * warmup_ratio)
             - min_lr_ratio (float)  (eta_min = base_lr * min_lr_ratio)
           Optional:
             - per_epoch (bool, default False)
               If True, LR is held constant within each epoch and only changes
               at epoch boundaries. The caller still steps the scheduler once
               per iteration (no training-loop change needed).
        """

        self.save_dir = save_dir
        type = config.get('type', None)
        if (type == 'StepLR' and config is not None):
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
        elif (type == 'ExponentialLR' and config is not None):
            self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])
        elif (type == 'OnecycleLR' and config is not None):
            self.scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                           max_lr=config['max_lr'],
                                                           div_factor=config['div_factor'], # starts at max_lr / 10
                                                           final_div_factor=config['final_div_factor'], # ends at lr / 10 / 10
                                                           pct_start=config['pct_start'], # reaches max_lr at 30% of total steps
                                                           steps_per_epoch=config['steps_per_epoch'],
                                                           epochs=config['epochs'],
                                                           cycle_momentum=False)
        elif (type == 'CosineAnnealing' and config is not None):
            warmup = config.get('warmup', None)
            warmup_iters = int(config.get('warmup_iters', 0))
            warmup_ratio = float(config.get('warmup_ratio', 1.0 / 3))
            min_lr_ratio = float(config.get('min_lr_ratio', 1e-3))
            max_lr = float(config.get('max_lr', 0.0005))

            if warmup not in (None, 'linear'):
                sys.exit(f"[Error] warmup='{warmup}' is not supported (use None or 'linear').")

            if 'total_iters' in config: 
                total_iters = int(config['total_iters'])
            else:
                total_iters = int(config['steps_per_epoch']) * int(config['epochs'])
            num_epochs = int(config['epochs'])
            per_epoch = bool(config.get('per_epoch', False))
            steps_per_epoch_cfg = int(config.get('steps_per_epoch', 1))

            if warmup is None:
                warmup_iters = 0
                warmup_ratio = 1.0

            self.scheduler = _WarmupCosineAnnealingLR(
                optimizer,
                total_iters=total_iters,
                warmup_iters=warmup_iters,
                warmup_ratio=warmup_ratio,
                min_lr_ratio=min_lr_ratio,
                num_epochs=num_epochs,
                max_lr=max_lr,
                per_epoch=per_epoch,
                steps_per_epoch=steps_per_epoch_cfg,
            )
        else:
            sys.exit(f'[Error] LR scheduler named {type} is not implemented!!')


        if True:
            self._visualize_lr_scheduler(config)



    def _visualize_lr_scheduler(self, config):
        # Visualization start ----
        sch_type = str(config.get('type', 'unknown'))
        steps_per_epoch = int(config.get('steps_per_epoch', 1))
        epochs = int(config.get('epochs', 1))
        total_iters = int(config.get('total_iters', max(1, steps_per_epoch * epochs)))

        dummy = nn.Linear(1, 1)
        lrs = []

        def _collect(dummy_opt, sched, n_steps):
            out = []
            for _ in range(n_steps):
                dummy_opt.step()
                sched.step()
                out.append(float(dummy_opt.param_groups[0]['lr']))
            return out

        if sch_type == 'OnecycleLR':
            d_opt = optim.Adam(dummy.parameters(), lr=float(config['max_lr']) / float(config['div_factor']))
            sch = optim.lr_scheduler.OneCycleLR(
                d_opt,
                max_lr=config['max_lr'],
                div_factor=config['div_factor'],
                final_div_factor=config['final_div_factor'],
                pct_start=config['pct_start'],
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                cycle_momentum=False,
            )
            n = int(steps_per_epoch * epochs)
            lrs = _collect(d_opt, sch, n)
        elif sch_type == 'CosineAnnealing':
            max_lr = float(config.get('max_lr', 0.0005))
            warmup = config.get('warmup', None)
            warmup_iters = int(config.get('warmup_iters', 0))
            warmup_ratio = float(config.get('warmup_ratio', 1.0 / 3))
            min_lr_ratio = float(config.get('min_lr_ratio', 1e-3))
            if 'total_iters' in config:
                n = int(config['total_iters'])
            else:
                n = int(config['steps_per_epoch']) * int(config['epochs'])
            if warmup is None:
                warmup_iters = 0
                warmup_ratio = 1.0
            per_epoch = bool(config.get('per_epoch', False))
            steps_per_epoch_cfg = int(config.get('steps_per_epoch', 1))
            d_opt = optim.Adam(dummy.parameters(), lr=max_lr)
            sch = _WarmupCosineAnnealingLR(
                d_opt,
                total_iters=max(1, n),
                warmup_iters=warmup_iters,
                warmup_ratio=warmup_ratio,
                min_lr_ratio=min_lr_ratio,
                max_lr=max_lr,
                per_epoch=per_epoch,
                steps_per_epoch=steps_per_epoch_cfg,
            )
            lrs = _collect(d_opt, sch, max(1, n))
        elif sch_type == 'StepLR':
            base_lr = float(config.get('base_lr', config.get('max_lr', 1e-3)))
            d_opt = optim.Adam(dummy.parameters(), lr=base_lr)
            sch = optim.lr_scheduler.StepLR(
                d_opt, step_size=config['step_size'], gamma=config['gamma']
            )
            lrs = _collect(d_opt, sch, total_iters)
        elif sch_type == 'ExponentialLR':
            base_lr = float(config.get('base_lr', config.get('max_lr', 1e-3)))
            d_opt = optim.Adam(dummy.parameters(), lr=base_lr)
            sch = optim.lr_scheduler.ExponentialLR(d_opt, gamma=config['gamma'])
            lrs = _collect(d_opt, sch, total_iters)
        else:
            return

        iters = np.arange(len(lrs))
        plt.figure(figsize=(10, 4))
        plt.plot(iters, lrs, linewidth=1.2)
        plt.xlabel('Iteration (scheduler step)')
        plt.ylabel('Learning rate')
        plt.title(f'LR schedule: {sch_type}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if self.save_dir is not None:
            out_path = os.path.join(self.save_dir, f'lr_scheduler_{sch_type}_vis.png')
        else:
            out_path = f'./lr_scheduler_{sch_type}_vis.png'
        plt.savefig(out_path, dpi=150)
        plt.close()
        # Visualization end ----



    def __call__(self):
        try:
            self.scheduler.step()
        except ValueError as e:
            # Handle case where scheduler tries to step beyond total_steps
            # This can happen due to off-by-one errors in batch counting
            if "Tried to step" in str(e) and "total steps" in str(e):
                # Silently skip the extra step
                pass
            else:
                raise



# --------------------------------
# BEV
class SpatialRegressionLoss(nn.Module):
    def __init__(self, norm, ignore_index=255, future_discount=1.0):
        super(SpatialRegressionLoss, self).__init__()
        self.norm = norm
        self.ignore_index = ignore_index
        self.future_discount = future_discount

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        assert len(prediction.shape) == 5, 'Must be a 5D tensor'
        # ignore_index is the same across all channels
        mask = target[:, :, :1] != self.ignore_index
        if mask.sum() == 0:
            return prediction.new_zeros(1)[0].float()

        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=-3, keepdims=True)

        seq_len = loss.shape[1]
        future_discounts = self.future_discount ** torch.arange(seq_len, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, seq_len, 1, 1, 1)
        loss = loss * future_discounts

        return loss[mask].mean()

class SegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False, top_k_ratio=1.0, future_discount=1.0):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount

    def forward(self, prediction, target):
        if target.shape[-3] != 1:
            raise ValueError('segmentation label must be an index-label with channel dimension = 1.')
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)
        loss = F.cross_entropy(
            prediction,
            target,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights.to(target.device),
        )

        loss = loss.view(b, s, h, w)

        future_discounts = self.future_discount ** torch.arange(s, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, s, 1, 1)
        loss = loss * future_discounts

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)

class ProbabilisticLoss(nn.Module):
    def forward(self, output):
        present_mu = output['present_mu']
        present_log_sigma = output['present_log_sigma']
        future_mu = output['future_mu']
        future_log_sigma = output['future_log_sigma']

        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
                present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                    2 * var_present)
        )

        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))

        return kl_loss

class SigmoidFocalLoss(torch.nn.Module):
    def __init__(self, alpha=-1.0, gamma=2.0, reduction='mean'):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, label):
        return sigmoid_focal_loss(pred, label, self.alpha, self.gamma, self.reduction)

class TopKBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, label_indices, min_visibility, use_top_k=False, top_k_ratio=1.0):
        super().__init__()

        self.label_indices = label_indices
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.min_visibility = min_visibility

    def calc_cross_entropy(self, pred, label, visibility=None, ignore_label_indices=False):

        if ignore_label_indices is False:
            if self.label_indices is not None:
                label = [label[:, idx].max(dim=1, keepdim=True).values for idx in self.label_indices]
                label = torch.cat(label, dim=1)

        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")

        if self.min_visibility is not None:
            mask = visibility >= self.min_visibility
            loss = loss[mask]

        if self.use_top_k is not None:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[0])
            loss, _ = torch.sort(loss, descending=True)
            loss = loss[:k]
        return loss.mean()

    def forward(self, pred, batch, alpha, beta, target):

        loss_bev = self.calc_cross_entropy(pred['bev'], batch['bev'].to(pred['bev']), batch['visibility'])
        loss_center = self.calc_cross_entropy(pred['center'], batch['center'].to(pred['bev']),
                                              batch['visibility'], ignore_label_indices=True)

        return alpha*loss_bev + beta*loss_center

class DiceLoss(nn.Module):
    def __init__(self, label_indices, min_visibility, smooth=1.0):
        super().__init__()

        self.label_indices = label_indices
        self.min_visibility = min_visibility
        self.smooth = smooth

    def calc_diceloss(self, pred, label, visibility=None, ignore_label_indices=False):

        if ignore_label_indices is False:
            if self.label_indices is not None:
                label = [label[:, idx].max(dim=1, keepdim=True).values for idx in self.label_indices]
                label = torch.cat(label, dim=1)

        if self.min_visibility is not None:
            mask = visibility >= self.min_visibility
            pred = torch.sigmoid(pred)[mask]
            label = label[mask]
        else:
            pred = torch.sigmoid(pred).view(-1)
            label = label.view(-1)

        intersection = (pred * label).sum()
        union = pred.sum() + label.sum()

        # dice coefficient
        dice = 2.0 * (intersection + self.smooth) / (union + 1e-10)

        # dice loss
        dice_loss = 1.0 - dice

        return dice_loss

    def forward(self, pred, batch, alpha, beta, target):

        loss_bev = self.calc_diceloss(pred['bev'], batch['bev'].to(pred['bev']), batch['visibility'])
        loss_center = self.calc_diceloss(pred['center'], batch['center'].to(pred['bev']),
                                              batch['visibility'], ignore_label_indices=True)

        return alpha*loss_bev + beta*loss_center

class BEVSegLoss(torch.nn.Module):
    """BEV segmentation loss calculation for multiple target classes."""
    
    def __init__(self, cfg, min_visibility=None):
        super().__init__()
        self.cfg = cfg
        self.target = cfg['BEV']['target']
        self.label_indices = cfg['BEV']['label_indices']
        self.min_visibility = min_visibility

        self.bce = SigmoidFocalLoss(
            alpha=cfg['Loss']['bce']['alpha'], 
            gamma=cfg['Loss']['bce']['gamma'], 
            reduction='none'
        )
        self.focal = SigmoidFocalLoss(
            alpha=cfg['Loss']['focal']['alpha'], 
            gamma=cfg['Loss']['focal']['gamma'], 
            reduction='none'
        )
        self.l1 = nn.L1Loss(reduction='none')
        self.l2 = nn.MSELoss(reduction='none')

    def _prepare_label_and_visibility(self, pred, label, label_indices, visibility, ignore_label_indices):
        """Prepare label and visibility tensors with proper sizing and indexing.
        
        Args:
            pred: Prediction tensor (b x c x h x w)
            label: Label tensor (b x c x h x w)
            label_indices: Indices to extract from label
            visibility: Visibility tensor (optional)
            ignore_label_indices: Whether to skip label indexing
        
        Returns:
            Tuple of (processed_label, processed_visibility)
        """
        # Extract label indices if needed
        if not ignore_label_indices and label_indices is not None:
            label = torch.cat([label[:, idx].max(dim=1, keepdim=True).values 
                              for idx in label_indices], dim=1)

        # Resize label/visibility to match prediction size if needed
        _, _, hp, _ = pred.size()
        _, _, hl, _ = label.size()
        
        if hp < hl:
            scale = hp / hl
            label = F.interpolate(label, scale_factor=scale, mode='nearest')
            if visibility is not None:
                visibility = F.interpolate(visibility.to(pred), scale_factor=scale, mode='nearest')
        
        return label, visibility

    def _apply_visibility_mask(self, loss, visibility):
        """Apply visibility mask to loss if min_visibility is set.
        
        Args:
            loss: Loss tensor
            visibility: Visibility tensor
        
        Returns:
            Masked loss tensor
        """
        if self.min_visibility is not None and visibility is not None:
            mask = visibility >= self.min_visibility
            loss = loss[mask]
        return loss

    def bce_loss(self, pred, label, label_indices=None, visibility=None, ignore_label_indices=False):
        """Calculate BCE loss with optional visibility masking."""
        label, visibility = self._prepare_label_and_visibility(
            pred, label, label_indices, visibility, ignore_label_indices
        )
        loss = self.bce(pred, label)
        return self._apply_visibility_mask(loss, visibility)

    def focal_loss(self, pred, label, label_indices=None, visibility=None, ignore_label_indices=False):
        """Calculate focal loss with optional visibility masking.
        
        Args:
            pred: Prediction tensor (b x 1 x h x w)
            label: Label tensor (b x 1 x h x w)
            label_indices: Indices to extract from label
            visibility: Visibility tensor (b x 1 x h x w, optional)
            ignore_label_indices: Whether to skip label indexing
        """
        label, visibility = self._prepare_label_and_visibility(
            pred, label, label_indices, visibility, ignore_label_indices
        )
        loss = self.focal(pred, label)
        return self._apply_visibility_mask(loss, visibility)

    def l1_loss(self, pred, label, visibility=None, instance=None):
        """Calculate L1 loss with optional visibility and instance masking.
        
        Args:
            pred: Prediction tensor (b x 2 x h x w)
            label: Label tensor (b x 2 x h x w)
            visibility: Visibility tensor (b x 1 x h x w, optional)
            instance: Instance tensor (b x 1 x h x w, optional)
        """
        if pred is None:
            return torch.zeros(1, device=label.device, dtype=label.dtype)

        loss = self.l1(pred, label)
        if visibility is not None and instance is not None:
            mask_visibility = visibility >= self.min_visibility
            mask_instance = instance > 0
            mask = torch.logical_and(mask_visibility, mask_instance)
            loss = loss[mask.repeat(1, pred.size(1), 1, 1)]
        return loss

    def main(self, pred, batch):
        """Calculate losses for all target classes.
        
        Args:
            pred: Prediction dictionary with class predictions
            batch: Batch dictionary with 'bev', 'center', 'visibility' keys
        
        Returns:
            Dictionary of losses for each target class
        """
        # Filter predictions to only include targets
        targets = [key for key in pred.keys() if key in self.target]
        if not targets:
            return {}

        # Prepare ground truth labels
        bev_gt = batch['bev'].to(pred[targets[0]][0])
        
        # Extract center and visibility if available
        visibility = None
        if 'visibility' in batch:
            visibility = {
                'vehicle': batch['visibility'][:, [0]],
                'pedestrian': batch['visibility'][:, [1]]
            }

        # Loss calculation strategies for different class types
        def calc_static_large_loss(target):
            # Just to be safe, 260112
            if (pred[target][0] is None):
                return torch.zeros(1).type(self.dtype).cuda()
            
            bce = self.bce_loss(pred[target][0], bev_gt, self.label_indices[target])
            focal = self.focal_loss(pred[target][0], bev_gt, self.label_indices[target])
            return 0.5 * (bce.mean() + focal.mean())

        def calc_static_small_loss(target):
            # Just to be safe, 260112
            if (pred[target][0] is None):
                return torch.zeros(1).type(self.dtype).cuda()
            
            focal = self.focal_loss(pred[target][0], bev_gt, self.label_indices[target])
            return focal.mean()

        def calc_dynamic_loss(target):
            # Just to be safe, 260112
            if (pred[target][0] is None):
                return torch.zeros(1).type(self.dtype).cuda()
            
            vis = visibility.get(target) if visibility else None
            focal = self.focal_loss(pred[target][0], bev_gt, self.label_indices[target], vis)
            return focal.mean()

        # -------------------------------------------
        # Basic BCE Loss
        losses = {}
        for target in targets:
            if target in STATIC_LARGE:
                loss_val = calc_static_large_loss(target)
            elif target in STATIC_SMALL:
                loss_val = calc_static_small_loss(target)
            elif target in DYNAMIC:
                loss_val = calc_dynamic_loss(target)
            else:
                sys.exit(f">> {target} is not supported for loss calculation!")
            
            losses[target] = {'loss': loss_val, 'weight': 1.0}

        return losses



# --------------------------------
# Online HD map
class FocalLossCost:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         alpha (int | float, optional): focal_loss alpha
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import FocalLossCost
         >>> import torch
         >>> self = FocalLossCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3236, -0.3364, -0.2699],
                [-0.3439, -0.3209, -0.4807],
                [-0.4099, -0.3795, -0.2929],
                [-0.1950, -0.1207, -0.2626]])
    """

    def __init__(self, weight=1., alpha=0.25, gamma=2, eps=1e-12):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight

class ClsCost:
    """Classification cost for Hungarian matching.
    
    Computes the classification cost matrix between predicted class probabilities
    and ground truth labels.
    
    Args:
        weight (float): Weight of the classification cost. Default 1.0.
        use_focal (bool): Whether to use focal loss style cost. Default True.
        gamma (float): Focal loss gamma parameter. Default 2.0.
        alpha (float): Focal loss alpha parameter. Default 0.25.
    """
    def __init__(self, weight=1.0, use_sigmoid=True, gamma=2.0, alpha=0.25, num_classes=3):

        self.weight = weight
        self.use_sigmoid = use_sigmoid
        self.num_classes = num_classes
        self.sigmoid_focal_loss = FocalLossCost(weight=weight, alpha=alpha, gamma=gamma)
    
    def __call__(self, cls_pred, gt_labels):
        """
        Compute classification cost matrix.
        
        Args:
            cls_pred (num_queries, num_classes), including background class
            gt_labels (num_gt,), NOT including background class index
        
        Returns:
            Tensor: Classification cost matrix of shape (num_queries, num_gt).
                Lower cost means better match.
        """

        if (self.use_sigmoid):
            # NOTE : 'cls_pred' must exclude the background class for sigmoid focal loss
            #        'gt_labels' must not include background class index
            assert gt_labels.max() < self.num_classes, "gt_labels must be less than num_classes"
            return self.sigmoid_focal_loss(cls_pred[:, :self.num_classes], gt_labels.long())

        # Convert logits to probabilities using softmax
        cls_pred = cls_pred.softmax(-1)

        # Create onehot vectors for gt labels
        gt_onehot = F.one_hot(gt_labels.long(), num_classes=cls_pred.size(-1))

        # Element-wise multiplication of cls_pred and gt_onehot
        match_probs = cls_pred.unsqueeze(1) * gt_onehot.unsqueeze(0) # (num_queries, num_gt, num_classes)
        cls_cost = -torch.log(match_probs.sum(dim=2) + 1e-8) # (num_queries, num_gt)
        
        return cls_cost * self.weight

class LineCost:
    """Line cost for Hungarian matching using Chamfer Distance.
    
    Computes the cost matrix between predicted lines and ground truth lines
    using Chamfer Distance, following MapTR assigner implementation.
    Handles multiple shifts in ground truth and returns the minimum cost.
    
    Args:
        weight (float): Weight of the line cost. Default 1.0.
        cost_type (str): Type of distance metric. 'chamfer', 'l1', or 'l2'. Default 'chamfer'.
    """
    def __init__(self, weight=1.0, cost_type='l1'):
        self.weight = weight
        self.cost_type = cost_type
            
    def __call__(self, line_pred, line_gt):
        """
        Compute line cost matrix between predicted and ground truth lines.
        Follows Mask2Map OrderedPtsL1Cost / MapTR assigner pattern.
        
        Args:
            line_pred (Tensor): Predicted lines.
                Shape (num_queries, num_points, 2).
            line_gt (Tensor): Ground truth lines with multiple shifts.
                Shape (num_gt, num_shifts, num_points, 2).
        
        Returns:
            line_cost (Tensor): Line cost matrix of shape (num_queries, num_gt).
                Lower cost means better match.
            order_index (Tensor): Best shift index for each query-gt pair.
                Shape (num_queries, num_gt).
        """
        num_queries = line_pred.size(0)
        num_gts, num_shifts, num_pts, num_coords = line_gt.shape

        if self.cost_type in ('l1', 'l2'):
            # Mask2Map / MapTR style: flatten points, single torch.cdist call
            p = 1 if self.cost_type == 'l1' else 2
            pred_flat = line_pred.view(num_queries, -1)                         # (num_queries, num_pts * 2)
            gt_flat = line_gt.flatten(2).view(num_gts * num_shifts, -1)         # (num_gts * num_shifts, num_pts * 2)
            line_cost_ordered = torch.cdist(pred_flat, gt_flat, p=p)            # (num_queries, num_gts * num_shifts)
            line_cost_ordered = line_cost_ordered.view(num_queries, num_gts, num_shifts)
        else:
            sys.exit(f'LineCost: cost_type={self.cost_type} is not supported.')
        
        # Take minimum over shifts to get best order (same as MapTR assigner)
        # line_cost: (num_queries, num_gt)
        # order_index: (num_queries, num_gt)
        line_cost, order_index = torch.min(line_cost_ordered, dim=2)
        
        return line_cost * self.weight, order_index

class MaskCost:
    """Mask cost for Hungarian matching using focal loss with chunking.
    
    Args:
        weight (float): Weight of the mask cost. Default 1.0.
        alpha (float): Focal loss alpha. Default 0.25.
        gamma (float): Focal loss gamma. Default 2.0.
        chunk_size (int): Queries per chunk (reduce if OOM). Default 16.
    """
    def __init__(self, weight=1.0, alpha=0.25, gamma=2.0, chunk_size=16):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.chunk_size = chunk_size
    
    def __call__(self, mask_pred, mask_gt):
        """
        Compute mask cost matrix: (num_queries, num_gt).
        
        Args:
            mask_pred: Predicted masks (logits). Shape (num_queries, h, w).
            mask_gt: Ground truth masks (binary). Shape (num_gt, h, w).
        """
        num_queries, num_gts = mask_pred.size(0), mask_gt.size(0)
        
        # Prepare tensors
        pred_prob = mask_pred.sigmoid()  # (num_queries, h, w)
        gt = mask_gt.float()             # (num_gt, h, w)
        cost_mat = torch.zeros(num_queries, num_gts, device=mask_pred.device)
        
        # Process in chunks to save memory
        for i in range(0, num_queries, self.chunk_size):
            j = min(i + self.chunk_size, num_queries)
            
            # Expand for broadcasting: (chunk, 1, h, w) vs (1, num_gt, h, w)
            p = pred_prob[i:j].unsqueeze(1)  # (chunk, 1, h, w)
            g = gt.unsqueeze(0)               # (1, num_gt, h, w)
            
            # Focal loss: pos_cost * gt + neg_cost * (1 - gt)
            neg = -(1 - self.alpha) * (p ** self.gamma) * (-(1 - p + 1e-8).log())
            pos = -self.alpha * ((1 - p) ** self.gamma) * (-(p + 1e-8).log())
            
            # (chunk, num_gt, h, w) -> (chunk, num_gt)
            cost_mat[i:j] = (pos * g + neg * (1 - g)).mean(dim=(2, 3))
        
        return cost_mat * self.weight

class BBoxL1Cost:
    """BBoxL1Cost.
    """

    def __init__(self, weight=1.):
        self.weight = weight


    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

class Assigner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Initialize classification cost function
        cls_weight = cfg.get('cls_cost_weight', 2.0)
        use_sigmoid = cfg.get('use_sigmoid', True)
        num_classes = cfg.get('num_classes', 3)
        self.cls_cost = ClsCost(weight=cls_weight, use_sigmoid=use_sigmoid, num_classes=num_classes)
        
        # Initialize bbox cost function
        self.pc_range = cfg.get('pc_range', [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0])
        bbox_weight = cfg.get('bbox_cost_weight', 1.0)
        self.reg_cost = BBoxL1Cost(weight=bbox_weight)


        # Initialize line cost function with Chamfer Distance (following MapTR)
        line_weight = cfg.get('line_cost_weight', 5.0)
        line_cost_type = cfg.get('line_cost_type', 'l1')
        self.line_cost = LineCost(
            weight=line_weight, 
            cost_type=line_cost_type
        )

        # NOTE : Deprecated        
        # # Initialize mask cost function (focal loss with chunking for memory efficiency)
        # mask_weight = cfg.get('mask_cost_weight', 1.0)
        # mask_chunk_size = cfg.get('mask_chunk_size', 16)
        # self.mask_cost = MaskCost(
        #     weight=mask_weight,
        #     chunk_size=mask_chunk_size
        # )

    def assign(self, line_pred, cls_pred, bbox_pred, line_gt, cls_gt, bbox_gt):
        '''
        line_pred : (num_lane_queries, num_points, 2), ** normalized (xy) format
        cls_pred : (num_lane_queries, num_classes), (includes background class)
        line_gt : (num_gt, num_shifts, num_points, 2), ** unnormalized (xy) format
        cls_gt : num_gt (does not include background class)
        bbox_pred : (num_lane_queries, 4), ** normalzed (cxcywh) format
        bbox_gt : (num_gt, 4), ** unnormalized (xyxy) format

        NOTE : All the coordinates follow (x, y) = (width, height) in BEV space.
        '''
        
        num_gts = cls_gt.size(0)
        num_queries = cls_pred.size(0)
        
        # Handle edge case: no ground truths
        if num_gts == 0:
            assigned_gt_inds = cls_pred.new_full((num_queries,), 0, dtype=torch.long)
            assigned_labels = cls_pred.new_full((num_queries,), -1, dtype=torch.long)
            return num_gts, assigned_gt_inds, assigned_labels, None
        
        # 1) Compute classification cost matrix (num_queries, num_gt)
        cls_cost = self.cls_cost(cls_pred, cls_gt.to(cls_pred))
        
        # 2) Compute bbox cost matrix (num_queries, num_gt)
        # NOTE : bbox_gt comes in (xyxy) format. Deprecated.
        # TODO : reflect that box coord. comes in (cxcywh) format.
        # normalized_bbox_gt = normalize_2d_bbox(bbox_gt, self.pc_range) # Both bbox_gt and bbox_pred come in (xywh) format.
        # reg_cost = self.reg_cost(bbox_pred[:, :4], normalized_bbox_gt[:, :4].to(bbox_pred))
        reg_cost = 0

        # 3) Compute line cost matrix
        # line_cost shape: (num_queries, num_gt)
        # order_index shape: (num_queries, num_gt)
        normalized_gt_pts = normalize_2d_pts(line_gt, self.pc_range)
        line_cost, order_index = self.line_cost(line_pred, normalized_gt_pts.to(line_pred))

        # 4) Mask cost - Deprecated
        # # Compute mask cost matrix (num_queries, num_gt)
        # mask_cost = self.mask_cost(mask_pred, mask_gt)
        mask_cost = 0

        # weighted sum of above three costs
        cost = cls_cost + line_cost + reg_cost + mask_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" ' "to install scipy first.")
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(line_pred.device).long()
        matched_col_inds = torch.from_numpy(matched_col_inds).to(line_pred.device).long()

        # 4. assign backgrounds and foregrounds
        assigned_gt_inds = cls_pred.new_full((num_queries,), 0, dtype=torch.long)
        assigned_labels = cls_pred.new_full((num_queries,), -1, dtype=torch.long)

        # assign foregrounds based on matching results
        cls_gt_gpu = cls_gt.to(cls_pred.device)
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = cls_gt_gpu[matched_col_inds].long()
        # return num_gts, assigned_gt_inds, assigned_labels, order_index, (matched_row_inds, matched_col_inds)

        pos_inds = (assigned_gt_inds > 0).nonzero(as_tuple=True)[0] # the same as 'matched_row_inds'
        neg_inds = (assigned_gt_inds == 0).nonzero(as_tuple=True)[0]
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1 # the same as 'matched_col_inds'

        return num_gts, pos_inds, neg_inds, pos_assigned_gt_inds, order_index

class ClassificationLoss(nn.Module):
    """Classification loss with label weights and avg_factor
    """
    def __init__(self, gamma=2.0, alpha=0.25, bg_weight=0.1, loss_weight=2.0, use_sigmoid=True, tar_weight=None):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.bg_weight = bg_weight
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        self.tar_weight = tar_weight

    def __call__(self, cls_score, label, label_weight, avg_factor=None):
        """
        Args:
            cls_score: (N, num_classes) logits, 
            label: (N,) class indices. ** num_classes = background **
            label_weight: (N,) per-sample weights.
            avg_factor: Scalar to normalize the loss.
        """

        num_queries = cls_score.size(0)
        num_classes = cls_score.size(-1) # 3
        if (avg_factor is None):
            avg_factor = 1.0

        if self.use_sigmoid:
            # cls_score = cls_score[:, :-1] # exclude background class
            target = F.one_hot(label, num_classes=num_classes+1).float() # including background class as 'num_classes+1'
            target = target[:, :-1].to(cls_score) # exclude background class
            focal_loss = sigmoid_focal_loss(cls_score, target, self.alpha, self.gamma, 'none')

            tar_weight = []
            if self.tar_weight is not None:
                for tar, weight in self.tar_weight.items():
                    tw = torch.ones((focal_loss.size(0), 1)).to(focal_loss)
                    tar_weight.append(tw * weight)
                tar_weight = torch.cat(tar_weight, dim=1)
                focal_loss = focal_loss * tar_weight

            return (focal_loss * self.loss_weight).sum() / avg_factor
        else:
            NotImplementedError(f">> [Error] ClassificationLoss is not supported for softmax-based focal loss!!")

class DistanceLoss(nn.Module):
    """Distance loss combining PtsL1Loss and PtsDirCosLoss (following mask2map).
    
    Args:
        pts_loss_weight (float): Weight for L1 point loss. Default 5.0.
        dir_loss_weight (float): Weight for direction cosine loss. Default 0.005.
        dir_interval (int): Interval for direction calculation. Default 1.
    """
    def __init__(self, pts_loss_weight=5.0, dir_loss_weight=0.005, dir_interval=1):
        super().__init__()
        self.pts_loss_weight = pts_loss_weight
        self.dir_loss_weight = dir_loss_weight
        self.dir_interval = dir_interval

    def pts_l1_loss(self, pred, target, weight, avg_factor):
        """L1 loss for points.
        
        Args:
            pred: (N, num_pts, 2) predicted points.
            target: (N, num_pts, 2) target points.
            weight: (N, num_pts, 2) per-point weights.
            avg_factor: Scalar to normalize the loss.
        """
        loss = F.l1_loss(pred, target, reduction='none')
        loss = (loss * weight).sum() / max(float(avg_factor), 1.0)
        return loss * self.pts_loss_weight

    def pts_dir_cos_loss(self, pred_dir, target_dir, weight, avg_factor):
        """Cosine loss for point directions.
        
        Args:
            pred_dir: (N, num_pts-dir_interval, 2) predicted direction vectors.
            target_dir: (N, num_pts-dir_interval, 2) target direction vectors.
            weight: (N, num_pts-dir_interval) per-direction weights.
            avg_factor: Scalar to normalize the loss.
        """
        # Normalize direction vectors
        pred_norm = F.normalize(pred_dir, p=2, dim=-1)
        target_norm = F.normalize(target_dir, p=2, dim=-1)
        
        # Cosine similarity: 1 - cos(theta) as loss
        cos_sim = (pred_norm * target_norm).sum(dim=-1)
        loss = 1 - cos_sim
        
        loss = (loss * weight).sum() / max(float(avg_factor), 1.0)
        return loss * self.dir_loss_weight

    def forward(self, pred, target, weight, avg_factor):
        """
        Args:
            pred: (N, num_pts, 2) predicted points (normalized coordinates).
            target: (N, num_pts, 2) target points (normalized coordinates).
            weight: (N, num_pts, 2) per-point weights.
            avg_factor: Scalar to normalize the loss.
        
        Returns:
            loss_pts: L1 point loss.
            loss_dir: Direction cosine loss.
        """
        # Filter out invalid samples (all-zero weights)
        valid_mask = weight.sum(dim=(-1, -2)) > 0
        if not valid_mask.any():
            return pred.sum() * 0, pred.sum() * 0
        
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        weight_valid = weight[valid_mask]
        
        # L1 loss on points
        loss_pts = self.pts_l1_loss(pred_valid, target_valid, weight_valid, avg_factor)
        
        # Direction loss: direction = pts[i+interval] - pts[i]
        pred_dir = pred_valid[:, self.dir_interval:, :] - pred_valid[:, :-self.dir_interval, :]
        target_dir = target_valid[:, self.dir_interval:, :] - target_valid[:, :-self.dir_interval, :]
        dir_weight = weight_valid[:, :-self.dir_interval, 0]  # Use first coord weight
        
        loss_dir = self.pts_dir_cos_loss(pred_dir, target_dir, dir_weight, avg_factor)
        
        return loss_pts, loss_dir

class MaskLoss(nn.Module):
    """Mask loss using Sigmoid Focal Loss.
    
    Args:
        loss_weight (float): Weight for the mask loss. Default 1.0.
        alpha (float): Focal loss alpha parameter. Default 0.25.
        gamma (float): Focal loss gamma parameter. Default 2.0.
    """
    def __init__(self, loss_weight=1.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, mask_pred, mask_target, mask_weight, avg_factor):
        """
        Args:
            mask_pred: (N, h, w) predicted mask logits.
            mask_target: (N, h, w) target masks (binary).
            mask_weight: (N, h, w) per-pixel weights.
            avg_factor: Scalar to normalize the loss.
        """
        # Filter out invalid samples (all-zero weights)
        valid_mask = mask_weight.sum(dim=(-1, -2)) > 0
        if not valid_mask.any():
            return mask_pred.sum() * 0
        
        pred_valid = mask_pred[valid_mask]
        target_valid = mask_target[valid_mask]

        loss = sigmoid_focal_loss(pred_valid, target_valid, self.alpha, self.gamma, 'none') # N x H x W
        return self.loss_weight * loss.mean(dim=(-1, -2)).sum() / max(float(avg_factor), 1.0)

class LossComputation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        pre_weight = {"divider": 1.0, "ped_crossing": 1.0, "boundary": 1.0}

        # Classification loss config (following mask2map: FocalLoss, use_sigmoid=True)
        loss_cls_cfg = cfg.get('loss_cls', {})
        gamma = loss_cls_cfg.get('gamma', 2.0)
        alpha = loss_cls_cfg.get('alpha', 0.25)
        loss_weight = loss_cls_cfg.get('loss_weight', 2.0)
        use_sigmoid = loss_cls_cfg.get('use_sigmoid', True)
        target = loss_cls_cfg.get('target', None)
        self.bg_cls_weight = loss_cls_cfg.get('bg_cls_weight', 0.1)
        self.sync_cls_avg_factor = loss_cls_cfg.get('sync_cls_avg_factor', True)

        tar_weight = {}
        for tar in target:
            tar_weight[tar] = pre_weight.get(tar, 0.1)        
        self.loss_cls = ClassificationLoss(
            gamma=gamma,
            alpha=alpha,
            bg_weight=self.bg_cls_weight,
            loss_weight=loss_weight,
            use_sigmoid=use_sigmoid,
            tar_weight=tar_weight
        )
        
        # Distance loss config (following mask2map: PtsL1Loss + PtsDirCosLoss)
        loss_pts_cfg = cfg.get('loss_pts', {})
        pts_loss_weight = loss_pts_cfg.get('loss_weight', 5.0)
        loss_dir_cfg = cfg.get('loss_dir', {})
        dir_loss_weight = loss_dir_cfg.get('loss_weight', 0.005)
        dir_interval = cfg.get('dir_interval', 1)
        
        self.loss_dist = DistanceLoss(
            pts_loss_weight=pts_loss_weight,
            dir_loss_weight=dir_loss_weight,
            dir_interval=dir_interval
        )
        
        # # Mask loss config (using Focal Loss)
        # loss_mask_cfg = cfg.get('loss_mask', {})
        # mask_loss_weight = loss_mask_cfg.get('loss_weight', 1.0)
        # mask_alpha = loss_mask_cfg.get('alpha', 0.25)
        # mask_gamma = loss_mask_cfg.get('gamma', 2.0)
        
        # self.loss_mask = MaskLoss(
        #     loss_weight=mask_loss_weight,
        #     alpha=mask_alpha,
        #     gamma=mask_gamma
        # )

    def __call__(self, preds, targets, num_total_pos, num_total_neg):
        '''
        Args:
            preds (dict):
                - pos: b num_lane_queries num_points 2
                - cls: b num_lane_queries num_classes, ** including background class **
                - mask: b num_lane_queries h w
            targets (dict):
                - labels: b num_lane_queries, ** class indices. num_classes-1 = background **
                - label_weights: b num_lane_queries
                - pos: b num_lane_queries num_points 2
                - pos_weights: b num_lane_queries num_points 2
                - mask: b num_lane_queries h w
                - mask_weights: b num_lane_queries h w
            num_total_pos (int): total positive samples
            num_total_neg (int): total negative samples
        '''
        # Classification loss (following mask2map_head_2p.py 735-743)
        cls = preds['cls']
        labels = targets['labels']
        label_weights = targets['label_weights']
        
        cls_scores = cls.reshape(-1, cls.size(-1))
        labels_flat = labels.reshape(-1)
        label_weights_flat = label_weights.reshape(-1)
        
        
        # Classification Loss
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor])).item()

        cls_avg_factor = max(cls_avg_factor, 1)        
        loss_cls = self.loss_cls(cls_scores, labels_flat, label_weights_flat, avg_factor=cls_avg_factor)

        # Distance loss (following mask2map_head_2p.py 766-798)
        pos_pred = preds['pos'].reshape(-1, preds['pos'].size(-2), preds['pos'].size(-1))
        pos_target = targets['pos_targets'].reshape(-1, targets['pos_targets'].size(-2), targets['pos_targets'].size(-1))
        pos_weight = targets['pos_weights'].reshape(-1, targets['pos_weights'].size(-2), targets['pos_weights'].size(-1))
        
        loss_pts, loss_dir = self.loss_dist(pos_pred, pos_target, pos_weight, avg_factor=num_total_pos)
        # loss_pts = loss_pts / float(preds['pos'].size(-2))

        # # NOTE : Deprecated
        # # Mask loss
        # mask_pred = preds['mask'].reshape(-1, preds['mask'].size(-2), preds['mask'].size(-1))
        # mask_target = targets['mask_targets'].reshape(-1, targets['mask_targets'].size(-2), targets['mask_targets'].size(-1))
        # mask_weight = targets['mask_weights'].reshape(-1, targets['mask_weights'].size(-2), targets['mask_weights'].size(-1))
        # loss_mask = self.loss_mask(mask_pred, mask_target, mask_weight, avg_factor=num_total_pos)
        loss_mask = 0.0


        lane_loss = loss_cls + loss_pts + loss_dir + loss_mask
        return {'lane_loss': lane_loss, 'loss_cls': loss_cls, 'loss_pts': loss_pts, 'loss_dir': loss_dir, 'loss_mask': loss_mask}

class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight, loss_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        self.loss_weight = loss_weight

    def forward(self, ypred, ytgt):
        # import ipdb;ipdb.set_trace()
        loss = self.loss_fn(ypred, ytgt)
        return loss*self.loss_weight













