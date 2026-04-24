import os
import sys
import csv
import pickle
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

from utils.functions import (read_config, config_update, save_read_latest_checkpoint_num, 
                             remove_past_checkpoint)
from utils.metrics import PolylinemAPMetric

from utils.print import (print_training_info, ANSI_COLORS, progress_bar, format_metrics, format_metrics_inline, save_config_as_document)
from models.common import denormalize_2d_bbox, denormalize_2d_pts

RENEW_KEYS = ['lane_loss', 'lane_loss_one2many', 'iter', 'loss_cls', 'loss_pts', 'loss_dir', 'loss_mask', 'depth_loss', 'pvseg_loss', 'bevseg_loss']  

# ANSI color codes (for terminal only)
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
WIDTH = 70


class LossTracker:
    def __init__(self, num_layers, keys=['cls', 'pts', 'dir']):
        self.losses = {'depth_loss': 0.0, 'pvseg_loss': 0.0, 'bevseg_loss': 0.0}
        self.num_layers = num_layers
        self.iter = 0
        self.learning_rate = 0.0
        for l in range(num_layers):
            for key in keys:
                self.losses[f'{l}_layer_loss_{key}'] = 0.0


    def update(self, losses, iter, learning_rate):
        self.iter = iter
        self.learning_rate = learning_rate
        for key in self.losses.keys():
            if key not in losses:
                continue
            v = losses[key]
            if torch.is_tensor(v):
                self.losses[key] += float(v.detach().item())
            else:
                self.losses[key] += float(v)

    def reset(self):
        self.iter = 0
        self.learning_rate = 0.0
        for key in self.losses.keys():
            self.losses[key] = 0.0

    def normalize(self):
        for key in self.losses.keys():
            self.losses[key] /= max(1, float(self.iter))


class Solver:
    '''BEVFormer model solver for training, evaluation, and checkpoint management.'''

    def __init__(self, args, num_batches, world_size=None, rank=None, logger=None, dtype=None, isTrain=True):

        '''
        Initialize the CVT model solver.

        Args:
            args: Argument parser containing model configuration.
            num_batches : Number of batches.
            world_size: Number of processes for distributed training.
            rank: Rank of the process.
            logger: Logger object for logging.
            dtype: PyTorch tensor dtype (e.g., torch.FloatTensor).
            isTrain: Whether to train the model.
        '''


        # save folder path
        self.save_dir = os.path.join('./saved_models/', f'{args.dataset_type}_{args.model_name}_model{args.exp_id}')
        self.args, self.rank, self.world_size = args, rank, world_size
        self.log, self.dtype = logger, dtype
        self.num_batches = num_batches

        # Load or save configuration
        self._handle_config(args, isTrain)
        
        if (self.rank == 0 and isTrain):
            save_config_as_document(self.cfg, self.save_dir)


        # Print training info (only on rank 0)
        if self.rank == 0:
            print_training_info(self.args, logger, return_print_dict())
        
        # mornitoring variables
        self.monitor = {'iter': 0, 'total_loss': 0, 'best_mAP': -1.0,'cur_lr': args.learning_rate}
        for key in RENEW_KEYS: self.monitor[key] = 0 
        decoder_type = self.cfg[args.model_name]['map_decoder_type']
        self.loss_tracker = LossTracker(self.cfg['MapDecoder'][decoder_type]['num_layers'], keys=['cls', 'pts', 'dir'])
        self._loss_csv_path = os.path.join(self.save_dir, 'layer_losses.csv')
        if self.rank == 0:
            self._init_loss_csv()
        # self.text_quality_metric = TextQualityMetric()
        # self.current_epoch = -1  # Track current epoch for beta scheduling
        # self.last_beta_update_epoch = -1  # Track last epoch we updated beta for

        # Model setup
        from models.Scratch.Scratch import Scratch
        model = Scratch(self.cfg, rank=rank, isTrain=isTrain)
        self.model = self._setup_model(model, args, dtype, rank)


        # Optimizer, loss, and scheduler
        from utils.loss import Optimizers, LRScheduler
        # backbone_lr_mult = float(getattr(args, 'backbone_lr_mult', 0.1))
        backbone_lr_mult = self.cfg['Optimizer'].get('backbone_lr_mult', 0.1)
        self.opt = Optimizers(self.model, args.optimizer_type, args.learning_rate, args.weight_decay,
                              config={'lr_mult': {'img_backbone': backbone_lr_mult}}).opt
        self.weights = {k: getattr(args, f'w_{k}') for k in ['alpha', 'beta', 'gamma']}

        # Mixed precision (AMP) configuration
        # Controlled by args.bool_mixed_precision (0/1), mirroring MapTR fp16 config usage.
        # fp16_loss_scale: initial scale for loss (MapTR uses 512) to prevent gradient underflow in FP16.
        self.use_fp16 = bool(getattr(args, 'bool_mixed_precision', 0))
        loss_scale = float(getattr(args, 'fp16_loss_scale', 512.0))
        self.scaler = GradScaler(init_scale=loss_scale, enabled=self.use_fp16)

        self.cfg_lr_scheduler = self.cfg['LR_Scheduler'].get(args.lr_schd_type, None)
        if self.cfg_lr_scheduler is None:
            raise ValueError(f"Invalid LR scheduler type: {args.lr_schd_type}")

        self.cfg_lr_scheduler.update({'type': args.lr_schd_type,
            'steps_per_epoch': self.num_batches,
            'epochs': args.num_epochs,
            'max_lr': args.learning_rate,
            'per_epoch': False # update, 260420
        })
        self.lr_scheduler = LRScheduler(self.opt, config=self.cfg_lr_scheduler, save_dir=self.save_dir) 
      


        # Load pretrained if needed
        if args.load_pretrained:
            self.load_pretrained_network_params(save_read_latest_checkpoint_num(self.save_dir, 0, isSave=False))

        if rank == 0:
            print(f">> Optimizer loaded from {os.path.basename(__file__)}")

    def _init_loss_csv(self):
        """Create layer_losses.csv with a header row if it does not yet exist."""
        if not os.path.exists(self._loss_csv_path):
            header = ['epoch']
            for l in range(self.loss_tracker.num_layers):
                for suffix in ('cls', 'pts', 'dir'):
                    header.append(f'L{l}_{suffix}')
            header += ['depth_loss', 'pvseg_loss', 'bevseg_loss', 'learning_rate']
            with open(self._loss_csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(header)

    def _handle_config(self, args, isTrain):
        '''Handle configuration loading/saving.'''
        config_path = os.path.join(self.save_dir, 'config.pkl')
        config_dict_path = os.path.join(self.save_dir, 'config_dict.pkl')

        if isTrain:
            if args.load_pretrained:
                if not os.path.exists(self.save_dir):
                    sys.exit(f'>> Path {self.save_dir} does not exist!')
                with open(config_path, 'rb') as f:
                    self.args = pickle.load(f)
                self.args.load_pretrained = 1
                self.args.ddp = args.ddp # updated, 260417
            elif self.rank == 0:
                with open(config_path, 'wb') as f:
                    pickle.dump(args, f)

            self.cfg = config_update(read_config(), self.args)
            with open(config_dict_path, 'wb') as f:
                pickle.dump(self.cfg, f)
        else:
            if os.path.exists(config_dict_path):
                with open(config_dict_path, 'rb') as f:
                    self.cfg = pickle.load(f)
            else:
                self.cfg = config_update(read_config(), args)

    def _setup_model(self, model, args, dtype, rank):
        '''Setup model for single or multi-GPU training.
        
        Preserves dtype of trainable parameters (e.g., LoRA adapters in float32)
        and frozen LLM parameters (in float16), while converting other non-trainable
        parameters to the target dtype.
        '''
        # Get the target dtype (e.g., torch.float32 from torch.FloatTensor)
        target_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
        
        # Preserve dtypes for parameters:
        for param in model.parameters():
            if param.requires_grad:
                continue
            elif param.dtype == torch.float16:
                continue
            else:
                param.data = param.data.to(target_dtype)
        
        # Move model to device (this doesn't change dtypes)
        if args.ddp:
            model.to(rank)
            ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
            return ddp_model

        return model.cuda()


    # --- Mode & Loss Management ---
    def mode_selection(self, isTrain=True, epoch=None):
        """
        Set model to train or eval mode.
        
        Args:
            isTrain: If True, set to train mode; if False, set to eval mode
            epoch: Optional epoch number. If provided and isTrain=True, updates beta schedule.
                   If None, will track epoch internally via learning_rate_step calls.
        """
        if isTrain:
            self.model.train()
        else:
            self.model.eval()


    def init_loss_tracker(self):
        self.loss_tracker.reset()
        for key in RENEW_KEYS:
            self.monitor[key] = 0

    def normalize_loss_tracker(self):
        self.loss_tracker.normalize()
        for key in RENEW_KEYS:
            self.monitor[key] /= self.num_batches

    def learning_rate_step(self, _e=None):
        if self.args.apply_lr_scheduling:
            self.lr_scheduler()

        


    # --- Checkpoint Management ---
    def load_pretrained_network_params(self, ckp_idx):
        file_name = f'{self.save_dir}/saved_chk_point_{ckp_idx}.pt'
        checkpoint = torch.load(file_name, map_location='cpu')

        if self.args.ddp:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in checkpoint['model_state_dict'].items())
            self.model.load_state_dict({k: v for k, v in state_dict.items() if k in self.model.state_dict()})

        self.opt.load_state_dict(checkpoint['opt'])
        self.lr_scheduler.scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        self.monitor.update({'iter': checkpoint['iter'], 'best_mAP': checkpoint.get('best_mAP', 0.0)})
        self.cfg = checkpoint['cfg']
        if self.rank == 0:
            self.log.info(f'>> Loaded parameters from {file_name}')
            self.log.info(f">> Best mAP: {self.monitor['best_mAP']:.4f}")
        
        
    def save_trained_network_params(self, e):
        save_read_latest_checkpoint_num(self.save_dir, e, isSave=True)
        file_name = f'{self.save_dir}/saved_chk_point_{e}.pt'
        torch.save({
            'epoch': e, 'model_state_dict': self.model.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(), 'opt': self.opt.state_dict(),
            'iter': self.monitor['iter'], 'cfg': self.cfg, 'best_mAP': self.monitor['best_mAP']
        }, file_name)
        self.log.info(">> Network saved")
        remove_past_checkpoint(self.save_dir, max_num=self.args.max_num_chkpts)

    # --- Progress Printing ---
    def print_status(self, e, start_epoch, end_epoch):

        if self.rank == 0:
            csv_row = [e]
            for l in range(self.loss_tracker.num_layers):
                parts = []
                for suffix in ('cls', 'pts', 'dir'):
                    lk = f'{l}_layer_loss_{suffix}'
                    v = self.loss_tracker.losses.get(lk, 0.0)
                    parts.append(f"{suffix}={v:.4f}")
                    csv_row.append(f"{v:.6f}")
                self.log.info(f"   layer {l}:  " + "  ".join(parts))
            csv_row += [
                f"{self.loss_tracker.losses.get('depth_loss', 0.0):.6f}",
                f"{self.loss_tracker.losses.get('pvseg_loss', 0.0):.6f}",
                f"{self.loss_tracker.losses.get('bevseg_loss', 0.0):.6f}",
                f"{float(getattr(self.loss_tracker, 'learning_rate', 0.0)):.10g}",
            ]
            with open(self._loss_csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(csv_row)

        if self.rank == 0:
            C = ANSI_COLORS
            hrs_left = (end_epoch - start_epoch) * (self.args.num_epochs - e - 1) / 3600.0
            
            # ========== EASILY ADD/MODIFY METRICS HERE ==========
            metrics = [
                {'label': '⏱ ETA',  'value': hrs_left, 'fmt': '.1f', 'color': 'GREEN'},
                # {'label': 'BCE', 'value': self.monitor['bce'], 'fmt': '.4f', 'color': 'MAGENTA'},
                {'label': 'LANE', 'value': self.monitor['lane_loss'], 'fmt': '.4f', 'color': 'YELLOW'},
                {'label': 'LANE_O2M', 'value': self.monitor['lane_loss_one2many'], 'fmt': '.4f', 'color': 'YELLOW'},
                {'label': 'DEPTH', 'value': self.monitor['depth_loss'], 'fmt': '.4f', 'color': 'YELLOW'},
                {'label': 'PVSEG', 'value': self.monitor['pvseg_loss'], 'fmt': '.4f', 'color': 'YELLOW'},
                {'label': 'BEVSEG', 'value': self.monitor['bevseg_loss'], 'fmt': '.4f', 'color': 'YELLOW'},
                # {'label': 'CLS', 'value': self.monitor['loss_cls'], 'fmt': '.4f', 'color': 'GREEN'},
                # {'label': 'PTS', 'value': self.monitor['loss_pts'], 'fmt': '.4f', 'color': 'GREEN'},
                # {'label': 'DIR', 'value': self.monitor['loss_dir'], 'fmt': '.4f', 'color': 'GREEN'},
                # {'label': 'MASK', 'value': self.monitor['loss_mask'], 'fmt': '.4f', 'color': 'GREEN'},
                {'label': 'LR',     'value': self.opt.param_groups[1]['lr'], 'fmt': '.2e', 'color': 'CYAN'},

                # Add more metrics here easily:
                # {'label': 'IoU',  'value': self.monitor.get('iou', 0), 'fmt': '.4f', 'color': 'GREEN'},
            ]
            # ====================================================
            
            colored_metrics, plain_metrics = format_metrics(metrics)
            
            # Colored terminal output
            epoch_str = f"{C['BOLD']}Epoch {C['YELLOW']}{e:03d}{C['RESET']}{C['DIM']}/{self.args.num_epochs}{C['RESET']}"
            print(f"{C['CYAN']}━━━{C['RESET']} {epoch_str} {C['DIM']}│{C['RESET']} {colored_metrics} {C['CYAN']}━━━{C['RESET']}")
            
            # Plain text for log file
            self.log.info(f"━━━ Epoch {e:03d}/{self.args.num_epochs} │ {plain_metrics} ━━━")

    def print_training_progress(self, e, b, elapsed):
        if self.rank == 0:
            C = ANSI_COLORS
            if b >= self.num_batches - 2:
                sys.stdout.write('\r' + ' ' * 120 + '\r')
            else:
                pbar = progress_bar(b, self.num_batches)
                
                # ========== EASILY ADD/MODIFY METRICS HERE ==========
                # bce_ = self.monitor['bce'] / self.monitor['iter']
                # vlm_loss_ = self.monitor['vlm_loss'] / self.monitor['iter']
                lane_loss_ = self.monitor['lane_loss'] / self.monitor['iter']
                pvseg_loss_ = self.monitor['pvseg_loss'] / self.monitor['iter']
                bevseg_loss_ = self.monitor['bevseg_loss'] / self.monitor['iter']
                lane_loss_one2many_ = self.monitor['lane_loss_one2many'] / self.monitor['iter']
                # loss_cls_ = self.monitor['loss_cls'] / self.monitor['iter']
                # loss_pts_ = self.monitor['loss_pts'] / self.monitor['iter']
                # loss_dir_ = self.monitor['loss_dir'] / self.monitor['iter']
                # loss_mask_ = self.monitor['loss_mask'] / self.monitor['iter']
                depth_loss_ = self.monitor['depth_loss'] / self.monitor['iter']
                metrics = [
                    {'label': '⏱', 'value': elapsed, 'fmt': '.3f', 'color': 'DIM'},
                    {'label': 'Total Loss', 'value': self.monitor['total_loss'], 'fmt': '.4f', 'color': 'MAGENTA'},
                    # {'label': 'BCE', 'value': bce_, 'fmt': '.4f', 'color': 'YELLOW'},
					# {'label': 'VLM_LOSS', 'value': vlm_loss_, 'fmt': '.4f', 'color': 'MAGENTA'},
                    {'label': 'LANE', 'value': lane_loss_, 'fmt': '.4f', 'color': 'CYAN'},
                    {'label': 'LANE_O2M', 'value': lane_loss_one2many_, 'fmt': '.4f', 'color': 'CYAN'},
                    {'label': 'DEPTH', 'value': depth_loss_, 'fmt': '.4f', 'color': 'YELLOW'},
                    {'label': 'PVSEG', 'value': pvseg_loss_, 'fmt': '.4f', 'color': 'YELLOW'},
                    {'label': 'BEVSEG', 'value': bevseg_loss_, 'fmt': '.4f', 'color': 'YELLOW'},
                    # {'label': 'CLS', 'value': loss_cls_, 'fmt': '.4f', 'color': 'GREEN'},
                    # {'label': 'PTS', 'value': loss_pts_, 'fmt': '.4f', 'color': 'GREEN'},
                    # {'label': 'DIR', 'value': loss_dir_, 'fmt': '.4f', 'color': 'GREEN'},
                    # {'label': 'MASK', 'value': loss_mask_, 'fmt': '.4f', 'color': 'GREEN'},
                    # Add more metrics here easily:
                    # {'label': 'Grad', 'value': grad_norm, 'fmt': '.2f', 'color': 'YELLOW'},
                ]
                # ====================================================
                
                metric_str = format_metrics_inline(metrics)
                
                line = (
                    f"\r {C['BOLD']}🚀 Train{C['RESET']} "
                    f"{C['DIM']}E{C['RESET']}{C['YELLOW']}{e:03d}{C['RESET']} "
                    f"{pbar} "
                    f"{C['DIM']}({b+1}/{self.num_batches}){C['RESET']} "
                    f"{metric_str}"
                )
                sys.stdout.write(line)
            sys.stdout.flush()

    def print_validation_progress(self, b, num_batches, **extra_metrics):
        """
        Print validation progress. Pass extra metrics as kwargs:
            print_validation_progress(b, num_batches, IoU=0.45, Loss=0.12)
        """
        if self.rank == 0:
            C = ANSI_COLORS
            if b >= num_batches - 2:
                sys.stdout.write('\r' + ' ' * 120 + '\r')
            else:
                pbar = progress_bar(b, num_batches)
                
                # ========== EASILY ADD/MODIFY METRICS HERE ==========
                metrics = [
                    # Add default metrics here if needed
                ]
                # Add any extra metrics passed as kwargs
                for label, value in extra_metrics.items():
                    metrics.append({'label': label, 'value': value, 'fmt': '.4f', 'color': 'GREEN'})
                # ====================================================
                
                metric_str = format_metrics_inline(metrics) if metrics else ""
                extra_str = f" {metric_str}" if metric_str else ""
                
                line = (
                    f"\r {C['BOLD']}🔍 Valid{C['RESET']} "
                    f"{pbar} "
                    f"{C['DIM']}({b+1}/{num_batches}){C['RESET']}"
                    f"{extra_str}"
                )
                sys.stdout.write(line)
            sys.stdout.flush()



    # --- Training  ---
    def return_label(self, label, label_indices):
        label = torch.cat([label[:, idx].max(dim=1, keepdim=True).values for idx in label_indices], dim=1)
        return label
    
    def reform_batch(self, batch, target_index, isTrain=True):
        '''Select target frame from sequential batch data.'''

        output = {}
        for k, v in batch.items():
            # Do not consider input_ids and vlm_labels for target selection
            if (k == 'input_ids' or k == 'vlm_labels' or k == 'polylines'):
                continue
            
            # Select target frame
            if (isTrain):
                output[k] = v[:, target_index]
            else:
                output[k] = v[target_index].unsqueeze(0)
        
        return output

    def train(self, batch):
        """
        Single training step with optional mixed-precision (AMP) support.
        Controlled by args.bool_mixed_precision (0/1).
        """

        self.opt.zero_grad()

        # PyTorch decides which ops are safe and beneficial in lower precision.
        with autocast(dtype=torch.float16, enabled=self.use_fp16):
            # prediction
            pred = self.model(batch, self.dtype, isTrain=True)

            # Loss calculation
            total_loss = torch.zeros(1).type(self.dtype).cuda()

            # Lane loss
            lane_loss = pred['lane_loss'] if 'lane_loss' in pred else torch.zeros(1).type(self.dtype).cuda()
            total_loss += lane_loss

            # Lane loss one2many
            lane_loss_one2many = pred['lane_loss_one2many'] if 'lane_loss_one2many' in pred else torch.zeros(1).type(self.dtype).cuda()
            total_loss += lane_loss_one2many            

            # Depth loss
            depth_loss = pred['depth_loss'] if 'depth_loss' in pred else torch.zeros(1).type(self.dtype).cuda()
            total_loss += depth_loss            

            # PV seg loss
            pvseg_loss = pred['pvseg_loss'] if 'pvseg_loss' in pred else torch.zeros(1).type(self.dtype).cuda()
            total_loss += pvseg_loss

            # BEV seg loss
            bevseg_loss = pred['bevseg_loss'] if 'bevseg_loss' in pred else torch.zeros(1).type(self.dtype).cuda()
            total_loss += bevseg_loss

            

        # Backpropagation
        if total_loss.requires_grad:
            if self.use_fp16:
                # AMP: scale loss, backward, unscale, optional grad clip, step, update scale
                self.scaler.scale(total_loss).backward()

                if self.args.grad_clip > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                # Standard FP32 training
                total_loss.backward()
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.opt.step()

        self.monitor.update({'total_loss': total_loss.item(), 'iter': self.monitor['iter'] + 1})
        self.monitor['lane_loss'] += lane_loss.item()
        self.monitor['lane_loss_one2many'] += lane_loss_one2many.item()
        self.monitor['pvseg_loss'] += pvseg_loss.item()
        self.monitor['bevseg_loss'] += bevseg_loss.item()
        self.monitor['depth_loss'] += depth_loss.item()
        self.loss_tracker.update(pred, self.monitor['iter'], self.opt.param_groups[1]['lr'])
    
    
    
    # --- Evaluation ---
    def eval(self, dataset, dataloader, _sampler, e):

        if self.args.eval_mode == 'lane':
            dataset_size = len(dataset) if dataset is not None else None
            return self.eval_lane(dataloader, e, dataset_size=dataset_size)
        else:
            raise ValueError(f"Invalid mode: {self.args.eval_mode}")


    def eval_lane(self, dataloader, e, dataset_size=None):
        num_batches = len(dataloader)

        polyline_metric = PolylinemAPMetric(
            class_names=[layer for layer in self.cfg['nuscenes']['OnlineHDmap']['target_classes']],
            chamfer_thresholds=[0.5, 1.0, 1.5],
            has_background_class=False,
        )

        # ── Padding-aware shard size ──────────────────────────────────────────
        # DistributedSampler (drop_last=False) pads the dataset so every rank
        # processes the same number of batches.  Padded samples are duplicates
        # of the first samples in the dataset, appended at the tail of the
        # higher-rank shards:
        #   rank r gets 1 padded sample  if  r >= world_size - num_padded
        # We track how many *real* samples belong to this rank and stop
        # accumulating into the metric once that count is reached.
        real_samples_this_rank = None
        if self.args.ddp and self.world_size > 1 and dataset_size is not None:
            num_padded       = (self.world_size - dataset_size % self.world_size) % self.world_size
            samples_per_rank = math.ceil(dataset_size / self.world_size)
            rank_padded      = 1 if (num_padded > 0 and self.rank >= self.world_size - num_padded) else 0
            real_samples_this_rank = samples_per_rank - rank_padded

        self.mode_selection(isTrain=False)
        processed = 0
        with torch.no_grad():
            for b, batch in enumerate(dataloader):
                pred = self.model(batch, self.dtype, isTrain=False)
                pred = pred['outs_one2one']


                pred_polylines = denormalize_2d_pts(pred['all_pts_preds'][-1], self.cfg['nuscenes']['pc_range'])  # bs num_queries num_pts 2
                pred_scores    = pred['all_cls_scores'][-1]   # bs num_queries num_classes
                gt_polylines   = batch['polylines']            # list[dict], one per batch item

                # Trim batch so padded duplicate samples at the shard tail are excluded
                if real_samples_this_rank is not None:
                    bs    = pred_polylines.shape[0]
                    valid = min(bs, max(0, real_samples_this_rank - processed))
                    if valid <= 0:
                        break
                    if valid < bs:
                        pred_polylines = pred_polylines[:valid]
                        pred_scores    = pred_scores[:valid]
                        gt_polylines   = list(gt_polylines)[:valid]
                    processed += valid

                polyline_metric.update(pred_polylines, pred_scores, gt_polylines)
                self.print_validation_progress(b, num_batches)

        # ── DDP gather: collect per-rank metric state on rank 0 ──────────────
        # Each rank's state contains compact numpy arrays (TP/FP flags and scores),
        # so gather_object is efficient.  The sort-by-score step inside compute()
        # is order-agnostic, so concatenation order across ranks does not matter.
        if self.args.ddp and self.world_size > 1:
            local_state = {
                'scores':  polyline_metric._pred_scores_by_cls,
                'tp':      polyline_metric._tp_by_cls_thr,
                'fp':      polyline_metric._fp_by_cls_thr,
                'num_gts': polyline_metric._num_gts_by_cls,
            }
            gathered_states = [None] * self.world_size
            dist.gather_object(local_state, gathered_states if self.rank == 0 else None, dst=0)

            if self.rank == 0:
                for r in range(1, self.world_size):
                    state = gathered_states[r]
                    for cls_id in range(polyline_metric.num_classes):
                        polyline_metric._pred_scores_by_cls[cls_id].extend(state['scores'][cls_id])
                        polyline_metric._num_gts_by_cls[cls_id] += state['num_gts'][cls_id]
                        for thr in polyline_metric.chamfer_thresholds:
                            polyline_metric._tp_by_cls_thr[cls_id][thr].extend(state['tp'][cls_id][thr])
                            polyline_metric._fp_by_cls_thr[cls_id][thr].extend(state['fp'][cls_id][thr])

        # ── Compute and log (rank 0 only) ─────────────────────────────────────
        if self.rank == 0:
            map_results = polyline_metric.compute()
            mAP_mean    = map_results['mAP']

            self.log.info(f">> Epoch {e:03d} - Lane mAP Metrics:")
            for key, value in map_results.items():
                if key == 'mAP':
                    self.log.info(f"   mAP: {value:.4f}")
                else:
                    self.log.info(f"   AP/{key.split('/')[-1]}: {value:.4f}")
            self.log.info(f"   Previous Best: {self.monitor['best_mAP']:.4f}")

            if self.monitor['best_mAP'] < mAP_mean:
                self.monitor['best_mAP'] = mAP_mean
                self.log.info(f">> ✓ New best mAP! Saving checkpoint...")
                self.save_trained_network_params(e)
            else:
                self.log.info(f">> Current mAP not better than previous best. Skipping save.")


def return_print_dict():
        
    return {
        # Items without section (printed first)
        'exp_id': {
            'Label': 'Experiment ID',
            'Color': GREEN,
            'use_color': True
        },
        'gpu_num': {
            'Label': 'GPU Number',
            'Color': GREEN,
            'use_color': True
        },
        'num_epochs': {
            'Label': 'Epochs',
            'Color': GREEN,
            'use_color': True
        },
        'batch_size': {
            'Label': 'Batch Size',
            'Color': GREEN,
            'use_color': True
        },
        
        # First section: "Basic Configuration" (all items with this section grouped together)
        'past_horizon_seconds': {
            'Label': 'Past Horizon',
            'Color': GREEN,
            'use_color': True,
            'format': lambda x: f"{x:.1f} sec",
            'section': 'Basic Configuration'
        },
        'future_horizon_seconds': {
            'Label': 'Future Horizon',
            'Color': GREEN,
            'use_color': True,
            'format': lambda x: f"{x:.1f} sec",
            'section': 'Basic Configuration'
        },
        'target_sample_period': {
            'Label': 'Sample Period',
            'Color': GREEN,
            'use_color': True,
            'format': lambda x: f"{x:.1f} Hz",
            'section': 'Basic Configuration'
        },
        
        # Second section: "Optimizer Settings" (all items with this section grouped together)
        'optimizer_type': {
            'Label': 'Optimizer',
            'Color': MAGENTA,
            'use_color': True,
            'section': 'Optimizer Settings'
        },
        'learning_rate': {
            'Label': 'Learning Rate',
            'Color': GREEN,
            'use_color': True,
            'format': '.5f',
            'section': 'Optimizer Settings'
        },
        'weight_decay': {
            'Label': 'Weight Decay',
            'Color': GREEN,
            'use_color': True,
            'format': '.8f',
            'section': 'Optimizer Settings'
        },
        'apply_lr_scheduling': {
            'Label': 'LR Scheduling',
            'Color': YELLOW,
            'use_color': True,
            'format': lambda x: 'Enabled' if x else 'Disabled',
            'section': 'Optimizer Settings'
        },
        'lr_schd_type': {
            'Label': 'LR Schedule Type',
            'Color': YELLOW,
            'use_color': True,
            'section': 'Optimizer Settings'
        },
        'fp16_loss_scale': {
            'Label': 'FP16 Loss Scale',
            'Color': GREEN,
            'use_color': True,
            'format': lambda x: f"{x:.1f}",
            'section': 'Optimizer Settings'
        },

        
        # Third section: "BEV Settings" (demonstrates multiple sections)
        'bool_apply_img_aug': {
            'Label': 'Image Augmentation',
            'Color': YELLOW,
            'use_color': True,
            'format': lambda x: 'Enabled' if x else 'Disabled',
            'section': 'BEV Settings'
        },
        'bool_apply_bev_aug': {
            'Label': 'BEV Augmentation',
            'Color': YELLOW,
            'use_color': True,
            'format': lambda x: 'Enabled' if x else 'Disabled',
            'section': 'BEV Settings'
        },
        'bool_apply_img_aug_photo': {
            'Label': 'Image Augmentation (Photo)',
            'Color': YELLOW,
            'use_color': True,
            'format': lambda x: 'Enabled' if x else 'Disabled',
            'section': 'BEV Settings'
        },
        'bool_one2many': {
            'Label': 'One2Many',
            'Color': YELLOW,
            'use_color': True,
            'format': lambda x: 'Enabled' if x else 'Disabled',
            'section': 'BEV Settings'
        },
        # Fifth section: "Parameter Settings" (demonstrates multiple sections)
        # 'w_alpha': {
        #     'Label': 'Alpha',
        #     'Color': GREEN,
        #     'use_color': True,
        #     'section': 'Param Settings'
        # },
        # 'w_beta': {
        #     'Label': 'Beta',
        #     'Color': GREEN,
        #     'use_color': True,
        #     'section': 'Param Settings'
        # },
        # 'w_gamma': {
        #     'Label': 'Gamma',
        #     'Color': GREEN,
        #     'use_color': True,
        #     'section': 'Param Settings'
        # },
        'eval_mode': {
            'Label': 'Evaluation Mode (bev or text or lane)',
            'Color': YELLOW,
            'use_color': True,
            'section': 'Param Settings'
        },

    }