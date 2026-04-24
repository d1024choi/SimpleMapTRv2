import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--app_mode', type=str, default='OnlineHDmap', help='BEV')

# ------------------------
# Exp Info
# ------------------------
parser.add_argument('--model_name', type=str, default='Scratch')
parser.add_argument('--exp_id', type=int, default=300)
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--load_pretrained', type=int, default=0)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--ddp', type=int, default=0)
parser.add_argument('--bool_mixed_precision', type=int, default=1)
parser.add_argument('--fp16_loss_scale', type=float, default=512.0,
                    help='Initial loss scale for FP16/mixed precision (MapTR uses 512). Prevents gradient underflow.')
parser.add_argument('--num_cores', type=int, default=1)

# ------------------------
# Dataset
# ------------------------
parser.add_argument('--dataset_dir', type=str, default='')
parser.add_argument('--label_dir', type=str, default='')
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--dataset_type', type=str, default='nuscenes')
parser.add_argument('--val_ratio', type=float, default=0.0, help='deprecated')
parser.add_argument('--past_horizon_seconds', type=float, default=0.5)
parser.add_argument('--future_horizon_seconds', type=float, default=0.0)
parser.add_argument('--target_sample_period', type=float, default=2)  # Hz ---


# ------------------------
# Training Env
# ------------------------
parser.add_argument('--num_epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--valid_step', type=int, default=1)
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--max_num_chkpts', type=int, default=3)


parser.add_argument('--optimizer_type', type=str, default='adamw', help='support adan and adamw only')
parser.add_argument('--learning_rate', type=float, default=0.0006)
parser.add_argument('--weight_decay', type=float, default=0.01, help='default for adam and adamW is 0 and 1e-2')
parser.add_argument('--grad_clip', type=float, default=35.0)
parser.add_argument('--apply_lr_scheduling', type=int, default=1)
parser.add_argument('--lr_schd_type', type=str, default='CosineAnnealing', help='StepLR, ExponentialLR, OnecycleLR, CosineAnnealing' )
parser.add_argument('--bool_find_unused_params', type=int, default=0)

parser.add_argument('--w_alpha', type=float, default=1.0)
parser.add_argument('--w_beta', type=float, default=0.0)
parser.add_argument('--w_gamma', type=float, default=1.0)



# ------------------------
# Network related
# ------------------------

parser.add_argument('--bool_apply_img_aug_photo', type=int, default=1)
parser.add_argument('--bool_temporal_model', type=int, default=0)
parser.add_argument('--bool_apply_img_aug', type=int, default=0)
parser.add_argument('--bool_apply_bev_aug', type=int, default=0)
parser.add_argument('--eval_mode', type=str, default='lane', help='bevmap, text, lane')

parser.add_argument('--bool_depth_aux', type=int, default=1)
parser.add_argument('--bool_pvseg_aux', type=int, default=1)
parser.add_argument('--bool_bevseg_aux', type=int, default=1)

parser.add_argument('--bool_one2many', type=int, default=1)

# parser.add_argument('--debug_position_embedding_type', type=str, default='learned', help='sine, learned')
# parser.add_argument('--debug_temp_attn_type', type=str, default='ori', help='ori, my')



args = parser.parse_args()
