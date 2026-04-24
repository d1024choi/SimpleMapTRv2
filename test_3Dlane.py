"""BEV Segmentation Testing Script."""

import os
import sys
import pickle
import logging
import argparse
import traceback

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.functions import get_dtypes, read_all_saved_param_idx, ANSI_COLORS, toNP
from utils.metrics import IoUMetric, PolylinemAPMetric
from helper import load_datasetloader, load_solvers
from dataset.NuscenesDataset.visualization import PolylineVisualizer
from models.common import denormalize_2d_pts



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BEV Segmentation Testing')
    
    # Basic settings
    parser.add_argument('--exp_id', type=int, default=71, help='Experiment ID')
    parser.add_argument('--gpu_num', type=int, default=0, help='GPU number')
    parser.add_argument('--dataset_type', type=str, default='nuscenes')
    parser.add_argument('--model_name', type=str, default='Scratch')
    
    # Test settings
    parser.add_argument('--is_test_all', type=int, default=0, help='Test all checkpoints')
    parser.add_argument('--model_num', type=int, default=24, help='Specific model number to test')
    parser.add_argument('--visualization', type=int, default=0, help='Enable visualization')
    
    return parser.parse_args()


def setup_logging(save_dir):
    """Setup logging to file and console."""
    logging.basicConfig(
        filename=os.path.join(save_dir, 'test.log'),
        filemode='w',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    return logger


def test(args, logger):
    """Run BEV segmentation testing."""
    C = ANSI_COLORS
    
    # CUDA setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    _, float_dtype = get_dtypes(useGPU=True)

    # Load saved configuration
    save_dir = f'./saved_models/{args.dataset_type}_{args.model_name}_model{args.exp_id}'
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    config_dict_path = os.path.join(save_dir, 'config_dict.pkl')
    if os.path.exists(config_dict_path):
        with open(config_dict_path, 'rb') as f:
            saved_cfg = pickle.load(f)

    # Override settings for testing
    saved_args.batch_size = 1
    saved_args.ddp = 0
    saved_args.save_dir = save_dir
    saved_args.exp_id = args.exp_id
    
    if (bool(args.visualization)):
        assert saved_args.batch_size == 1, "Batch size must be 1 for visualization"

    # Initialize polyline visualizer (once per checkpoint).
    # Matches __TEST__.py: pc_range-based BEV, fixed_num_pts GT, per-class
    # matplotlib colors (divider=orange, ped_crossing=blue, boundary=green).
    class_names = [c for c in saved_cfg['nuscenes']['OnlineHDmap']['target_classes']]
    polyline_vis = PolylineVisualizer(
        cfg=saved_cfg,
        score_threshold=0.3,
        gt_thickness=1,
        pred_thickness=1,
        use_bg_class=False,
        pc_range=saved_cfg.get('nuscenes', {}).get('pc_range'),
        class_names=class_names,
        class_colors_plt=['orange', 'b', 'g'],   # same as __TEST__.py colors_plt
        car_icon_path='./figs/lidar_car.png',
        dpi=150,
    )

    
    polyline_metric = PolylinemAPMetric(
        class_names=[layer for layer in saved_cfg['nuscenes']['OnlineHDmap']['target_classes']],
        chamfer_thresholds=[0.5, 1.0, 1.5],
        has_background_class=False,
    )

    # Load data and model
    dataset, data_loader, _ = load_datasetloader(args=saved_args, dtype=torch.FloatTensor, world_size=1, rank=0, mode='test')
    solver = load_solvers(saved_args, dataset.num_scenes, world_size=1, rank=0, logger=logger, dtype=float_dtype, isTrain=False)


    # Determine which checkpoints to test
    ckp_idx_list = read_all_saved_param_idx(solver.save_dir)
    target_models = ckp_idx_list if args.is_test_all else [args.model_num]


    # Test each checkpoint
    for ckp_id in ckp_idx_list:
        if ckp_id not in target_models:
            logger.info(f'{C["DIM"]}[SKIP] Model {ckp_id} not in target list{C["RESET"]}')
            continue

        # Save visualization images
        vis_save_dir = os.path.join(save_dir, 'visualizations', f'ckp_{ckp_id}')
        os.makedirs(vis_save_dir, exist_ok=True)


        # Create metrics and load model
        solver.load_pretrained_network_params(ckp_id)
        solver.mode_selection(isTrain=False)
        polyline_metric.reset()

        # Run inference
        for b_idx, batch in enumerate(tqdm(data_loader, desc=f'Test (ckp {ckp_id})')):

            # inference
            with torch.no_grad():
                pred = solver.model(batch, float_dtype, isTrain=False)
                pred = pred['outs_one2one']
                
            pred_polylines = denormalize_2d_pts(pred['all_pts_preds'][-1], saved_cfg['nuscenes']['pc_range']) # bs num_lane_queries num_points 2
            pred_scores = pred['all_cls_scores'][-1] # bs num_lane_queries num_classes
            gt_polylines = batch['polylines']  # list of dicts, one per batch item

            # Update polyline mAP metric
            polyline_metric.update(pred_polylines, pred_scores, gt_polylines)



            # Visualize polylines + surround-view cameras (__TEST__.py style)
            if bool(args.visualization):
                # batch['images']: (B, seq, n_cam, C, H, W) — take the last timestep
                cam_images = batch['images'][:, -1] if 'images' in batch else None
                vis_images = polyline_vis.visualize(
                    pred_polylines=pred_polylines,
                    pred_scores=pred_scores,
                    gt_polylines=gt_polylines,
                    images=cam_images,
                )
                for img_idx, vis_img in enumerate(vis_images):
                    save_path = os.path.join(vis_save_dir, f'batch{b_idx:04d}_sample{img_idx:02d}.png')
                    cv2.imwrite(save_path, vis_img)

        # Compute polyline mAP results
        map_results = polyline_metric.compute()
        result_lines = [
            f"Polyline mAP: {map_results['mAP']:.4f}",
            *(f"  AP/{k.split('/')[-1]}: {v:.4f}" for k, v in map_results.items() if k != 'mAP'),
        ]

        # Log to test.log and print to terminal (logger has file + StreamHandler)
        logger.info(f"─── Results for Checkpoint {ckp_id} ───")
        logger.info("3D Lane / Polyline mAP:")
        for line in result_lines:
            logger.info(line)

        # Print styled summary to terminal
        print(f"\n{C['CYAN']}{'─' * 50}{C['RESET']}")
        print(f"{C['BOLD']}📊 Results for Checkpoint {C['YELLOW']}{ckp_id}{C['RESET']}")
        print(f"{C['CYAN']}{'─' * 50}{C['RESET']}")
        print(f"{C['BOLD']}3D Lane / Polyline mAP:{C['RESET']}")
        for line in result_lines:
            print(f"  {line}")


def main():
    args = parse_args()
    save_dir = f'./saved_models/{args.dataset_type}_{args.model_name}_model{args.exp_id}'
    logger = setup_logging(save_dir)
    
    try:
        test(args, logger)
    except Exception:
        error_msg = traceback.format_exc()
        logger.error(error_msg)  # Logs to both console and file
        print(f"\n{ANSI_COLORS['RED']}{'─' * 50}")
        print("❌ ERROR OCCURRED")
        print(f"{'─' * 50}{ANSI_COLORS['RESET']}")
        print(error_msg)


if __name__ == '__main__':
    main()
