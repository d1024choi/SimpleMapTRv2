"""NuScenes Dataset Loader for BEV Segmentation and Trajectory Prediction."""

import copy
import glob
import os
from os import path as osp
import sys
import random
from pathlib import Path
import json
import cv2
import numpy as np
from sympy.sets.sets import true
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pyquaternion import Quaternion
from shapely.geometry import MultiPolygon, Polygon
import matplotlib.pyplot as plt
from shapely.geometry import LineString, LinearRing, MultiLineString, box, Polygon
import shapely.ops as ops
from shapely.errors import GEOSException
from shapely import affinity

from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.utils.data_classes import LidarPointCloud, Box

from utils.functions import read_config, config_update, check_dataset_path_existence
from utils.geometry import (
    calculate_birds_eye_view_parameters,
    update_intrinsics, resize_and_crop_image, get_random_ref_matrix, quaternion_yaw, zero_padding_image
)

from scipy.spatial.transform import Rotation as R
from utils.augmentation import ImageAugmentation, PhotoMetricDistortionMultiViewImage
from dataset.NuscenesDataset.common import (
    CAMERAS, MAP_NAMES, STATIC, DYNAMIC, DIVIDER, INTERPOLATION, CLASSES, SCENE_BLACKLIST, get_pose, perspective
)

import nuscenes as nuscenes_module
import nuscenes.utils.data_classes as dc
from dataset.NuscenesDataset.map import LiDARInstanceLines

import warnings
warnings.filterwarnings(action='ignore')

class DatasetLoader(Dataset):

    def __init__(self, args, dtype, world_size=None, rank=None, mode='train', nusc=None):
        random.seed(1024)

        # Define mode and split
        split = 'train' if mode in ['train', 'val', 'valid'] else 'val'

        # Configuration
        self.mode, self.dtype, self.rank, self.args = mode, dtype, rank, args
        self.cfg = config_update(read_config(), args)
        self.obs_len, self.pred_len = self.cfg['obs_len'], self.cfg['pred_len']
        self.seq_len = self.obs_len + self.pred_len
        self.nusc = nusc
        self.cfg_model = self.cfg[self.cfg['model_name']]

        # Image preprocessing parameters
        cfg_img = self.cfg['nuscenes']['image']
        ori_dims = (cfg_img['original_size']['w'], cfg_img['original_size']['h'])
        resize_dims = (cfg_img['target_size']['w'], cfg_img['target_size']['h'] + cfg_img['target_size']['top_crop'])
        crop = (0, cfg_img['target_size']['top_crop'], resize_dims[0], resize_dims[1])
        self.img_prepro_params = {
            'scale_width': resize_dims[0] / ori_dims[0],
            'scale_height': resize_dims[1] / ori_dims[1],
            'resize_dims': resize_dims,
            'crop': crop
        }
        self.img_to_tensor = transforms.Compose([transforms.ToTensor()])

        # Image augmentation
        # TODO : Add slight rotation 
        data_aug_conf = {
            'crop_offset': int(resize_dims[1] * 0.2),
            'resize_lim': [0.8, 1.2],
            'final_dim': (cfg_img['target_size']['h'], cfg_img['target_size']['w'])
        }
        self.img_aug = ImageAugmentation(data_aug_conf=data_aug_conf)
        self.img_aug_photo = PhotoMetricDistortionMultiViewImage()

        # Lidar data loader
        grid_config = self.cfg['nuscenes']['grid_config']
        from dataset.NuscenesDataset.data_preprocess import StandaloneDepthInputsPipeline
        self.lidar_preprocessor = StandaloneDepthInputsPipeline(grid_config, size_divisor=self.cfg['nuscenes']['image']['size_divisor'])


        # Map related parameters
        from dataset.NuscenesDataset.common import CLASS2LABEL  
        self.hdmap_label_config = self.cfg['nuscenes'][self.cfg['nuscenes']['map_data_type']]
        self.vec_classes = self.hdmap_label_config['target_classes']
        self.polygon_classes = ['road_segment', 'lane']        
        self.line_classes=['road_divider', 'lane_divider']
        self.ped_crossing_classes=['ped_crossing']
        self.contour_classes=['road_segment', 'lane']        
        self.CLASS2LABEL = CLASS2LABEL

        # updated according to the original implementation (x, y) -> (width, height)
        self.pc_range = self.cfg['nuscenes']['pc_range'] # [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        patch_w = self.pc_range[3]-self.pc_range[0] 
        patch_h = self.pc_range[4]-self.pc_range[1]        
        self.patch_size = (patch_h, patch_w) # approximately 60m x 30m       
        self.canvas_size = (self.cfg['nuscenes']['bev']['bev_h'], self.cfg['nuscenes']['bev']['bev_w'])
        self.scale_x = self.canvas_size[1] / self.patch_size[1]
        self.scale_y = self.canvas_size[0] / self.patch_size[0]


        polyline_dict = self.hdmap_label_config['polyline']
        self.sample_dist = polyline_dict.get('sample_dist', 1)
        self.num_samples = polyline_dict.get('num_samples', 250)
        self.padding = polyline_dict.get('padding', False)
        self.fixed_num = polyline_dict.get('fixed_ptsnum_per_line', 20)
        self.padding_value = polyline_dict.get('padding_value', -10000)

        # # BEV grid parameters
        # self.bev_resolution, self.bev_start_position, self.bev_dimension = \
        #     calculate_birds_eye_view_parameters(
        #         self.cfg['BEV']['lift']['x_bound'],
        #         self.cfg['BEV']['lift']['y_bound'],
        #         self.cfg['BEV']['lift']['z_bound'],
        #         isnumpy=True
        #     )

        # Load data
        self._load_nuscenes_data(world_size, split, mode)
        if (split == 'train'):
            self.sample_dir = self.hdmap_label_config['train_dir']
        else:
            self.sample_dir = self.hdmap_label_config['valid_dir']
        
            

        if rank == 0:
            print(f">> Dataset is loaded from {{{os.path.basename(__file__)}}} ")
            print(f'>> Number of available {mode} samples is {self.num_scenes}')

    def _load_nuscenes_data(self, world_size, split, mode):
        """Load data directly from NuScenes."""
        # Initialize NuScenes
        if (self.nusc is None):
            self.nusc = NuScenes(version=self.cfg['nuscenes']['version'], dataroot=self.cfg['nuscenes']['dataset_dir'], verbose=False)
        else:
            print(f">> nuScenes API is inherited from another source...")
        # self.nusc_map = {v: NuScenesMap(dataroot=self.cfg['nuscenes']['dataset_dir'], map_name=v) for v in MAP_NAMES}
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in MAP_NAMES:
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.cfg['nuscenes']['dataset_dir'], map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])        
        if self.rank == 0:
            print(f">> nuScenes is loaded...")
        

        # Get scene splits
        self.target_scenes = self._get_split(split) # Splits into train/valid/test
        self.sample_records = self._get_ordered_sample_records() # scene0-00, scene0-01, ..., scene1-00, scene1-01, ...
        seq_sample_indices = self._get_seq_sample_indices() # [[0, 1, 2, ...], [1, 2, 3, ...], [2, 3, 4, ...], ...]

        # Skip samples in blacklist
        if self.cfg['nuscenes']['bool_filter_samples_in_blacklist']:
            seq_sample_indices = self._filter_samples_in_blacklist(seq_sample_indices)

        # Split into train/val
        if mode in ['train', 'val', 'valid']:
            num_val = int(len(seq_sample_indices) * self.cfg['args']['val_ratio'])
            num_train = len(seq_sample_indices) - num_val
            train_scenes = seq_sample_indices[:num_train]
            val_scenes = seq_sample_indices[num_train:] * world_size
            self.scenes = train_scenes if mode == 'train' else val_scenes
        else:
            self.scenes = seq_sample_indices
        self.num_scenes = len(self.scenes)
        # self.num_scenes = 16

    def __len__(self):
        return self.num_scenes
        # return 64
        
    def __getitem__(self, idx):

        # consecutive record indices over the entire time horizon (obs_len + pred_len)
        seq_indices = np.array(self.scenes[idx])[self.cfg['target_frame_indices']] 

        # extract sequence data
        data = self._extract_seq_data(seq_indices)

        return data


    # ==================== Data Extraction ====================

    def _extract_seq_data(self, seq_indices):
        """Extract sequence data.
        
        Returns:
            images (1 x n x c x h x w, float32): Normalized to [0,1], optionally by mean/var
            intrinsics (1 x n x 3 x 3, float32): Crop and scaling applied
            extrinsics (1 x n x 4 x 4, float32): Ego-lidar to camera transform
            bev (1 x 12 x h x w, float32): BEV features (flipped upside-down and left-right)
            center (1 x 1 x h x w, float32): Center map
            visibility (1 x 1 x h x w, uint8): Visibility mask
            w2e, e2w (1 x 4 x 4, float32): World-to-ego and ego-to-world transforms
        
        Note:
            Apply np.flipud(np.fliplr(bev)) = torch.flip(bev, dims=(2, 3)) to match ego-centric frame (forward-up, side-left).
        
        example) 
        o       x                 -------------> y
                ^                 |         
                |                 |
                |                 | 
        y <-----------            v x          o

        (x, y) = (50m, 50m)    (x_p, y_p) = (200, 200)       
        
        """
               
        seq_images, seq_intrinsics, seq_cam2egos, seq_ego2globals, seq_lidar2cams, seq_cam2lidars, seq_lidar2egos, seq_gt_depths = [], [], [], [], [], [], [], []
        seq_can_bus = []
        seq_gt_pv_semantic_masks, seq_gt_semantic_masks = [], []
        seq_bev_aug_mat, seq_polylines, seq_bbox_anns = [], {}, {}
        sources, locations = [], []

        skip_img = False
        for i in seq_indices:
            
            # get log info.
            rec = self.sample_records[i]
            scene  = self.nusc.get('scene', rec['scene_token'])
            log    = self.nusc.get('log', scene['log_token'])
            location = log['location']   # e.g. "boston-seaport", "singapore-hollandvillage", ...
            city = "boston" if location.startswith("boston") else "singapore" if location.startswith("singapore") else location
            locations.append(city)
            
            sample_token = rec['token']
            rec_onlinehdmap = pickle.load(open(os.path.join(self.sample_dir, f'{sample_token}.pkl'), 'rb'))

            # ------------------------------------------------------------------
            # GT Polylines
            # polylines: num_instances num_rotations num_points 2
            # labels: num_instances
            bboxes, polylines, polylines_shift, labels, vectors = self._get_polyline_labels(rec, rec_onlinehdmap)

            # ------------------------------------------------------------------
            # Camera data
            if not skip_img:
                # seq x cam x ch x H x W
                images, intrinsics, cam2egos, ego2globals, lidar2cams, cam2lidars, lidar2egos, gt_depths, gt_pv_semantic_masks, gt_semantic_masks = \
                self._get_image_data(rec, rec_onlinehdmap, vectors)
                seq_images.append(images)
                seq_intrinsics.append(intrinsics)
                seq_cam2egos.append(cam2egos)
                seq_ego2globals.append(ego2globals)
                seq_lidar2cams.append(lidar2cams)
                seq_cam2lidars.append(cam2lidars)
                seq_lidar2egos.append(lidar2egos)
                seq_gt_depths.append(gt_depths)
                seq_gt_pv_semantic_masks.append(gt_pv_semantic_masks)
                seq_gt_semantic_masks.append(gt_semantic_masks)


            # ------------------------------------------------------------------
            # BEV augmentation matrix data
            bev_aug_mat = self._get_bev_aug()
           
          

            # ------------------------------------------------------------------
            # BBoxes
            bbox_anns = self._get_anns_by_category(rec, DYNAMIC)


            # Verification code ----------------------
            if not skip_img and self.cfg['nuscenes'].get('bool_verif_bbox_proj', False) and len(bbox_anns) > 0:

                _BOX_EDGES = [(0,1),(1,2),(2,3),(3,0),
                              (4,5),(5,6),(6,7),(7,4),
                              (0,4),(1,5),(2,6),(3,7)]

                # Build global→lidar transform from rec_onlinehdmap
                lidar2ego = np.eye(4, dtype=np.float32)
                lidar2ego[:3, :3] = Quaternion(rec_onlinehdmap['lidar2ego_rotation']).rotation_matrix
                lidar2ego[:3,  3] = rec_onlinehdmap['lidar2ego_translation']
                ego2global = np.eye(4, dtype=np.float32)
                ego2global[:3, :3] = Quaternion(rec_onlinehdmap['ego2global_rotation']).rotation_matrix
                ego2global[:3,  3] = rec_onlinehdmap['ego2global_translation']
                global2lidar = np.linalg.inv(ego2global @ lidar2ego)  # (4,4)

                # Transform GT boxes: global → lidar frame (x=forward, y=left, z=up)
                boxes_lidar_corners = []
                for box in bbox_anns:
                    b = copy.deepcopy(box)
                    b.translate(-ego2global[:3, 3])
                    b.rotate(Quaternion(matrix=ego2global[:3, :3]).inverse)
                    b.translate(-lidar2ego[:3, 3])
                    b.rotate(Quaternion(matrix=lidar2ego[:3, :3]).inverse)
                    boxes_lidar_corners.append(b.corners().T.astype(np.float32))  # (8, 3)

                # Distinct colors per box
                n_boxes = len(boxes_lidar_corners)
                tab20   = plt.cm.tab20(np.linspace(0, 1, max(n_boxes, 1)))
                box_colors_u8 = [(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in tab20]
                box_colors_f  = [tuple(c[:3]) for c in tab20]

                # Project boxes into each camera and annotate
                cam_order        = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                    'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT']
                cam_types_in_order = list(rec_onlinehdmap['cams'].keys())
                imgs_annotated   = {}

                for cam_idx, cam_type in enumerate(cam_types_in_order):
                    L2C     = lidar2cams[0, cam_idx].numpy()                    # (4,4)
                    K       = intrinsics[0, cam_idx].numpy()                    # (3,3)
                    img_vis = images[0, cam_idx].permute(1, 2, 0).numpy().astype(np.uint8).copy()
                    H_img, W_img = img_vis.shape[:2]

                    for bi, corners in enumerate(boxes_lidar_corners):
                        bcolor      = box_colors_u8[bi % len(box_colors_u8)]
                        corners_h   = np.hstack([corners,
                                                  np.ones((8, 1), dtype=np.float32)]).T   # (4,8)
                        corners_cam = L2C @ corners_h                                      # (4,8)
                        in_front_c  = corners_cam[2] > 0.1
                        z_safe      = np.where(in_front_c, corners_cam[2], 1.0)
                        proj_all    = K @ corners_cam[:3]                                  # (3,8)
                        u_c = (proj_all[0] / z_safe).astype(int)
                        v_c = (proj_all[1] / z_safe).astype(int)

                        for e0, e1 in _BOX_EDGES:
                            if not (in_front_c[e0] and in_front_c[e1]):
                                continue
                            ret, p0, p1 = cv2.clipLine(
                                (0, 0, W_img, H_img),
                                (u_c[e0], v_c[e0]), (u_c[e1], v_c[e1]))
                            if ret:
                                cv2.line(img_vis, p0, p1, bcolor, 2, cv2.LINE_AA)

                        for ci_c in range(8):
                            if in_front_c[ci_c] and 0 <= u_c[ci_c] < W_img and 0 <= v_c[ci_c] < H_img:
                                cv2.circle(img_vis, (u_c[ci_c], v_c[ci_c]), 4, bcolor, -1)

                    cv2.putText(img_vis, cam_type.replace('CAM_', ''), (8, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2, cv2.LINE_AA)
                    imgs_annotated[cam_type] = img_vis

                # Compose 2×3 camera grid + 3D scatter
                if len(imgs_annotated) == 6:
                    fallback  = np.zeros_like(next(iter(imgs_annotated.values())))
                    ordered   = [imgs_annotated.get(c, fallback) for c in cam_order]
                    H0, W0    = ordered[0].shape[:2]

                    fig = plt.figure(figsize=(28, 10))
                    gs  = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.4],
                                           wspace=0.04, hspace=0.04)

                    for row in range(2):
                        for col in range(3):
                            ax = fig.add_subplot(gs[row, col])
                            ax.imshow(cv2.resize(ordered[row * 3 + col], (W0, H0)))
                            ax.axis('off')

                    # 3D view: GT boxes in lidar frame + heading arrows
                    ax3d = fig.add_subplot(gs[:, 3], projection='3d')
                    for bi, corners in enumerate(boxes_lidar_corners):
                        fc = box_colors_f[bi % len(box_colors_f)]
                        for e0, e1 in _BOX_EDGES:
                            ax3d.plot([corners[e0, 0], corners[e1, 0]],
                                      [corners[e0, 1], corners[e1, 1]],
                                      [corners[e0, 2], corners[e1, 2]],
                                      color=fc, linewidth=1.5)

                    ax3d.scatter([0], [0], [0], c='white', edgecolors='black', s=100, zorder=5)
                    _al = 5.0
                    for (dx, dy, dz), color, lbl in [
                            ((0, _al, 0),  'green', 'X forward'),
                            ((-_al, 0, 0), 'red',   'Y left'),
                            ((0, 0, _al),  'blue',  'Z up'),
                    ]:
                        ax3d.quiver(0, 0, 0, dx, dy, dz,
                                    color=color, linewidth=2.5, arrow_length_ratio=0.2)
                        ax3d.text(dx*1.1, dy*1.1, dz*1.1, lbl,
                                  color=color, fontsize=8, fontweight='bold')

                    ax3d.set_xlabel('X  right (m)', fontsize=8)
                    ax3d.set_ylabel('Y  forward (m)', fontsize=8)
                    ax3d.set_zlabel('Z  up (m)', fontsize=8)
                    ax3d.set_title(f'GT 3D BBoxes in lidar frame  ({n_boxes} boxes)', fontsize=9)
                    ax3d.view_init(elev=35, azim=-60)
                    ax3d.set_box_aspect([1, 1, 0.3])

                    plt.savefig('./verif_bbox_projection.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f'[Verif] Saved GT bbox projection → verif_bbox_projection.png  ({n_boxes} boxes)')
            # Verification code ----------------------

            # ------------------------------------------------------------------
            # can bus
            can_bus = rec_onlinehdmap['can_bus'] if 'can_bus' in rec_onlinehdmap else None
            rotation = Quaternion(rec_onlinehdmap['ego2global_rotation'])
            translation = rec_onlinehdmap['ego2global_translation']
            can_bus[:3] = translation
            can_bus[3:7] = rotation
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            if patch_angle < 0:
                patch_angle += 360
            can_bus[-2] = patch_angle / 180 * np.pi
            can_bus[-1] = patch_angle


            # save in lists
            seq_polylines.update({i: {
                'bboxes': bboxes,
                'polylines': polylines,
                'polylines_shift': polylines_shift,
                'labels': labels
            }})
            seq_bbox_anns.update({i: bbox_anns})
            seq_bev_aug_mat.append(torch.from_numpy(bev_aug_mat).unsqueeze(0))
            seq_can_bus.append(torch.from_numpy(can_bus).unsqueeze(0)) # 1 x can_bus_len

		
        result = {
            'polylines': seq_polylines,
            'bev_aug_mat': torch.cat(seq_bev_aug_mat, dim=0),
            'bbox_anns': seq_bbox_anns
        }
        
        if not skip_img:
            result.update({
                'images': torch.cat(seq_images, dim=0),
                'intrinsics': torch.cat(seq_intrinsics, dim=0),
                'cam2egos': torch.cat(seq_cam2egos, dim=0),
                'ego2globals': torch.cat(seq_ego2globals, dim=0),
                'lidar2cams': torch.cat(seq_lidar2cams, dim=0),
                'cam2lidars': torch.cat(seq_cam2lidars, dim=0),
                'lidar2egos': torch.cat(seq_lidar2egos, dim=0),
                'can_bus': torch.cat(seq_can_bus, dim=0),
                'gt_depths': torch.cat(seq_gt_depths, dim=0),
                'gt_pv_semantic_masks': torch.cat(seq_gt_pv_semantic_masks, dim=0),
                'gt_semantic_masks': torch.cat(seq_gt_semantic_masks, dim=0),
            })

        return result


    # ==================== NuScenes Data Loading ====================

    def _get_image_data(self, sample_record, sample_onlinehdmap, vectors):
        """Get camera images, intrinsics, and extrinsics."""
        
        top_crop = self.img_prepro_params['crop'][1]
        left_crop = self.img_prepro_params['crop'][0]
        scale_width = self.img_prepro_params['scale_width']
        scale_height = self.img_prepro_params['scale_height']


        # camp_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']
        # cam_info is a dictionary of camera information
        images, intrinsics, cam2egos, ego2globals, lidar2cams, cam2lidars, lidar2egos = [], [], [], [], [], [], []
        gt_depths, gt_pv_semantic_masks, gt_semantic_masks = [], [], []

        # --------------------------------------------------
        # bev seg
        gt_semantic_mask = np.zeros((self.canvas_size[0], self.canvas_size[1]), dtype=np.uint8)
        if self.cfg_model['aux_tasks']['bev_seg']:
            for instance, instance_type in vectors:
                if instance_type != -1:
                    if instance.geom_type == 'LineString':
                        gt_semantic_mask = self.line_ego_to_mask(instance, gt_semantic_mask, color=1, thickness=3)
        gt_semantic_mask = torch.from_numpy(gt_semantic_mask).unsqueeze(0).unsqueeze(0)
        gt_semantic_masks.append(gt_semantic_mask)

        lidar2ego = np.eye(4).astype(np.float32) # lidar2ego
        lidar2ego[:3, :3] = Quaternion(sample_onlinehdmap['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = np.array(sample_onlinehdmap['lidar2ego_translation'])
        lidar2ego = torch.from_numpy(lidar2ego).unsqueeze(0).unsqueeze(0)
        lidar2egos.append(lidar2ego)


        for cam_type, cam_info in sample_onlinehdmap['cams'].items():

            # --------------------------
            # Image & Intrinsic
            img = Image.open(Path(self.nusc.get_sample_data_path(cam_info['sample_data_token'])))
            img = resize_and_crop_image(img, resize_dims=self.img_prepro_params['resize_dims'], crop=self.img_prepro_params['crop'])
            img_size = (img.size[0], img.size[1])

            # NOTE : we don't need to reflect this padding into intrinsic matrix because only bottom and right part are padded with zeros
            img = zero_padding_image(img, size_divisor=self.cfg['nuscenes']['image']['size_divisor'])
            
            # Augmentation (training only, phtometric)
            if self.mode == 'train' and self.cfg['nuscenes']['image']['bool_apply_img_aug_photo']:
                img = self.img_aug_photo({'img': [img]})['img'][0]

            intrinsic = torch.from_numpy(np.array(cam_info['cam_intrinsic'])) # cam2img
            intrinsic = update_intrinsics(intrinsic, top_crop, left_crop, scale_width=scale_width, scale_height=scale_height)

            # Augmentation (training only, resize and crop)
            if self.mode == 'train' and self.cfg['nuscenes']['image']['bool_apply_img_aug'] and np.random.rand() < 0.5:
                img, intrinsic = self.img_aug(img, intrinsic)

            # --------------------------
            # Extrinsic
            cam2ego = np.eye(4).astype(np.float32) # cam2camego
            cam2ego[:3, :3] = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
            cam2ego[:3, 3] = np.array(cam_info['sensor2ego_translation'])

            ego2global = np.eye(4).astype(np.float32) # camego2global
            ego2global[:3, :3] = Quaternion(cam_info['ego2global_rotation']).rotation_matrix
            ego2global[:3, 3] = np.array(cam_info['ego2global_translation'])

            cam2lidar = np.eye(4).astype(np.float32)
            cam2lidar[:3, :3] = cam_info['sensor2lidar_rotation']
            cam2lidar[:3, 3] = cam_info['sensor2lidar_translation']
            lidar2cam = np.linalg.inv(cam2lidar)

            # --------------------------------------------------
            # depth
            gt_depth = np.zeros((img_size[1], img_size[0]), dtype=np.float32)
            if self.cfg_model['aux_tasks']['depth']:
                input_dict = {'lidar_path': sample_onlinehdmap['lidar_path'],
                'camera_intrinsics': intrinsic,
                'lidar2cam': lidar2cam,
                'img_size': img_size,
                }
                gt_depth = self.lidar_preprocessor(input_dict)['gt_depth']
            gt_depth = torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(0)


            


            # --------------------------------------------------
            # pv seg
            feat_down_sample = self.cfg['Aux_tasks']['pvsegnet']['feat_down_sample']
            gt_pv_semantic_mask = np.zeros((img.size[1] // feat_down_sample, img.size[0] // feat_down_sample), dtype=np.uint8)
            if self.cfg_model['aux_tasks']['pv_seg']:
                intrin_homo = np.eye(4)
                intrin_homo[:3, :3] = intrinsic
                lidar2img = intrin_homo @ lidar2cam

                scale_factor = np.eye(4)
                scale_factor[0, 0] *= 1/feat_down_sample
                scale_factor[1, 1] *= 1/feat_down_sample
                lidar2feat = scale_factor @ lidar2img

                for instance, instance_type in vectors:
                    if instance_type != -1:
                        if instance.geom_type == 'LineString':
                            gt_pv_semantic_mask = self.line_ego_to_pvmask(instance, gt_pv_semantic_mask, lidar2feat, color=1, thickness=1)
            gt_pv_semantic_mask = torch.from_numpy(gt_pv_semantic_mask).unsqueeze(0).unsqueeze(0)


            # --------------------------------------------------
            # To Tensor
            img = torch.from_numpy(np.array(img)) # Need to be normalized!!!
            img = img.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            intrinsic = intrinsic.unsqueeze(0).unsqueeze(0)
            cam2ego = torch.from_numpy(cam2ego).unsqueeze(0).unsqueeze(0)
            ego2global = torch.from_numpy(ego2global).unsqueeze(0).unsqueeze(0)
            cam2lidar = torch.from_numpy(cam2lidar).unsqueeze(0).unsqueeze(0)
            lidar2cam = torch.from_numpy(lidar2cam).unsqueeze(0).unsqueeze(0)
            

            # Verification code -----------------------
            # if self.cfg['nuscenes'].get('bool_verif_lidar_proj', False):
            if False:
                if not hasattr(self, '_verif_buffer'):
                    self._verif_buffer = []

                if not hasattr(self, '_verif_boxes'):
                    _BOX_EDGES = [(0,1),(1,2),(2,3),(3,0),
                                  (4,5),(5,6),(6,7),(7,4),
                                  (0,4),(1,5),(2,6),(3,7)]

                    # ---- Diagnostic boxes at fixed axis positions ----
                    # Each box sits exactly on one axis so we can read off which
                    # axis is forward by checking which camera it appears in.
                    _DIAG = [
                        ( 12,  0, -0.5, '+X'),
                        (-12,  0, -0.5, '-X'),
                        (  0, 12, -0.5, '+Y'),
                        (  0,-12, -0.5, '-Y'),
                    ]
                    # Colors: red=+X, cyan=-X, green=+Y, magenta=-Y
                    _DIAG_COLORS_F  = [(1,.1,.1), (.1,.9,.9), (.1,.8,.1), (.9,.1,.9)]
                    _DIAG_COLORS_U8 = [(255,25,25), (25,230,230), (25,200,25), (230,25,230)]

                    box_corners_list = []
                    for cx, cy, cz, _ in _DIAG:
                        dx, dy, dz = 2.0, 1.0, 0.9   # fixed box size for all
                        local = np.array([
                            [-dx,-dy,-dz], [ dx,-dy,-dz],
                            [ dx, dy,-dz], [-dx, dy,-dz],
                            [-dx,-dy, dz], [ dx,-dy, dz],
                            [ dx, dy, dz], [-dx, dy, dz],
                        ], dtype=np.float32)
                        corners = local + np.array([cx, cy, cz], dtype=np.float32)
                        box_corners_list.append(corners)

                    self._verif_boxes          = box_corners_list
                    self._verif_box_edges      = _BOX_EDGES
                    self._verif_box_labels     = [d[3] for d in _DIAG]
                    self._verif_box_colors_u8  = _DIAG_COLORS_U8
                    self._verif_box_colors_f   = _DIAG_COLORS_F

                # ---- Per-camera projection (boxes only) ----
                L2C = lidar2cam.squeeze().numpy()                              # (4,4)
                K   = intrinsic.squeeze().numpy()                              # (3,3)
                img_vis = img.squeeze().permute(1, 2, 0).numpy().astype(np.uint8).copy()
                H_img, W_img = img_vis.shape[:2]

                for bi, box_corners in enumerate(self._verif_boxes):
                    bcolor      = self._verif_box_colors_u8[bi]
                    corners_h   = np.hstack([box_corners,
                                             np.ones((8, 1), dtype=np.float32)]).T  # (4,8)
                    corners_cam = L2C @ corners_h                              # (4,8)
                    in_front_c  = corners_cam[2] > 0.1                        # (8,)
                    z_safe      = np.where(in_front_c, corners_cam[2], 1.0)
                    proj_all    = K @ corners_cam[:3]                          # (3,8)
                    u_c = (proj_all[0] / z_safe).astype(int)
                    v_c = (proj_all[1] / z_safe).astype(int)

                    for e0, e1 in self._verif_box_edges:
                        if not (in_front_c[e0] and in_front_c[e1]):
                            continue
                        ret, p0, p1 = cv2.clipLine(
                            (0, 0, W_img, H_img),
                            (u_c[e0], v_c[e0]), (u_c[e1], v_c[e1]))
                        if ret:
                            cv2.line(img_vis, p0, p1, bcolor, 2, cv2.LINE_AA)

                    for ci in range(8):
                        if in_front_c[ci] and 0 <= u_c[ci] < W_img and 0 <= v_c[ci] < H_img:
                            cv2.circle(img_vis, (u_c[ci], v_c[ci]), 5, bcolor, -1)

                    # Label the box by its axis tag at the projected box centre
                    vis_corners = [(u_c[ci], v_c[ci]) for ci in range(8)
                                   if in_front_c[ci]
                                   and 0 <= u_c[ci] < W_img and 0 <= v_c[ci] < H_img]
                    if vis_corners:
                        cx_px = int(np.mean([p[0] for p in vis_corners]))
                        cy_px = int(np.mean([p[1] for p in vis_corners]))
                        cv2.putText(img_vis, self._verif_box_labels[bi],
                                    (cx_px - 15, cy_px),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, bcolor, 2, cv2.LINE_AA)

                cv2.putText(img_vis, cam_type.replace('CAM_', ''), (8, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2, cv2.LINE_AA)
                self._verif_buffer.append(img_vis)

                if len(self._verif_buffer) == 6:
                    cam_order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                 'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT']
                    cam_types_in_order = list(sample_onlinehdmap['cams'].keys())
                    idx_order = [cam_types_in_order.index(c) if c in cam_types_in_order else i
                                 for i, c in enumerate(cam_order)]
                    ordered = [self._verif_buffer[i] for i in idx_order]

                    # ---- Final canvas: 2×3 camera grid (left) + 3D scene (right) ----
                    fig = plt.figure(figsize=(28, 10))
                    gs  = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.4],
                                           wspace=0.04, hspace=0.04)

                    H0, W0 = ordered[0].shape[:2]
                    for row in range(2):
                        for col in range(3):
                            ax = fig.add_subplot(gs[row, col])
                            ax.imshow(cv2.resize(ordered[row * 3 + col], (W0, H0)))
                            ax.axis('off')

                    # ---- 3D view: box wireframes + ego heading arrows ----
                    ax3d = fig.add_subplot(gs[:, 3], projection='3d')

                    # Draw diagnostic boxes + their axis labels
                    for bi, box_corners in enumerate(self._verif_boxes):
                        fc    = self._verif_box_colors_f[bi]
                        label = self._verif_box_labels[bi]
                        for e0, e1 in self._verif_box_edges:
                            ax3d.plot([box_corners[e0, 0], box_corners[e1, 0]],
                                      [box_corners[e0, 1], box_corners[e1, 1]],
                                      [box_corners[e0, 2], box_corners[e1, 2]],
                                      color=fc, linewidth=2.0)
                        # Label at the top-centre of the box
                        cx = box_corners[:, 0].mean()
                        cy = box_corners[:, 1].mean()
                        cz = box_corners[:, 2].max() + 0.5
                        ax3d.text(cx, cy, cz, label, color=fc,
                                  fontsize=10, fontweight='bold', ha='center')

                    # Ego/lidar origin marker
                    ax3d.scatter([0], [0], [0],
                                 c='white', edgecolors='black', s=100, zorder=5)

                    # Heading arrows — confirmed lidar frame convention:
                    #   Y = forward (heading),  X = right,  Z = up
                    _arrow_len = 7.0
                    for (dx, dy, dz), color, label in [
                            ((0, _arrow_len, 0), 'green', 'Y  forward'),
                            ((_arrow_len, 0, 0), 'red',   'X  right'),
                            ((0, 0, _arrow_len), 'blue',  'Z  up'),
                    ]:
                        ax3d.quiver(0, 0, 0, dx, dy, dz,
                                    color=color, linewidth=2.5, arrow_length_ratio=0.2)
                        ax3d.text(dx * 1.1, dy * 1.1, dz * 1.1,
                                  label, color=color, fontsize=9, fontweight='bold')

                    ax3d.set_xlabel('X  right (m)', fontsize=8)
                    ax3d.set_ylabel('Y  forward (m)', fontsize=8)
                    ax3d.set_zlabel('Z  up (m)', fontsize=8)
                    ax3d.set_title(
                        'Lidar frame:  Y=forward  X=right  Z=up\n'
                        '  green=+Y(front)  magenta=-Y(back)\n'
                        '  red=+X(right)    cyan=-X(left)', fontsize=8)
                    ax3d.view_init(elev=35, azim=-50)
                    ax3d.set_box_aspect([1, 1, 0.3])

                    plt.savefig('./verif_lidar_projection.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print('[Verif] Saved projection to verif_lidar_projection.png')
                    print('[Verif] Lidar frame confirmed:  Y=forward  X=right  Z=up')
                    del self._verif_buffer, self._verif_boxes, self._verif_box_edges
                    del self._verif_box_colors_u8, self._verif_box_colors_f
                    del self._verif_box_labels
            # Verification code -----------------------


            # Save to lists
            images.append(img)
            intrinsics.append(intrinsic)
            cam2egos.append(cam2ego)
            ego2globals.append(ego2global)
            lidar2cams.append(lidar2cam)
            cam2lidars.append(cam2lidar)
            gt_depths.append(gt_depth)
            gt_pv_semantic_masks.append(gt_pv_semantic_mask)
            
        

        # Verification code -----------------------
        # Project Lidar point cloud to image
        if False:
            cam_order = [
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
            ]
            cam_keys = list(sample_onlinehdmap['cams'].keys())
            key_to_idx = {k: i for i, k in enumerate(cam_keys)}
            _alpha = float(self.cfg['nuscenes'].get('verif_gt_depth_overlay_alpha', 0.65))
            _pt_radius = int(self.cfg['nuscenes'].get('verif_gt_depth_point_radius', 5))

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            for slot, cam in enumerate(cam_order):
                ax = axes[slot]
                if cam not in key_to_idx:
                    ax.set_title(cam.replace('CAM_', '') + ' (missing)', fontsize=9)
                    ax.axis('off')
                    continue
                idx = key_to_idx[cam]
                img_t = images[idx].squeeze(0).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                if img_t.dtype != np.uint8:
                    img_t = np.clip(img_t, 0, 255).astype(np.uint8)
                vis = img_t.astype(np.float32)

                dep = gt_depths[idx].squeeze(0).squeeze(0).detach().cpu().numpy()
                H, W = vis.shape[:2]
                dep_rs = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST)
                valid = dep_rs > 1e-3
                dep_u8 = np.zeros((H, W), dtype=np.uint8)
                if valid.any():
                    dv = dep_rs[valid]
                    dmin, dmax = float(dv.min()), float(dv.max())
                    if dmax > dmin:
                        dep_u8[valid] = ((dep_rs[valid] - dmin) / (dmax - dmin) * 255.0).astype(
                            np.uint8
                        )
                    else:
                        dep_u8[valid] = 255
                dep_bgr = cv2.applyColorMap(dep_u8, cv2.COLORMAP_TURBO)
                dep_rgb = cv2.cvtColor(dep_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

                v_f = valid.astype(np.float32)
                if _pt_radius > 0:
                    ksz = 2 * _pt_radius + 1
                    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                    dep_spread = np.zeros_like(dep_rgb)
                    for c in range(3):
                        dep_spread[..., c] = cv2.dilate(
                            dep_rgb[..., c] * v_f, kern
                        )
                    valid_ov = (cv2.dilate(v_f, kern) > 0.5)
                else:
                    dep_spread = dep_rgb
                    valid_ov = valid

                blend = vis.copy()
                for c in range(3):
                    blend[..., c] = np.where(
                        valid_ov,
                        (1.0 - _alpha) * vis[..., c] + _alpha * dep_spread[..., c],
                        vis[..., c],
                    )
                ax.imshow(np.clip(blend, 0, 255).astype(np.uint8))
                ax.set_title(cam.replace('CAM_', ''), fontsize=10)
                ax.axis('off')

            fig.suptitle(
                'RGB + gt_depth (turbo, morphologically dilated hits; '
                f'r={_pt_radius}px, α={_alpha})',
                fontsize=12,
            )
            plt.tight_layout()
            _tok = sample_record.get('token', 'sample')
            _out = f'./verif_gt_depth_grid_{_tok}.png'
            plt.savefig(_out, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f'[Verif] Saved gt_depth grid to {_out}')
        # -----------------------------------------

        # Verification code -----------------------
        # Shows images, PV semantic mask, and BEV semantic mask
        if False:
            cam_order = [
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
            ]
            cam_keys = list(sample_onlinehdmap['cams'].keys())
            key_to_idx = {k: i for i, k in enumerate(cam_keys)}

            def _tensor_to_uint8_hwc(img_t):
                arr = (
                    img_t.squeeze(0)
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                return arr

            def _mask_to_2d_numpy(x):
                if torch.is_tensor(x):
                    m = x.detach().cpu().float().numpy()
                else:
                    m = np.asarray(x, dtype=np.float32)
                return np.squeeze(m)

            fig = plt.figure(figsize=(20, 18))
            gs = fig.add_gridspec(
                5, 3, height_ratios=[1.0, 1.0, 1.0, 1.0, 1.1], hspace=0.22, wspace=0.06
            )

            for slot, cam in enumerate(cam_order):
                row, col = slot // 3, slot % 3
                ax_rgb = fig.add_subplot(gs[row, col])
                if cam not in key_to_idx:
                    ax_rgb.set_title(cam.replace('CAM_', '') + ' (missing)', fontsize=9)
                    ax_rgb.axis('off')
                    continue
                idx = key_to_idx[cam]
                if idx >= len(images) or idx >= len(gt_pv_semantic_masks):
                    ax_rgb.set_title(cam.replace('CAM_', '') + ' (idx oob)', fontsize=9)
                    ax_rgb.axis('off')
                    continue

                ax_rgb.imshow(_tensor_to_uint8_hwc(images[idx]))
                ax_rgb.set_title(cam.replace('CAM_', '') + '  RGB', fontsize=10)
                ax_rgb.axis('off')

            for slot, cam in enumerate(cam_order):
                row, col = slot // 3, slot % 3
                ax_pv = fig.add_subplot(gs[2 + row, col])
                if cam not in key_to_idx:
                    ax_pv.axis('off')
                    continue
                idx = key_to_idx[cam]
                if idx >= len(gt_pv_semantic_masks):
                    ax_pv.axis('off')
                    continue

                pv = _mask_to_2d_numpy(gt_pv_semantic_masks[idx])
                vmax = max(1.0, float(pv.max()) if pv.size else 1.0)
                ax_pv.imshow(pv, cmap='turbo', vmin=0.0, vmax=vmax, interpolation='nearest')
                ax_pv.set_title(cam.replace('CAM_', '') + '  PV sem.', fontsize=10)
                ax_pv.axis('off')

            ax_bev = fig.add_subplot(gs[4, :])
            if gt_semantic_masks:
                bev = gt_semantic_masks[-1]
            else:
                bev = gt_semantic_mask
            bev = _mask_to_2d_numpy(bev)
            if bev.size == 0:
                ax_bev.text(0.5, 0.5, 'empty BEV mask', ha='center', va='center')
                ax_bev.axis('off')
            else:
                bmax = max(1.0, float(bev.max()))
                ax_bev.imshow(bev, cmap='turbo', vmin=0.0, vmax=bmax, interpolation='nearest', aspect='auto')
                ax_bev.set_title('BEV gt_semantic_mask', fontsize=11)
                ax_bev.set_xlabel('W'); ax_bev.set_ylabel('H')

            _tok = sample_record.get('token', 'sample')
            fig.suptitle(f'Semantic GT verification  (sample {_tok})', fontsize=13)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            _out = f'./verif_semantic_grid_{_tok}.png'
            plt.savefig(_out, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f'[Verif] Saved RGB / PV / BEV semantic grid to {_out}')
        # -----------------------------------------


        images = torch.cat(images, dim=1)
        intrinsics = torch.cat(intrinsics, dim=1)
        cam2egos = torch.cat(cam2egos, dim=1)
        ego2globals = torch.cat(ego2globals, dim=1)
        lidar2cams = torch.cat(lidar2cams, dim=1)
        cam2lidars = torch.cat(cam2lidars, dim=1)
        lidar2egos = torch.cat(lidar2egos, dim=1)
        gt_depths = torch.cat(gt_depths, dim=1)
        gt_pv_semantic_masks = torch.cat(gt_pv_semantic_masks, dim=1)
        gt_semantic_masks = torch.cat(gt_semantic_masks, dim=1)
        return images, intrinsics, cam2egos, ego2globals, lidar2cams, cam2lidars, lidar2egos, gt_depths, gt_pv_semantic_masks, gt_semantic_masks

    # ==================== Label Generation =========================


    def _get_polyline_labels(self, sample_record, sample_onlinehdmap):


        # ------------------------------------------------
        # Pose information generation

        # Build lidar2ego (4×4): rotation from quaternion + translation
        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(sample_onlinehdmap['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = sample_onlinehdmap['lidar2ego_translation']

        # Build ego2global (4×4): rotation from quaternion + translation
        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(sample_onlinehdmap['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = sample_onlinehdmap['ego2global_translation']

        # Chain transforms: LiDAR → ego → global
        lidar2global = ego2global @ lidar2ego

        # Extract the global position (XYZ) and orientation (quaternion) of the LiDAR sensor.
        # These are passed to the map query so it knows which area of the HD map to crop.
        lidar2global_translation = list(lidar2global[:3,3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)


        # ------------------------------------------------
        # Convert map layers (divider / ped_crossing / boundary) into vectorized polylines.
        location = sample_onlinehdmap['map_location']
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)

        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location)
                line_instances_dict = self.line_geoms_to_instances(line_geom) # multiline to singl lines -> dictionary{'road_divider': [line1, line2, ...], 'lane_divider': [line1, line2, ...]}    
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        vectors.append((instance, self.CLASS2LABEL.get(line_type, -1)))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    vectors.append((instance, self.CLASS2LABEL.get('ped_crossing', -1)))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for contour in poly_bound_list:
                    vectors.append((contour, self.CLASS2LABEL.get('contours', -1)))
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')
        
        
        gt_labels, gt_instances = [], []
        for instance, type in vectors:
            if type != -1:
                gt_instances.append(instance)
                gt_labels.append(type)        
        

        if (len(gt_instances) == 0):
            return [], [], [], [], []


        # ------------------------------------------------
        # Encapsulate into LiDARInstanceLines class
        gt_instances = LiDARInstanceLines(gt_instances, self.sample_dist,
                        self.num_samples, self.padding, self.fixed_num, self.padding_value, patch_size=self.patch_size)



        # ------------------------------------------------
        # Get the final polyline instances
        if self.hdmap_label_config['polyline']['gt_shift_pts_pattern'] == "v2":
            gt_bboxes = gt_instances.bbox # num_instances 4, (x_min, y_min, x_max, y_max)
            gt_polylines = gt_instances.fixed_num_sampled_points # num_instances num_points 2, (x, y) = (W, H) = (100, 200)
            gt_polylines_shift = gt_instances.shift_fixed_num_sampled_points_v2 # num_instances num_rotations num_points 2

            # # NOTE : All the coordinates follow (x, y) = (W, H) = (100, 200) in BEV space.
            # # Therefore, this part is deprecated.
            # gt_bboxes = gt_bboxes[:, [1, 0, 3, 2]]
            # gt_polylines = gt_polylines[:, :, [1, 0]]
            # gt_polylines_shift = gt_polylines_shift[:, :, :, [1, 0]]


        else:
            raise ValueError(f'WRONG gt_shift_pts_pattern: {self.cfg["nuscenes"]["OnlineHDmap"]["polyline"]["gt_shift_pts_pattern"]}')

        # For visualization ------------------
        # NuScenes convention: x = forward, y = left, z = up
        # BEV image layout:  top = forward (+x),  left = left (+y)
        # NOTE : gt_bboxes, gt_polylines, gt_polylines_shift are all xy swapped.

        if (False):
            from dataset.NuscenesDataset.visualization import visualize_polyline_on_bev, visualize_points_on_bev

            map_size_r = 60 * 5
            map_size_c = 30 * 5

            # gt_bboxes: (N, 4) as (xmin, ymin, xmax, ymax)
            # → gt_bbox_corners: (N, 4, 2)  clockwise from top-left
            #   corner order: TL(xmin,ymax), TR(xmax,ymax), BR(xmax,ymin), BL(xmin,ymin)
            xmin, ymin, xmax, ymax = gt_bboxes[:, 0], gt_bboxes[:, 1], gt_bboxes[:, 2], gt_bboxes[:, 3]
            gt_bbox_corners = torch.stack([
                torch.stack([xmin, ymax], dim=-1),  # top-left
                torch.stack([xmax, ymax], dim=-1),  # top-right
                torch.stack([xmax, ymin], dim=-1),  # bottom-right
                torch.stack([xmin, ymin], dim=-1),  # bottom-left
            ], dim=1)  # (N, 4, 2)

            for i in range(gt_bboxes.shape[0]):
                bev = np.zeros(shape=(map_size_r, map_size_c, 3))
                bev = visualize_polyline_on_bev(bev, gt_polylines[[i]], pc_range=self.pc_range, map_size=(map_size_r, map_size_c), color=(0, 0, 255))
                bev = visualize_points_on_bev(bev, gt_bbox_corners[[i]], pc_range=self.pc_range, map_size=(map_size_r, map_size_c), color=(255, 0, 0))
                cv2.imwrite(f'./line_drawings_verif_{i}.png', bev.astype(np.uint8))
        # -------------------------------------


        return gt_bboxes, gt_polylines, gt_polylines_shift, torch.from_numpy(np.array(gt_labels)), vectors

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        """
        Retrieve map geometries for the given patch and requested layers.

        Args:
            patch_box:    (cx, cy, H, W) defining the BEV crop region in map coords.
            patch_angle:  rotation angle (deg) of the patch, aligns it with ego heading.
            layer_names:  list of map layer names to query (e.g. ['lane_divider', 'road_segment']).
            location:     nuScenes map location string (e.g. 'boston-seaport').

        Returns:
            list of (layer_name, geoms) tuples, one per requested layer.
        """
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                # Lane dividers, road edges, etc. — represented as open polylines
                geoms = self.get_divider_line(patch_box, patch_angle, layer_name, location)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                # Drivable area, walkways, etc. — boundaries extracted as closed contour lines
                geoms = self.get_contour_line(patch_box, patch_angle, layer_name, location)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                # Pedestrian crossings — converted to lines spanning the crossing width
                geoms = self.get_ped_crossing_line(patch_box, patch_angle, location)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []
        
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(single_line))
                elif line.geom_type == 'LineString':
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []
        
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def ped_poly_geoms_to_instances(self, ped_geom):
        ped = ped_geom[0][1]
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def poly_geoms_to_instances(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def line_geoms_to_vectors(self, line_geom):

        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_instances = self._one_type_line_geom_to_instances(a_type_of_lines)
            line_instances_dict[line_type] = one_type_instances

        return line_instances_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_contour_line(self,patch_box,patch_angle,layer_name,location):
        if layer_name not in self.map_explorer[location].map_api.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_explorer[location].map_api, layer_name)

        polygon_list = []
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_explorer[location].map_api.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]

                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        else:
            for record in records:
                polygon = self.map_explorer[location].map_api.extract_polygon(record['polygon_token'])

                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        return polygon_list

    def get_divider_line(self,patch_box,patch_angle,layer_name,location):
        if layer_name not in self.map_explorer[location].map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == 'traffic_light':
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)

        line_list = []
        records = getattr(self.map_explorer[location].map_api, layer_name)
        for record in records:
            line = self.map_explorer[location].map_api.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)


        # For visualization ------------------
        # # NuScenes convention: x = forward, y = left, z = up
        # # BEV image layout:  top = forward (+x),  left = left (+y)
        # x_range = (self.pc_range[0], self.pc_range[3])
        # y_range = (self.pc_range[1], self.pc_range[4])
        # map_size = 512
        # axis_range_y = y_range[1] - y_range[0]
        # axis_range_x = x_range[1] - x_range[0]
        # scale_y = float(map_size - 1) / axis_range_y
        # scale_x = float(map_size - 1) / axis_range_x
        # img = np.zeros(shape=(map_size, map_size, 3))
        # for line in line_list:
        #     xy = np.array(line.xy).T  # (seq_len, 2): col 0 = x (forward), col 1 = y (left)

        #     seq_len = xy.shape[0]
        #     # y (left) → col: +y_max (leftmost) lands at col 0 (left image edge)
        #     col_img = -(xy[:, 1] * scale_y).astype(np.int32)
        #     # x (forward) → row: +x_max (furthest forward) lands at row 0 (top image edge)
        #     row_img = -(xy[:, 0] * scale_x).astype(np.int32)

        #     col_img += int(np.trunc(y_range[1] * scale_y))
        #     row_img += int(np.trunc(x_range[1] * scale_x))

        #     # pts layout for cv2: (col, row) = (horizontal, vertical)
        #     pts = np.concatenate([col_img.reshape(seq_len, 1), row_img.reshape(seq_len, 1)], axis=-1)
        #     cv2.polylines(img, [pts[:-1].reshape((-1, 1, 2))], isClosed=False, color=(0, 0, 255), thickness=2)
        # cv2.imwrite('./line_drawings_verif.png', img.astype(np.uint8))
        # -------------------------------------




        return line_list

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)
        polygon_list = []
        records = getattr(self.map_explorer[location].map_api, 'ped_crossing')
        # records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].map_api.extract_polygon(record['polygon_token'])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        return polygon_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)


        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples


        return sampled_points, num_valid

    def _get_bev_aug(self):
        
        if (self.mode == 'train' and self.cfg['nuscenes']['bev']['bool_apply_bev_aug'] and np.random.rand() < 0.5):
            return get_random_ref_matrix(self.cfg['nuscenes']['bev']['bev_aug'])
        else:
            return np.eye(4, dtype=np.float64)

    def line_ego_to_pvmask(self,
                          line_ego, 
                          mask, 
                          lidar2feat,
                          color=1, 
                          thickness=1,
                          z=-1.6):
        distances = np.linspace(0, line_ego.length, 200)
        coords = np.array([list(line_ego.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        pts_num = coords.shape[0]
        zeros = np.zeros((pts_num,1))
        zeros[:] = z
        ones = np.ones((pts_num,1))
        lidar_coords = np.concatenate([coords,zeros,ones], axis=1).transpose(1,0)
        pix_coords = perspective(lidar_coords, lidar2feat)
        return cv2.polylines(mask, np.int32([pix_coords]), False, color=color, thickness=thickness)
        
    def line_ego_to_mask(self, 
                         line_ego, 
                         mask, 
                         color=1, 
                         thickness=3):
        ''' Rasterize a single line to mask.
        
        Args:
            line_ego (LineString): line
            mask (array): semantic mask to paint on
            color (int): positive label, default: 1
            thickness (int): thickness of rasterized lines, default: 3
        '''

        trans_x = self.canvas_size[1] / 2
        trans_y = self.canvas_size[0] / 2
        line_ego = affinity.scale(line_ego, self.scale_x, self.scale_y, origin=(0, 0))
        line_ego = affinity.affine_transform(line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
        # print(np.array(list(line_ego.coords), dtype=np.int32).shape)
        coords = np.array(list(line_ego.coords), dtype=np.int32)[:, :2]
        coords = coords.reshape((-1, 2))
        assert len(coords) >= 2
        
        return cv2.polylines(mask, np.int32([coords]), False, color=color, thickness=thickness)

    # ==================== Utility Methods ====================

    def _get_split(self, split):
        """Get scene names for a split."""
        split_path = Path(__file__).parent / f'splits/{split}.txt'
        scene_list = split_path.read_text().strip().split('\n')

        # update, 260420
        # for scene in scene_list:
        #     if scene in SCENE_BLACKLIST:
        #         scene_list.remove(scene)
        
        return scene_list

    def _get_ordered_sample_records(self):
        """Get sample records ordered by scene and timestamp."""
        samples = [s for s in self.nusc.sample 
                  if self.nusc.get('scene', s['scene_token'])['name'] in self.target_scenes]       
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))
        return samples

    def _get_seq_sample_indices(self):
        """Get valid sequence indices."""
        indices = []
        for index in range(len(self.sample_records)):
            current_indices = []
            previous_rec = None
            is_valid = True

            for t in range(self.seq_len):
                idx_t = index + t
                if idx_t >= len(self.sample_records):
                    is_valid = False
                    break
                rec = self.sample_records[idx_t]
                if previous_rec and rec['scene_token'] != previous_rec['scene_token']:
                    is_valid = False
                    break
                current_indices.append(idx_t)
                previous_rec = rec

            if is_valid:
                indices.append(current_indices)

        return indices

    def _get_anns_by_category(self, sample, categories):
        
        # Get annotations grouped by category
        anns_by_category = [[] for _ in categories]
        for ann_token in self.nusc.get('sample', sample['token'])['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            tokens = ann['category_name'].split('.')
            for i, cat in enumerate(categories):
                if cat in tokens:
                    anns_by_category[i].append(ann)
                    break
        
        box_anns = []
        for anns in anns_by_category:
            for ann in anns:
                # to global_coords
                box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                box_anns.append(box)

        return box_anns

    def _get_ego_pose(self, pose_token):
        return self.nusc.get('ego_pose', pose_token)

    def _transform_box(self, box, ego_pose):
        box.transform_to_pose(ego_pose)

    def _is_valid_agent(self, category, attribute):
        """Check if agent should be included."""
        is_ped = 'pedestrian' in category and 'stroller' not in category and 'wheelchair' not in category
        is_veh = 'vehicle' in category and 'bicycle' not in category and 'motorcycle' not in category and 'parked' not in attribute
        return is_ped or is_veh

    def _make_rot_matrix(self, headings):
        """Create rotation matrices from headings."""
        n = len(headings)
        m_cos = np.cos(headings).reshape(n, 1)
        m_sin = np.sin(headings).reshape(n, 1)
        return np.concatenate([m_cos, -m_sin, m_sin, m_cos], axis=1).reshape(n, 2, 2)

    def _filter_samples_in_blacklist(self, seq_sample_indices):
        '''
        'seq_sample_indices' is a list of sequence indices.
        '''
        
        # Read frame tokens out of map
        frame_tokens_out_of_map_path = Path(__file__).parent / 'splits/frame_tokens_out_of_map.txt'
        frame_tokens_out_of_map = frame_tokens_out_of_map_path.read_text().strip().split('\n')
        key_frame_blacklist = [token.split(':')[0] for token in frame_tokens_out_of_map]
        
        new_seq_sample_indices = []
        for seq_index in seq_sample_indices:
            is_valid = True
            for index in seq_index:
                rec = self.sample_records[index]
                if rec['scene_token'] in key_frame_blacklist:
                    is_valid = False
                    break
            
            if is_valid:
                new_seq_sample_indices.append(seq_index)
        
        return new_seq_sample_indices

 