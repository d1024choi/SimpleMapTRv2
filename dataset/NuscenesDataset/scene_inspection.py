"""Inspection of NuScenes Dataset."""

import copy
import glob
import os
import sys
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pyquaternion import Quaternion
from shapely.geometry import MultiPolygon
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.data_classes import LidarPointCloud, Box

from utils.functions import read_config, config_update, check_dataset_path_existence
from utils.geometry import calculate_birds_eye_view_parameters, update_intrinsics, resize_and_crop_image, get_random_ref_matrix
from dataset.NuscenesDataset.common import CAMERAS, MAP_NAMES, STATIC, DYNAMIC, DIVIDER, INTERPOLATION, get_pose
from dataset.NuscenesDataset.map import Map


class DatasetLoader(Dataset):

    def __init__(self, mode='train'):

        # Define mode and split
        split = 'train' if mode in ['train', 'val', 'valid'] else 'val'

        # Configuration
        self.mode = mode
        self.cfg = read_config()
        self.seq_len = 3
        self.img_norm = transforms.Compose([transforms.ToTensor()])

        # Image preprocessing parameters
        ori_dims = (self.cfg['BEV']['original_image']['w'], self.cfg['BEV']['original_image']['h'])
        resize_dims = (480, 224 + 46)
        crop = (0, 46, resize_dims[0], resize_dims[1])
        self.img_prepro_params = {
            'scale_width': resize_dims[0] / ori_dims[0],
            'scale_height': resize_dims[1] / ori_dims[1],
            'resize_dims': resize_dims,
            'crop': crop
        }

        # BEV grid parameters
        self.bev_resolution, self.bev_start_position, self.bev_dimension = \
            calculate_birds_eye_view_parameters(
                self.cfg['BEV']['lift']['x_bound'],
                self.cfg['BEV']['lift']['y_bound'],
                self.cfg['BEV']['lift']['z_bound'],
                isnumpy=True
            )

        # Load data
        self._load_nuscenes_data(split, mode)

    def _load_nuscenes_data(self, split, mode):
        """Load data directly from NuScenes."""
        
        # Initialize NuScenes
        self.nusc_map = {v: NuScenesMap(dataroot=self.cfg['nuscenes']['dataset_dir'], map_name=v) for v in MAP_NAMES}
        self.nusc = NuScenes(version=self.cfg['nuscenes']['version'], dataroot=self.cfg['nuscenes']['dataset_dir'], verbose=False)

        # from dataset.NuscenesDataset.map import Map
        self.hdmap = Map(self.cfg['nuscenes']['dataset_dir'], self.nusc)

        # Get scene splits
        self.target_scenes = self._get_split(split) # Splits into train/valid/test
        self.sample_records = self._get_ordered_sample_records() # scene0-00, scene0-01, ..., scene1-00, scene1-01, ...
        seq_sample_indices = self._get_seq_sample_indices() # [[0, 1, 2, ...], [1, 2, 3, ...], [2, 3, 4, ...], ...]

        # Skip samples in blacklist
        if self.cfg['BEV']['bool_filter_samples_in_blacklist']:
            seq_sample_indices = self._filter_samples_in_blacklist(seq_sample_indices)

        # Split into train/val
        self.scenes = seq_sample_indices
        self.num_scenes = len(self.scenes)


    def __len__(self):
        return self.num_scenes

    def __getitem__(self, idx):

        seq_indices = np.array(self.scenes[idx])[self.cfg['target_frame_indices']] # same record indices over the entire time horizon
        return self._extract_seq_data(seq_indices)


    # ==================== Data Extraction ====================

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
                if rec['token'] in key_frame_blacklist:
                    is_valid = False
                    break
            
            if is_valid:
                new_seq_sample_indices.append(seq_index)
        
        return new_seq_sample_indices

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
            Apply np.flipud(np.fliplr(bev)) to match ego-centric frame (forward-up, side-left).
        """
        
        skip_img = False
                
        seq_images, seq_intrinsics, seq_extrinsics, seq_egolidar2worlds = [], [], [], []
        seq_bev, seq_bev_no_aug, seq_center, seq_visibility = [], [], [], []
        token = []
        instance_map = {}
        seq_bev_aug_mat = []

        for i in seq_indices:

            rec = self.sample_records[i]
            token.append(rec['token'])
            
            # Camera data
            if not skip_img:
                images, intrinsics, extrinsics, egolidar2world = self._get_input_data(rec)
                seq_images.append(images)
                seq_intrinsics.append(intrinsics)
                seq_extrinsics.append(extrinsics)
                seq_egolidar2worlds.append(egolidar2world[None])

            # bev_aug generation
            bev_aug_mat = np.eye(4, dtype=np.float64)
            

            # BEV labels
            data, instance_map = self._get_bev_labels(rec, instance_map, bev_aug_mat)
            bev = torch.flip(data['bev'], dims=(2, 3))
            bev_no_aug = torch.flip(data['bev_no_aug'], dims=(2, 3))
            center = torch.cat((data['aux']['center_score_veh'], data['aux']['center_score_ped']), dim=1)
            center = torch.flip(center, dims=(2, 3))
            visibility = torch.flip(data['visibility'], dims=(2, 3))
            bev_aug_mat = torch.from_numpy(bev_aug_mat).unsqueeze(0)

            seq_bev.append(bev)
            seq_bev_no_aug.append(bev_no_aug)
            seq_center.append(center)
            seq_visibility.append(visibility)
            seq_bev_aug_mat.append(bev_aug_mat)


        result = {
            'bev': torch.cat(seq_bev, dim=0),
            'bev_no_aug': torch.cat(seq_bev_no_aug, dim=0),
            'center': torch.cat(seq_center, dim=0),
            'visibility': torch.cat(seq_visibility, dim=0),
            'bev_aug_mat': torch.cat(seq_bev_aug_mat, dim=0),
            'token': token,
        }
        
        if not skip_img:
            result.update({
                'image': torch.cat(seq_images, dim=0),
                'intrinsics': torch.cat(seq_intrinsics, dim=0),
                'extrinsics': torch.cat(seq_extrinsics, dim=0),
                'egolidar2worlds': torch.cat(seq_egolidar2worlds, dim=0),
            })

        return result

      
        

    # ==================== NuScenes Data Loading ====================

    def _get_input_data(self, sample_record):
        """Get camera images, intrinsics, and extrinsics."""
       
        top_crop = self.img_prepro_params['crop'][1]
        left_crop = self.img_prepro_params['crop'][0]
        scale_width = self.img_prepro_params['scale_width']
        scale_height = self.img_prepro_params['scale_height']

        # Ego poses
        lidar_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        egopose_lidar = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])
        egolidar2world = torch.from_numpy(get_pose(egopose_lidar['rotation'], egopose_lidar['translation'], flat=False))

        images, proj_depths, intrinsics, extrinsics, egocam2worlds = [], [], [], [], []
        for camera in CAMERAS:
            cam_token = sample_record['data'][camera]
            cam_record = self.nusc.get('sample_data', cam_token)
            cam = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
            egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
            
            # Intrinsics
            intrinsic = torch.from_numpy(np.array(cam['camera_intrinsic']))
            intrinsic = update_intrinsics(intrinsic, top_crop, left_crop, scale_width=scale_width, scale_height=scale_height)
            
            # Extrinsics
            extrinsic = torch.from_numpy(get_pose(cam['rotation'], cam['translation'], flat=False, inv=True))
            
            # Egocam to world
            # egocam2world = torch.from_numpy(get_pose(egocam['rotation'], egocam['translation'], flat=False, inv=False))
            
            # Image
            img = Image.open(Path(self.nusc.get_sample_data_path(cam_token)))
            img = resize_and_crop_image(img, resize_dims=self.img_prepro_params['resize_dims'], crop=self.img_prepro_params['crop'])
            img = self.img_norm(img) # TODO : add boolean flog
            images.append(img.unsqueeze(0).unsqueeze(0))
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            extrinsics.append(extrinsic.unsqueeze(0).unsqueeze(0))
            # egocam2worlds.append(egocam2world.unsqueeze(0).unsqueeze(0))

        images = torch.cat(images, dim=1)
        intrinsics = torch.cat(intrinsics, dim=1)
        extrinsics = torch.cat(extrinsics, dim=1)

        return images, intrinsics, extrinsics, egolidar2world

    # ==================== BEV Label Generation ====================

    def _get_bev_labels(self, sample_record, instance_map, bev_aug):
        
        """Generate BEV segmentation labels."""
        scene_token = sample_record['scene_token']
        scene_record = self.nusc.get('scene', scene_token)
        location = self.nusc.get('log', scene_record['log_token'])['location']
        lidar_sample = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        egopose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])

        # Get annotations
        anns_dynamic = self._get_anns_by_category(sample_record, DYNAMIC)

        # Generate BEV layers       
        static = self._get_static_layers(location, egopose, bev_aug, STATIC)
        dividers = self._get_line_layers(location, egopose, bev_aug, DIVIDER)
        dynamic = self._get_dynamic_layers(anns_dynamic, egopose, bev_aug)
        bev = np.concatenate((static, dividers, dynamic), -1)

        # debug ---
        static = self._get_static_layers(location, egopose, np.eye(4, dtype=np.float64), STATIC)
        dividers = self._get_line_layers(location, egopose, np.eye(4, dtype=np.float64), DIVIDER)
        dynamic = self._get_dynamic_layers(anns_dynamic, egopose, np.eye(4, dtype=np.float64))
        bev_no_aug = np.concatenate((static, dividers, dynamic), -1)


        # Get dynamic object annotations: aux dict (center scores/offsets/instances for veh/ped),
        # visibility maps (vis_veh, vis_ped), and updated instance_map
        anns_all = [ann for anns_list in anns_dynamic for ann in anns_list]
        aux, vis_veh, vis_ped, instance_map = self._get_dynamic_objects(anns_all, egopose, bev_aug, instance_map)


        # Convert to tensors
        bev = torch.from_numpy(bev).permute(2, 0, 1).unsqueeze(0).contiguous()
        bev_no_aug = torch.from_numpy(bev_no_aug).permute(2, 0, 1).unsqueeze(0).contiguous() # debug ---
        aux_tensors = {k: torch.from_numpy(v).permute(2, 0, 1).unsqueeze(0).contiguous() for k, v in aux.items()}
        visibility = torch.cat([
            torch.from_numpy(vis_veh).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(vis_ped).unsqueeze(0).unsqueeze(0)], dim=1)

        return {'bev': bev, 'aux': aux_tensors, 'visibility': visibility, 'bev_no_aug': bev_no_aug,
                'location': location, 'egopose': egopose, 'bev_aug': bev_aug}, instance_map

    def _get_static_layers(self, location, egopose, bev_aug, layers, patch_radius=150):
        """Get static map layers (drivable area, etc.).      
        Layers = ['lane', 'road_segment', 'ped_crossing', 'walkway', 'carpark_area', 'stop_line']
        """
        
        h, w = self.cfg['BEV']['bev']['h'], self.cfg['BEV']['bev']['w']
        trans = -np.array(egopose['translation'])[:2]
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw/2), vector=[0, 0, np.sin(yaw/2)]).inverse.rotation_matrix[:2, :2]

        # Prepare BEV augmentation transformation
        if bev_aug is not None and not np.allclose(bev_aug, np.eye(4)):
            R_aug = bev_aug[:3, :3].T
            rot_aug_2d = R_aug[:2, :2]  # Extract 2D rotation from 3D rotation matrix
            t_aug = np.array([-1, -1, 1]) * bev_aug[:3, 3]
            t_aug_2d = t_aug[:2]  # Extract 2D translation
        else:
            rot_aug_2d = np.eye(2)
            t_aug_2d = np.zeros(2)

        bev_center = -self.bev_start_position[:2] + 0.5 * self.bev_resolution[:2]
        bev_res = self.bev_resolution[:2]

        pose = get_pose(egopose['rotation'], egopose['translation'], flat=True)
        x, y = pose[0][-1], pose[1][-1]
        box_coords = (x - patch_radius, y - patch_radius, x + patch_radius, y + patch_radius)
        records = self.nusc_map[location].get_records_in_patch(box_coords, layers, 'intersect')

        result = []
        for layer in layers:
            render = np.zeros((h, w), dtype=np.uint8)

            for r in records[layer]:
                polygon_token = self.nusc_map[location].get(layer, r)
                polygon_tokens = polygon_token['polygon_tokens'] if layer == 'drivable_area' else [polygon_token['polygon_token']]

                for p_token in polygon_tokens:
                    polygon = MultiPolygon([self.nusc_map[location].extract_polygon(p_token)])

                    for poly in polygon.geoms:
                        # Exterior
                        ext = np.array(poly.exterior.coords).T
                        ext = rot @ (ext.T + trans).T
                        
                        # Apply BEV augmentation: translate first, then rotate
                        ext = rot_aug_2d @ (ext.T + t_aug_2d).T
                        
                        ext = np.round((ext.T + bev_center) / bev_res).astype(np.int32)
                        ext = np.fliplr(ext)
                        cv2.fillPoly(render, [ext], 1, INTERPOLATION)

                        # Interior holes
                        for interior in poly.interiors:
                            int_pts = np.array(interior.coords).T
                            int_pts = rot @ (int_pts.T + trans).T
                            # Apply BEV augmentation: translate first, then rotate
                            int_pts = rot_aug_2d @ (int_pts.T + t_aug_2d).T
                            int_pts = np.round((int_pts.T + bev_center) / bev_res).astype(np.int32)
                            int_pts = np.fliplr(int_pts)
                            cv2.fillPoly(render, [int_pts], 0, INTERPOLATION)

            result.append(render)

        return np.stack(result, -1).astype('float32')

    def _get_line_layers(self, location, egopose, bev_aug, layers, patch_radius=150, thickness=1):
        """Get line layers (lane dividers, etc.)."""
        trans = -np.array(egopose['translation'])[:2]
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw/2), vector=[0, 0, np.sin(yaw/2)]).inverse.rotation_matrix[:2, :2]

        # Prepare BEV augmentation transformation
        if bev_aug is not None and not np.allclose(bev_aug, np.eye(4)):
            R_aug = bev_aug[:3, :3].T
            rot_aug_2d = R_aug[:2, :2]  # Extract 2D rotation from 3D rotation matrix
            t_aug = np.array([-1, -1, 1]) * bev_aug[:3, 3]
            t_aug_2d = t_aug[:2]  # Extract 2D translation
        else:
            rot_aug_2d = np.eye(2)
            t_aug_2d = np.zeros(2)

        bev_center = -self.bev_start_position[:2] + 0.5 * self.bev_resolution[:2]
        bev_res = self.bev_resolution[:2]

        pose = get_pose(egopose['rotation'], egopose['translation'], flat=True)
        x, y = pose[0][-1], pose[1][-1]
        box_coords = (x - patch_radius, y - patch_radius, x + patch_radius, y + patch_radius)
        records = self.nusc_map[location].get_records_in_patch(box_coords, layers, 'intersect')

        result = []
        h, w = self.cfg['BEV']['bev']['h'], self.cfg['BEV']['bev']['w']

        for layer in layers:
            render = np.zeros((h, w), dtype=np.uint8)

            for r in records[layer]:
                polygon_token = self.nusc_map[location].get(layer, r)
                line = self.nusc_map[location].extract_line(polygon_token['line_token'])
                p = np.float32(line.xy)
                p = rot @ (p.T + trans).T
                # Apply BEV augmentation: translate first, then rotate
                p = rot_aug_2d @ (p.T + t_aug_2d).T
                p = np.round((p.T + bev_center) / bev_res).astype(np.int32)
                p = np.fliplr(p)
                cv2.polylines(render, [p], False, 1, thickness=thickness)

            result.append(render)

        return np.stack(result, -1).astype(np.float32)

    def _get_dynamic_layers(self, anns_by_category, egopose, bev_aug):
        """Get dynamic object layers (vehicles, pedestrians)."""
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw/2), vector=[0, 0, np.sin(yaw/2)]).inverse
        # rot = Quaternion(egopose['rotation']).inverse


        bev_center = -self.bev_start_position[:2] + 0.5 * self.bev_resolution[:2]
        bev_res = self.bev_resolution[:2]
        h, w = self.cfg['BEV']['bev']['h'], self.cfg['BEV']['bev']['w']

        result = []
        for anns in anns_by_category:
            render = np.zeros((h, w), dtype=np.uint8)
            for ann in anns:
                box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                box.translate(trans)
                box.rotate(rot)

                # Apply BEV augmentation to the box
                if bev_aug is not None and not np.allclose(bev_aug, np.eye(4)):
                    R_aug = bev_aug[:3, :3].T 
                    rot_scipy = R.from_matrix(R_aug)
                    quat_xyzw = rot_scipy.as_quat()  # [x, y, z, w]
                    rot_aug = Quaternion(scalar=quat_xyzw[3], vector=quat_xyzw[:3])  # [w, x, y, z]
                    t_aug = np.array([-1, -1, 1]) * bev_aug[:3, 3]
                    
                    # Apply transformation: translate first, then rotate
                    box.translate(t_aug)
                    box.rotate(rot_aug)


                pts = box.bottom_corners()[:2].T
                pts = np.round((pts + bev_center) / bev_res).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(render, [pts], 1.0, INTERPOLATION)
            result.append(render)

        return np.stack(result, -1).astype('float32')

    def _get_dynamic_objects(self, anns, egopose, bev_aug, ins_map):
        """Get dynamic object attributes (center, offset, visibility)."""
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw/2), vector=[0, 0, np.sin(yaw/2)]).inverse

        h, w = self.cfg['BEV']['bev']['h'], self.cfg['BEV']['bev']['w']
        bev_center = -self.bev_start_position[:2] + 0.5 * self.bev_resolution[:2]
        bev_res = self.bev_resolution[:2]

        # Initialize arrays
        center_score_veh = np.zeros((h, w), dtype=np.float32)
        center_score_ped = np.zeros((h, w), dtype=np.float32)
        center_offset_veh = np.zeros((h, w, 2), dtype=np.float32)
        center_offset_ped = np.zeros((h, w, 2), dtype=np.float32)
        visibility_veh = np.full((h, w), 255, dtype=np.uint8)
        visibility_ped = np.full((h, w), 255, dtype=np.uint8)
        instance_veh = np.zeros((h, w), dtype=np.uint8)
        instance_ped = np.zeros((h, w), dtype=np.uint8)

        sigma = 1
        buf = np.zeros((h, w), dtype=np.uint8)
        coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)

        for ann in anns:
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
            box.translate(trans)
            box.rotate(rot)

            # Apply BEV augmentation to the box
            if bev_aug is not None and not np.allclose(bev_aug, np.eye(4)):
                R_aug = bev_aug[:3, :3].T 
                rot_scipy = R.from_matrix(R_aug)
                quat_xyzw = rot_scipy.as_quat()  # [x, y, z, w]
                rot_aug = Quaternion(scalar=quat_xyzw[3], vector=quat_xyzw[:3])  # [w, x, y, z]
                t_aug = np.array([-1, -1, 1]) * bev_aug[:3, 3]
                
                # Apply transformation: translate first, then rotate
                box.translate(t_aug)
                box.rotate(rot_aug)

            p = box.bottom_corners()[:2].T
            p = np.round((p + bev_center) / bev_res).astype(np.int32)
            p[:, [1, 0]] = p[:, [0, 1]]

            center = np.round((box.center[:2] + bev_center) / bev_res).astype(np.int32).reshape(1, 2)
            center = np.fliplr(center)

            buf.fill(0)
            cv2.fillPoly(buf, [p], 1, INTERPOLATION)
            mask = buf > 0

            if not np.count_nonzero(mask):
                continue

            # Instance ID
            if ann['instance_token'] not in ins_map:
                ins_map[ann['instance_token']] = len(ins_map) + 1
            ins_id = ins_map[ann['instance_token']]

            # Update vehicle or pedestrian arrays
            if 'vehicle' in ann['category_name']:
                visibility_veh[mask] = ann['visibility_token']
                instance_veh[mask] = ins_id
                center_offset_veh[mask] = center - coords[mask]
                center_score_veh[mask] = np.exp(-(center_offset_veh[mask] ** 2).sum(-1) / (sigma ** 2))
            elif 'pedestrian' in ann['category_name']:
                visibility_ped[mask] = ann['visibility_token']
                instance_ped[mask] = ins_id
                center_offset_ped[mask] = center - coords[mask]
                center_score_ped[mask] = np.exp(-(center_offset_ped[mask] ** 2).sum(-1) / (sigma ** 2))

        result = {
            'center_score_veh': center_score_veh[..., None],
            'center_score_ped': center_score_ped[..., None],
            'center_offset_veh': center_offset_veh,
            'center_offset_ped': center_offset_ped,
            'instance_veh': instance_veh[..., None],
            'instance_ped': instance_ped[..., None]
        }

        return result, visibility_veh, visibility_ped, ins_map


    # ==================== Utility Methods ====================

    def _get_split(self, split):
        """Get scene names for a split."""
        split_path = Path(__file__).parent / f'splits/{split}.txt'
        return split_path.read_text().strip().split('\n')

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
        """Get annotations grouped by category."""
        result = [[] for _ in categories]
        for ann_token in self.nusc.get('sample', sample['token'])['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            tokens = ann['category_name'].split('.')
            for i, cat in enumerate(categories):
                if cat in tokens:
                    result[i].append(ann)
                    break
        return result

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

    def reform_batch(self, batch, target_index, isTrain=True):
        """Select target frame from sequential batch data."""
        result = {}
        for k, v in batch.items():
            if k == 'token':
                # Handle token list separately - extract target_index from each batch item's token list
                # After DataLoader batching, token is a list of lists: [['token1', 'token2'], ['token3', 'token4']]
                if isinstance(v[0], list):
                    # Batched: extract target_index from each item's token list
                    result[k] = [item[target_index] if len(item) > abs(target_index) else item[-1] for item in v]
                else:
                    # Single item (batch_size=1): v is already a single list
                    result[k] = v[target_index] if len(v) > abs(target_index) else v[-1]
            else:
                result[k] = v[:, target_index] if isTrain else v[target_index].unsqueeze(0)
        return result

    def return_label(self, label, label_indices):
        label = torch.cat([label[:, idx].max(dim=1, keepdim=True).values for idx in label_indices], dim=1)
        return label

if __name__ == "__main__":
    """Example usage of DatasetLoader with batch loading."""
    from torch.utils.data import DataLoader
    
    # Initialize DatasetLoader
    print("Initializing DatasetLoader...")
    dataset = DatasetLoader(mode='train')
    print(f"Dataset loaded with {len(dataset)} scenes")
    
   
    # Create DataLoader for batching
    batch_size = 1
    num_workers = 0  # Set to 0 for debugging, increase for faster loading
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Set to True for training
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    
    # Open file to save tokens
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    token_file_path = output_dir / 'tokens_with_low_drivable.txt'
    token_file = open(token_file_path, 'w')
    
    # Load and process batches
    print(f"\nLoading batches (batch_size={batch_size})...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches", unit="batch")):

        batch = dataset.reform_batch(batch, target_index=-1, isTrain=True)
        bev = batch['bev'] # b x 12 x h x w
        image = batch['image'] # b x n x 3 x h x w
        token = batch['token'] # b x 1

        drivable_label = dataset.return_label(bev, dataset.cfg['BEV']['label_indices']['drivable']) # b x 1 x h x w
        vehicle_label = dataset.return_label(bev, dataset.cfg['BEV']['label_indices']['vehicle']) # b x 1 x h x w
        divider_label = dataset.return_label(bev, dataset.cfg['BEV']['label_indices']['divider']) # b x 1 x h x w
        ped_crossing_label = dataset.return_label(bev, dataset.cfg['BEV']['label_indices']['ped_crossing']) # b x 1 x h x w
        walkway_label = dataset.return_label(bev, dataset.cfg['BEV']['label_indices']['walkway']) # b x 1 x h x w
        carpark_area_label = dataset.return_label(bev, dataset.cfg['BEV']['label_indices']['carpark_area']) # b x 1 x h x w


        if (drivable_label[0].sum() < 3):
            # Save token to file
            token_str = token[0] if isinstance(token, list) else str(token[0])
            token_file.write(f"{token_str}\n")
            
            # Convert tensors to numpy for visualization (take first batch item)
            drivable_np = drivable_label[0, 0].cpu().numpy()  # h x w
            vehicle_np = vehicle_label[0, 0].cpu().numpy()  # h x w
            divider_np = divider_label[0, 0].cpu().numpy()  # h x w
            ped_crossing_np = ped_crossing_label[0, 0].cpu().numpy()  # h x w
            walkway_np = walkway_label[0, 0].cpu().numpy()  # h x w
            carpark_area_np = carpark_area_label[0, 0].cpu().numpy()  # h x w
            
            h, w = drivable_np.shape
            
            # Define colors (BGR format for cv2, 0-255)
            colors = {
                'drivable': (110, 110, 110),      # dark grey
                'vehicle': (255, 158, 0),         # orange/blue (BGR: RGB(0,158,255))
                'divider': (0, 0, 255),           # red (BGR: RGB(255,0,0))
                'ped_crossing': (153, 154, 251), # light pink (BGR: RGB(251,154,153))
                'walkway': (0, 128, 0),           # dark green (BGR: RGB(0,128,0))
                'carpark_area': (0, 127, 255),   # orange (BGR: RGB(255,127,0))
                'background': (200, 200, 200)     # light grey
            }
            
            # Create label visualization canvas (h x w x 3)
            label_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            label_canvas[:] = colors['background']
            
            # Overlay labels with different colors (order matters - later labels overwrite earlier ones)
            # Use threshold to determine presence of label
            threshold = 0.5
            
            # Drivable area (base layer)
            mask = drivable_np > threshold
            label_canvas[mask] = colors['drivable']
            
            # Walkway
            mask = walkway_np > threshold
            label_canvas[mask] = colors['walkway']
            
            # Carpark area
            mask = carpark_area_np > threshold
            label_canvas[mask] = colors['carpark_area']
            
            # Pedestrian crossing
            mask = ped_crossing_np > threshold
            label_canvas[mask] = colors['ped_crossing']
            
            # Divider (lines)
            mask = divider_np > threshold
            label_canvas[mask] = colors['divider']
            
            # Vehicle (top layer)
            mask = vehicle_np > threshold
            label_canvas[mask] = colors['vehicle']
            
            # Add red circle at top center to mark center
            center_x = w // 2
            circle_radius = 5
            cv2.circle(label_canvas, (center_x, circle_radius), circle_radius, (255, 0, 0), -1)  # Red filled circle
            
            # Prepare images for 2x3 grid
            # image shape: b x n x 3 x h_img x w_img
            image_np = image[0].cpu().numpy()  # n x 3 x h_img x w_img
            n_cameras = image_np.shape[0]
            
            # Convert from CHW to HWC and denormalize if needed
            images_list = []
            for i in range(min(n_cameras, 6)):  # Take up to 6 images
                img = image_np[i].transpose(1, 2, 0)  # H x W x 3
                # Denormalize if normalized to [0, 1]
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                images_list.append(img)
            
            # Pad with black images if less than 6
            while len(images_list) < 6:
                h_img, w_img = images_list[0].shape[:2] if images_list else (224, 480)
                images_list.append(np.zeros((h_img, w_img, 3), dtype=np.uint8))
            
            # Create 2x3 grid of images
            h_img, w_img = images_list[0].shape[:2]
            image_grid = np.zeros((2 * h_img, 3 * w_img, 3), dtype=np.uint8)
            
            for idx, img in enumerate(images_list[:6]):
                row = idx // 3
                col = idx % 3
                y_start = row * h_img
                y_end = y_start + h_img
                x_start = col * w_img
                x_end = x_start + w_img
                image_grid[y_start:y_end, x_start:x_end] = img
            
            # Convert image_grid from RGB to BGR (cv2 format)
            image_grid_bgr = cv2.cvtColor(image_grid, cv2.COLOR_RGB2BGR)
            
            # Resize label canvas to match image grid height for side-by-side placement
            label_h, label_w = label_canvas.shape[:2]
            target_height = image_grid_bgr.shape[0]
            scale_factor = target_height / label_h
            new_label_w = int(label_w * scale_factor)
            label_canvas_resized = cv2.resize(label_canvas, (new_label_w, target_height), interpolation=cv2.INTER_NEAREST)
            
            # Concatenate label canvas and image grid side-by-side (both in BGR format)
            final_image = np.concatenate([label_canvas_resized, image_grid_bgr], axis=1)
            
            # Save the image
            output_dir = Path('outputs')
            output_dir.mkdir(exist_ok=True)
            token_str = token[0] if isinstance(token, list) else str(token[0])
            save_path = output_dir / f'visualization_batch{batch_idx}_token_{token_str[:8]}.png'
            
            # Save image (cv2.imwrite expects BGR, which we're already using)
            cv2.imwrite(str(save_path), final_image)
            
            print(f"Visualization saved to: {save_path}")
            print(f"  - Label canvas shape: {label_canvas.shape}")
            print(f"  - Image grid shape: {image_grid.shape}")
            print(f"  - Final image shape: {final_image.shape}")
            print(f"  - Token: {token_str}")

    # Close the token file
    token_file.close()
    print(f"\nTokens saved to: {token_file_path}")
    

