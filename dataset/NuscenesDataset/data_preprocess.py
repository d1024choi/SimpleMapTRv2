import numpy as np
import torch
from nuscenes.eval.common.utils import Quaternion

class _StandalonePoints:
    """Minimal points container used by standalone depth pipeline.

    The original pipeline stores points as a `BasePoints` instance with a `.tensor`
    attribute. For the standalone validation we only need `.tensor`.
    """

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor


class _StandaloneCustomPointToMultiViewDepth(object):
    """Standalone copy of CustomPointToMultiViewDepth without importing its module.

    This avoids importing `projects/.../pipelines/loading.py` which imports `mmcv`
    at module import time.
    """

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        # imgs = np.stack(results['img'])
        # img_aug_matrix = results['img_aug_matrix']
        # post_rots = [torch.tensor(single_aug_matrix[:3, :3]).to(torch.float) for single_aug_matrix in img_aug_matrix]
        # post_trans = torch.stack([torch.tensor(single_aug_matrix[:3, 3]).to(torch.float) for single_aug_matrix in img_aug_matrix])
        intrins = results['camera_intrinsics']
        img_size = results['img_size']
        # depth_map_list = []

        # for cid in range(results['num_cams']):
            # lidar2lidarego = torch.tensor(results['lidar2ego']).to(torch.float32)

            # lidarego2global = np.eye(4, dtype=np.float32)
            # lidarego2global[:3, :3] = Quaternion(results['ego2global_rotation']).rotation_matrix
            # lidarego2global[:3, 3] = results['ego2global_translation']
            # lidarego2global = torch.from_numpy(lidarego2global)


        # cam2camego = torch.tensor(results['camera2ego'][cid])
        # camego2global = results['camego2global'][cid]
        cam2img = torch.as_tensor(intrins, dtype=torch.float32)
        if cam2img.shape == (3, 3):
            z = torch.zeros((3, 1), dtype=cam2img.dtype, device=cam2img.device)
            bottom = torch.tensor(
                [[0.0, 0.0, 0.0, 1.0]], dtype=cam2img.dtype, device=cam2img.device
            )
            cam2img = torch.cat([torch.cat([cam2img, z], dim=1), bottom], dim=0)
        lidar2cam = torch.tensor(results['lidar2cam']).to(torch.float32)

        # lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(lidarego2global.matmul(lidar2lidarego))
        lidar2img = cam2img.matmul(lidar2cam)

        points_img = points_lidar.tensor[:, :3].matmul(lidar2img[:3, :3].T.to(torch.float)) + lidar2img[:3, 3].to(torch.float).unsqueeze(0)
        points_img = torch.cat([points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]], 1)
        # points_img = points_img.matmul(
        #     post_rots[cid].T) + post_trans[cid:cid + 1, :]
        depth_map = self.points2depthmap(points_img, img_size[1], img_size[0])



        results['gt_depth'] = depth_map
        return results


class StandaloneDepthInputsPipeline:
    """Standalone mini-pipeline equivalent to the config steps:
    - LoadPointsFromFile
    - CustomPointToMultiViewDepth
    - PadMultiViewImageDepth

    This is intended for validation / debugging where we want to run just the
    depth-related transforms on an already-prepared `results` dict.
    """

    def __init__(self, grid_config, downsample=1, size_divisor=32, pad_val=0):
        # Keep this standalone helper free of `mmcv` dependencies.
        # Import here to avoid import-order issues when this module is loaded as a dataset.
        # Points loading params (match config maptrv2_nusc_r50_24ep.py:219-224)
        self._coord_type = 'LIDAR'
        self._load_dim = 5
        self._use_dim = list(range(5))

        self.points_to_depth = _StandaloneCustomPointToMultiViewDepth(
            downsample=downsample,
            grid_config=grid_config,
        )
        self._size_divisor = int(size_divisor)
        self._pad_val = pad_val

    @staticmethod
    def _pad_to_multiple(arr: np.ndarray, divisor: int, pad_val=0) -> np.ndarray:
        """Pad HxW or HxWxC arrays on bottom/right to make H,W divisible by divisor."""
        if arr.ndim == 2:
            h, w = arr.shape
            c = None
        elif arr.ndim == 3:
            h, w, c = arr.shape
        else:
            raise ValueError(f'Unsupported array ndim for padding: {arr.ndim}')

        pad_h = (divisor - (h % divisor)) % divisor
        pad_w = (divisor - (w % divisor)) % divisor
        if pad_h == 0 and pad_w == 0:
            return arr

        if c is None:
            return np.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=pad_val)
        return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=pad_val)

    def _pad_multi_view_image_depth(self, results: dict) -> dict:

        gt_depth = results['gt_depth']
        if isinstance(gt_depth, torch.Tensor):
            gt_depth = self._pad_to_multiple(gt_depth.cpu().numpy(), self._size_divisor, pad_val=self._pad_val)
        else:
            depth_arr = np.asarray(gt_depth)
            gt_depth = self._pad_to_multiple(depth_arr, self._size_divisor, pad_val=self._pad_val)
        results['gt_depth'] = gt_depth
        return results

    def _load_points_from_file(self, results: dict) -> dict:
        # Equivalent of mmdet3d.datasets.pipelines.LoadPointsFromFile for local disk.
        pts_filename = results.get('pts_filename', None) or results.get('lidar_path', None)
        if pts_filename is None:
            raise KeyError('Expected `pts_filename` (or `lidar_path`) in results for point loading.')

        # Temporary code
        # pts = np.fromfile(pts_filename, dtype=np.float32)
        try:
            pts = np.fromfile(pts_filename, dtype=np.float32)
        except:
            if ('etri' in pts_filename):
                pts_filename = pts_filename.replace('etri', 'dooseop')
            else:
                pts_filename = pts_filename.replace('dooseop', 'etri')
            pts = np.fromfile(pts_filename, dtype=np.float32)

        # nuScenes lidar is typically stored as float32 with 4 dims (x, y, z, intensity),
        # but some pipelines expect 5 dims. Auto-detect the per-point dimension if needed.
        load_dim = int(self._load_dim)
        if pts.size % load_dim != 0:
            for candidate_dim in (4, 5, 6, 7, 8):
                if pts.size % candidate_dim == 0:
                    load_dim = candidate_dim
                    break
            else:
                raise ValueError(
                    f'Unsupported lidar binary size: {pts.size} float32 values in {pts_filename!r}; '
                    f'cannot reshape with load_dim={self._load_dim} and no common candidate matches.'
                )

        use_dim = self._use_dim
        if max(use_dim) >= load_dim:
            # Be permissive: keep the available dims (we only need xyz for depth projection).
            use_dim = list(range(load_dim))

        
        pts = pts.reshape(-1, load_dim)[:, use_dim]
        results['points'] = _StandalonePoints(torch.from_numpy(pts).to(torch.float32))
        return results

    def __call__(self, results: dict) -> dict:
        results = self._load_points_from_file(results)
        results = self.points_to_depth(results)
        results = self._pad_multi_view_image_depth(results)
        return results
