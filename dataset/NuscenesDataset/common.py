import numpy as np
import cv2
import PIL.Image as Image
from pathlib import Path
from pyquaternion import Quaternion


STATIC = ['lane', 'road_segment', 'ped_crossing', 'walkway', 'carpark_area', 'stop_line']
DIVIDER = ['road_divider', 'lane_divider']
DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
]

CLASSES = STATIC + DIVIDER + DYNAMIC

MAP_NAMES = ['singapore-onenorth',
             'singapore-hollandvillage',
             'singapore-queenstown',
             'boston-seaport']

CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

DATA_KEYS = ['cam_idx', 'image', 'intrinsics', 'extrinsics', 'bev', 'view',
             'visibility', 'center', 'pose', 'future_egomotion', 'label', 'instance']

INTERPOLATION = cv2.LINE_8

SCENE_BLACKLIST = ['scene-0515', 'scene-0517']

CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'ped_crossing': 1,
        'contours': 2,
        'others': -1
        }

def get_split(split, dataset_name):
    split_dir = Path(__file__).parent / 'nuscenes/splits'
    split_path = split_dir / f'{split}.txt'

    return split_path.read_text().strip().split('\n')


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters

    return np.float32([
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ])


def get_pose(rotation, translation, inv=False, flat=False):
    """Convert quaternion rotation and translation to 4x4 transformation matrix.
    
    Args:
        rotation: Quaternion rotation (w, x, y, z) or dict with 'rotation' key
        translation: Translation vector (x, y, z) or dict with 'translation' key
        inv: If True, compute inverse transformation matrix
        flat: If True, use only yaw angle (for 2D transforms like EL2W)
    
    Returns:
        4x4 transformation matrix (np.float32)
    """
    # Extract rotation matrix
    if flat:
        yaw = Quaternion(rotation).yaw_pitch_roll[0]
        R = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).rotation_matrix
    else:
        R = Quaternion(rotation).rotation_matrix
    t = np.array(translation, dtype=np.float32)
    
    # Build transformation matrix
    pose = np.eye(4, dtype=np.float32)
    if inv:
        pose[:3, :3] = R.T
        pose[:3, -1] = R.T @ -t
    else:
        pose[:3, :3] = R
        pose[:3, -1] = t
    
    return pose


def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords


