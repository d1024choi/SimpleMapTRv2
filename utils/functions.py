# from utils.libraries import *
import json
import os
import glob
import sys
import numpy as np
import shutil
import pickle
from pathlib import Path
import cv2
import time
from tqdm import tqdm
import logging
import traceback
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from collections.abc import Sequence

# ANSI color codes for terminal output
ANSI_COLORS = {
    'CYAN': "\033[96m", 'GREEN': "\033[92m", 'YELLOW': "\033[93m",
    'MAGENTA': "\033[95m", 'RED': "\033[91m", 'BLUE': "\033[94m",
    'BOLD': "\033[1m", 'DIM': "\033[2m", 'RESET': "\033[0m"
}


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_config(path=None):

    if (path is None):
        cfg = read_json(path='./config/config.json')
        cfg.update(read_json(path=f'./config/data.json'))
        cfg.update(read_json(path=f'./config/model.json'))
        cfg.update(read_json(path=f'./config/loss.json'))

    else:
        file_path = os.path.join(path, 'config.json')
        cfg = read_json(path=file_path)

        file_path = os.path.join(path, f'data.json')
        cfg.update(read_json(path=file_path))

        file_path = os.path.join(path, f'model.json')
        cfg.update(read_json(path=file_path))

        file_path = os.path.join(path, f'loss.json')
        cfg.update(read_json(path=file_path))

    return cfg

def config_update(cfg, args):
    '''
    Merge all args into cfg. Access via cfg['args']['param_name'] or legacy paths.
    '''
    # Store all args as dictionary
    cfg['args'] = vars(args).copy()

    # Legacy mappings for backward compatibility
    cfg['app_mode'] = getattr(args, 'app_mode', 'BEV')
    cfg['model_name'] = getattr(args, 'model_name', 'BEVFormer')
    cfg['ddp'] = bool(getattr(args, 'ddp', 0))

    cfg['train']['bool_mixed_precision'] = bool(getattr(args, 'bool_mixed_precision', 0))
    cfg['nuscenes']['image']['bool_apply_img_aug_photo'] = bool(getattr(args, 'bool_apply_img_aug_photo', 0))

    cfg[cfg['model_name']]['aux_tasks']['depth'] = bool(getattr(args, 'bool_depth_aux', 0))
    cfg[cfg['model_name']]['aux_tasks']['pv_seg'] = bool(getattr(args, 'bool_pvseg_aux', 0))
    cfg[cfg['model_name']]['aux_tasks']['bev_seg'] = bool(getattr(args, 'bool_bevseg_aux', 0))
    cfg[cfg['model_name']]['aux_tasks']['one2many']['flag'] = bool(getattr(args, 'bool_one2many', 0))

    # Derived values
    past_horizon_seconds = getattr(args, 'past_horizon_seconds', 0.5)
    future_horizon_seconds = getattr(args, 'future_horizon_seconds', 0.0)
    target_sample_period = getattr(args, 'target_sample_period', 2)
    cfg['obs_len'] = int(past_horizon_seconds * target_sample_period)
    cfg['pred_len'] = int(future_horizon_seconds * target_sample_period)
    cfg['target_frame_indices'] = list(range(cfg['obs_len'] + cfg['pred_len']))
    cfg['cur_frame_index'] = cfg['obs_len'] - 1

    # Path update
    cfg['nuscenes']['dataset_dir'] = check_dataset_path_existence(cfg['nuscenes']['dataset_dir'])
    cfg['nuscenes']['OnlineHDmap']['train_dir'] = check_dataset_path_existence(cfg['nuscenes']['OnlineHDmap']['train_dir'])
    cfg['nuscenes']['OnlineHDmap']['valid_dir'] = check_dataset_path_existence(cfg['nuscenes']['OnlineHDmap']['valid_dir'])
    cfg['nuscenes']['OnlineHDmap']['map_expansion_dir'] = check_dataset_path_existence(cfg['nuscenes']['OnlineHDmap']['map_expansion_dir'])

    cfg['nuscenes']['OnlineHDmapV2']['train_dir'] = check_dataset_path_existence(cfg['nuscenes']['OnlineHDmapV2']['train_dir'])
    cfg['nuscenes']['OnlineHDmapV2']['valid_dir'] = check_dataset_path_existence(cfg['nuscenes']['OnlineHDmapV2']['valid_dir'])
    cfg['nuscenes']['OnlineHDmapV2']['map_expansion_dir'] = check_dataset_path_existence(cfg['nuscenes']['OnlineHDmapV2']['map_expansion_dir'])    

    return cfg

def toNP(x):
    return x.detach().to('cpu').numpy()

def toTS(x, dtype):
    return torch.from_numpy(x).to(dtype)

def check_dataset_path_existence(path, candidates=['etri', 'dooseop']):
    """Check if path exists, trying different user directories."""
    cur_id = next((c for c in candidates if c in path), None)
    if cur_id is None:
        sys.exit(f" [Error] {path} doesn't exist!")
    for candi in candidates:
        new_path = path.replace(cur_id, candi)
        if os.path.exists(new_path):
            return new_path
    sys.exit(f" [Error] {path} doesn't exist!")

def get_dtypes(useGPU=True):
    return torch.LongTensor, torch.FloatTensor

# def init_weights(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         nn.init.kaiming_normal_(m.weight)

def save_read_latest_checkpoint_num(path, val, isSave):

    file_name = path + '/checkpoint.txt'
    index = 0

    if (isSave):
        file = open(file_name, "w")
        file.write(str(int(val)) + '\n')
        file.close()
    else:
        if (os.path.exists(file_name) == False):
            print('[Error] there is no such file in the directory')
            sys.exit()
        else:
            f = open(file_name, 'r')
            line = f.readline()
            index = int(line[:line.find('\n')])
            f.close()

    return index

def read_all_saved_param_idx(path):
    ckp_idx_list = []
    files = sorted(glob.glob(os.path.join(path, '*.pt')))
    for i, file_name in enumerate(files):
        start_idx = 0
        for j in range(-3, -10, -1):
            if (file_name[j] == '_'):
                start_idx = j+1
                break
        ckp_idx = int(file_name[start_idx:-3])
        ckp_idx_list.append(ckp_idx)
    return ckp_idx_list[::-1]

def copy_chkpt_every_N_epoch(args):

    def get_file_number(fname):

        # read checkpoint index
        for i in range(len(fname) - 3, 0, -1):
            if (fname[i] != '_'):
                continue
            index = int(fname[i + 1:len(fname) - 3])
            return index

    root_path = args.model_dir + str(args.exp_id)
    save_directory =  root_path + '/copies'
    if save_directory != '' and not os.path.exists(save_directory):
        os.makedirs(save_directory)

    fname_list = []
    fnum_list = []
    all_file_names = os.listdir(root_path)
    for fname in all_file_names:
        if "saved" in fname:
            chk_index = get_file_number(fname)
            fname_list.append(fname)
            fnum_list.append(chk_index)

    max_idx = np.argmax(np.array(fnum_list))
    target_file = fname_list[max_idx]

    src = root_path + '/' + target_file
    dst = save_directory + '/' + target_file
    shutil.copy2(src, dst)

    print(">> {%s} is copied to {%s}" % (target_file, save_directory))

def remove_past_checkpoint(path, max_num=5):

    def get_file_number(fname):

        # read checkpoint index
        for i in range(len(fname) - 3, 0, -1):
            if (fname[i] != '_'):
                continue
            index = int(fname[i + 1:len(fname) - 3])
            return index


    num_remain = max_num - 1
    fname_list = []
    fnum_list = []

    all_file_names = os.listdir(path)
    for fname in all_file_names:
        if "saved" in fname:
            chk_index = get_file_number(fname)
            fname_list.append(fname)
            fnum_list.append(chk_index)

    if (len(fname_list)>num_remain):
        sort_results = np.argsort(np.array(fnum_list))
        for i in range(len(fname_list)-num_remain):
            del_file_name = fname_list[sort_results[i]]
            os.remove('./' + path + '/' + del_file_name)

