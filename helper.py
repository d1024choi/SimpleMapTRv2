import sys
import os
import importlib
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.functions import read_config



def load_datasetloader(args, dtype, world_size, rank, mode='train'):
    '''
    Load dataset and create PyTorch DataLoader with optional DDP support.

    Args:
        args: Argument parser containing dataset/model configuration.
        dtype: PyTorch tensor dtype (e.g., torch.FloatTensor).
        world_size: Number of processes for distributed training.
        rank: Process rank for distributed training.
        mode: 'train', 'val', 'valid', or 'test'.

    Returns:
        For train/val mode:
            (train_dataset, val_dataset), (train_loader, val_loader), (train_sampler, val_sampler)
        For test mode:
            dataset, loader, None
    '''


    cfg = read_config()
    
    # Current setting verification
    _src = os.path.basename(__file__)
    if args.app_mode not in cfg['supported_app_modes']:
        sys.exit(f'[{_src}] The mode {args.app_mode} is not supported!')

    if args.dataset_type not in list(cfg['supported_datasets'].keys()):
        sys.exit(f'[{_src}] The dataset {args.dataset_type} is not supported!')
    
    if args.model_name not in cfg['supported_models']:
        sys.exit(f'[{_src}] The model {args.model_name} is not supported!')
    
    # Load dataset loader and collate function
    collate_fn_path = cfg[args.model_name].get('collate_fn')
    if collate_fn_path:
        module_path, fn_name = collate_fn_path.rsplit('.', 1)
        collate_fn = getattr(importlib.import_module(module_path), fn_name)
    else:
        collate_fn = None
    DatasetLoader = importlib.import_module(cfg[args.model_name]['loader_path']).DatasetLoader
    
    # Test mode
    if mode not in ['train', 'val', 'valid']:
        dataset = DatasetLoader(args=args, dtype=dtype, world_size=1, rank=0, mode='test')
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_cores, drop_last=False, collate_fn=collate_fn)
        return dataset, loader, None

    # Train/Val mode
    train_dataset = DatasetLoader(args=args, dtype=dtype, world_size=world_size if args.ddp else 1,
                                  rank=rank if args.ddp else 0, mode='train')
    # val_dataset = DatasetLoader(args=args, dtype=dtype, world_size=world_size if args.ddp else 1,
    #                             rank=rank if args.ddp else 0, mode='val', nusc=train_dataset.nusc)
    test_dataset = DatasetLoader(args=args, dtype=dtype, world_size=world_size if args.ddp else 1,
                                  rank=rank if args.ddp else 0, mode='test', nusc=train_dataset.nusc)    

    if args.ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_cores, pin_memory=True,
                                  sampler=train_sampler, collate_fn=collate_fn)
        # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        #                         num_workers=args.num_cores, pin_memory=True,
        #                         sampler=val_sampler, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_cores, pin_memory=True,
                                sampler=test_sampler, collate_fn=collate_fn)
    else:
        # train_sampler, val_sampler = None, None
        train_sampler, test_sampler = None, None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, # Debug -----
                                  num_workers=args.num_cores, drop_last=True, collate_fn=collate_fn)
        # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        #                         num_workers=args.num_cores, drop_last=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_cores, drop_last=False, collate_fn=collate_fn)

    # return (train_dataset, val_dataset), (train_loader, val_loader), (train_sampler, val_sampler)
    return (train_dataset, test_dataset), (train_loader, test_loader), (train_sampler, test_sampler)


def load_solvers(args, num_batches, logger, dtype, world_size=None, rank=None, isTrain=True):
    '''
    Load the appropriate solver based on the model name.

    Args:
        args: Argument parser containing model configuration.
        num_batches: Number of batches.
        logger: Logger object for logging.
        dtype: PyTorch tensor dtype (e.g., torch.FloatTensor).
        world_size: Number of processes for distributed training.
    '''

    cfg = read_config()
    solver_path = cfg[args.model_name]['solver_path']
    if solver_path:
        module_path, fn_name = solver_path.rsplit('.', 1)
        Solver = getattr(importlib.import_module(module_path), fn_name)
    else:
        sys.exit(f'[{_src}] No solver available for {args.model_name}!')
    return Solver(args, num_batches, world_size, rank, logger, dtype, isTrain)
