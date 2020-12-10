"""
Utility functions for pytorch distributed tests on NERSC systems.
"""

import os

import torch
import torch.distributed as dist

def _get_sync_file():
    """Logic for naming sync file using nersc + slurm env variables"""
    sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH']
    os.makedirs(sync_file_dir, exist_ok=True)
    sync_file = 'file://%s/pytorch_sync.%s.%s' % (
        sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
    return sync_file

def init_workers(backend, init_method, port='29500'):
    """
    Args:
      - backend: can be 'mpi', 'nccl', or 'gloo'
      - init_method: None, 'slurm', 'file'
    """

    init_args = dict(backend=backend)

    if init_method == 'slurm':
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']
        os.environ['MASTER_PORT'] = port
        init_args.update(rank=rank, world_size=world_size)

    elif init_method == 'file':
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        sync_file = _get_sync_file()
        init_args.update(rank=rank, world_size=world_size, init_method=sync_file)

    # Initialize the distributed backend
    dist.init_process_group(**init_args)
