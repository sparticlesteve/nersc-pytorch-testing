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

def init_workers(backend, init_method, port='29507'):
    """
    Args:
      - backend: can be 'mpi', 'nccl', or 'gloo'
      - init_method: None, 'slurm', 'file'
    """

    init_args = dict(backend=backend)

    if init_method == 'env':
        # Use SLURM variables as backup
        if 'RANK' not in os.environ:
            os.environ['RANK'] = os.environ['SLURM_PROCID']
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
        print('Distributed init with master addr ' +
              f'{os.environ["MASTER_ADDR"]} port {os.environ["MASTER_PORT"]}')

    elif init_method == 'slurm':
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']
        os.environ['MASTER_PORT'] = port
        print(f'Distributed init with master addr {os.environ["MASTER_ADDR"]} port {port}')
        init_args.update(rank=rank, world_size=world_size)

    elif init_method == 'file':
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        sync_file = _get_sync_file()
        print(f'Distributed init with sync file {sync_file}')
        init_args.update(rank=rank, world_size=world_size, init_method=sync_file)

    # Initialize the distributed backend
    dist.init_process_group(**init_args)
