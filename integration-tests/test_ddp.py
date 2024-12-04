"""
Test of PyTorch DistributedDataParallel for GPU installations
"""

import os
import socket
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torchvision

from distributed import init_workers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50',
                        choices=['simple', 'resnet50'])
    parser.add_argument('--ranks-per-node', type=int)
    parser.add_argument('--backend', default='nccl',
                        choices=['mpi', 'nccl', 'gloo'])
    parser.add_argument('--init-method', default='env',
                        choices=['env', 'slurm', 'file'])
    parser.add_argument('--gpu', action='store_true', help='Use GPUs')
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize distributed library
    init_workers(args.backend, args.init_method)
    rank, n_ranks = dist.get_rank(), dist.get_world_size()
    # Backwards compatible ranks-per-node CL argument
    if args.ranks_per_node is not None:
        local_rank = rank % args.ranks_per_node
    else:
        local_rank = int(os.environ['LOCAL_RANK'])
    print('Initialized host', socket.gethostname(),
          'rank', rank, 'local-rank', local_rank, 'size', n_ranks)

    if args.gpu:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Random number dataset
    print('Generating a batch of data')
    batch_size = 32
    sample_shape = [3, 224, 224]
    n_classes = 32
    x = torch.randn([batch_size] + sample_shape).to(device)
    y = torch.randint(n_classes, (batch_size,)).to(device)

    # Construct a simple model
    print('Constructing model')

    # ResNet50
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(num_classes=n_classes).to(device)

    # Simple CNN
    else:
        hidden_size = 256
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, hidden_size, kernel_size=3),
            torch.nn.Conv2d(hidden_size, n_classes, kernel_size=3),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten()).to(device)

    # Wrap model for distributed training
    device_ids = [device] if args.gpu else None
    model = DistributedDataParallel(model, device_ids=device_ids)

    #if rank == 0:
    #    print(model)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training
    print('Performing one training step')
    batch_output = model(x)
    loss = loss_fn(batch_output, y)
    loss.backward()
    optimizer.step()

    # Testing a barrier at the end
    torch.distributed.barrier()

    print('Finished')

if __name__ == '__main__':
    main()
