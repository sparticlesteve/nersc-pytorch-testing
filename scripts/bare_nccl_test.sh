#!/bin/bash
#SBATCH -C gpu
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 10
#SBATCH --time 10
#SBATCH --image nersc/pytorch:ngc-20.12-v0
#SBATCH -o bare-nccl-%j.out

srun shifter \
    --env CUDA_HOME=/usr/local/cuda-11.1/ \
    --env NCCL_DEBUG=INFO \
    --env NCCL_SOCKET_IFNAME=eth \
    ../nccl-tests/build/all_reduce_perf $@
