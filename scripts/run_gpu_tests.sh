#!/bin/bash -e
#SBATCH -J gpu-tests
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --exclusive
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

# PyTorch summary dump
srun -N 1 -n 1 -u python utils/pytorch_info.py

# GPU unit tests
./scripts/run_gpu_unit_tests.sh

# DDP NCCL training tests
export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
./scripts/run_ddp_test.sh --backend nccl --init-method slurm
./scripts/run_ddp_test.sh --backend gloo --init-method slurm

# PyTorch Geometric training test
./scripts/run_gcn_test.sh

# MPI4Py test
./scripts/run_mpi4py_test.sh
