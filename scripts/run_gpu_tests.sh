#!/bin/bash -e
#SBATCH -J gpu-tests
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --exclusive
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

srun -N 1 -n 1 -u nvidia-smi

# PyTorch summary dump
if [ ! -z $SLURM_SPANK_SHIFTER_IMAGE ]; then
    srun -N 1 -n 1 -u shifter python utils/pytorch_info.py
else
    srun -N 1 -n 1 -u python utils/pytorch_info.py
fi

# GPU unit tests
./scripts/run_gpu_unit_tests.sh

# DDP NCCL training tests
export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
./scripts/run_ddp_test.sh --backend nccl
./scripts/run_ddp_test.sh --backend gloo

# PyTorch Geometric training test
./scripts/run_gcn_test.sh

# MPI4Py test
./scripts/run_mpi4py_test.sh
