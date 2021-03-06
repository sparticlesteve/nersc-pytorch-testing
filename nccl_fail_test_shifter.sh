#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 10
#SBATCH --time 5
#SBATCH --image nersc/pytorch:ngc-20.12-v0
#SBATCH -o nccl-test-shifter-%j.out

set -x

# NCCL debug output
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_4:1,mlx5_6:1

# Dump some information
srun -N 1 -n 1 shifter python pytorch_info.py

# Run a distributed training test
srun -u -l shifter python test_ddp.py --gpu --backend nccl --init-method slurm \
    --ranks-per-node $SLURM_NTASKS_PER_NODE

set +x

echo "All done!"
