#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 10
#SBATCH --time 5
#SBATCH -o nccl-test-%j.out

# Software
module load cgpu pytorch/1.7.1-gpu
module list

set -x

# NCCL debug output
export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=0
#export NCCL_SOCKET_IFNAME=eth,ib

# Dump some information
srun -N 1 -n 1 python pytorch_info.py

# Run a distributed training test
srun -u -l python test_ddp.py --gpu --backend nccl --init-method slurm \
    --ranks-per-node $SLURM_NTASKS_PER_NODE

set +x

echo "All done!"
