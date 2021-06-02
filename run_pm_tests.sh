#!/bin/bash -e
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH -t 10
#SBATCH -o slurm-gpu-test-%j.out

module load gcc/9.3.0
module load pytorch/1.8.0
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

srun -u -l python test_ddp.py --ranks-per-node 4 --gpu --backend nccl --init-method slurm
