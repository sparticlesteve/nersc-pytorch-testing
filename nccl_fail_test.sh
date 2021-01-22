#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 10
#SBATCH --time 30
#SBATCH -o nccl-test-%j.out

module load cgpu pytorch/1.7.1-gpu
module list

export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

echo "DDP NCCL training test"

srun -u -l python test_ddp.py --gpu --backend nccl --init-method slurm

echo "All done!"
