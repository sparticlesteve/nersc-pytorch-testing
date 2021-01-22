#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 10
#SBATCH --time 30
#SBATCH -o nccl-test-%j.out

# Software
module load cgpu pytorch/1.7.1-gpu
module list

# NCCL debug output
export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

# Dump some information
echo "DDP NCCL training test"
srun -N 1 -n 1 python pytorch_info.py

# Run a distributed training test
srun -u -l python test_ddp.py --gpu --backend nccl --init-method slurm

echo "All done!"
