#!/bin/bash -e
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH -t 10
#SBATCH -o slurm-gpu-shifter-test-%j.out

export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=hsn0
#SID=e72123e7afd40f5cba8c2279ce1c1031fb05dd5bd6a2a2eea0d875b599a4c646 # nersc/pytorch:ngc-20.12-v0
#SID=0bf23154fb052b359cceee9706b9a48c61b0d9bca289405f6a13b6e279d9ba6e # nersc/pytorch:ngc-21.02-v0
SID=a5986639e4cf01eb35c0c0a9ca9fb9c6f905cc1b546966b78de4f69d15b894cf # nvcr.io/nvidia/pytorch:21.05-py3
cd /home/sfarrell/pytorch-testing
srun -u -l shifter --image=id:${SID} --module gpu python test_ddp.py \
    --ranks-per-node 4 --gpu --backend nccl --init-method slurm
