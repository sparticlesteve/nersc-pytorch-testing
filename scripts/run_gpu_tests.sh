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

cd integration-tests

echo "-------------------------------------------------------------------------"
echo "Single GPU unit tests"
srun -N 1 -n 1 -u python pytorch_info.py
srun -N 1 -n 1 -u python test_install.py --cuda --vision --geometric

#echo "-------------------------------------------------------------------------"
#echo "Multi GPU unit tests"
#srun --ntasks-per-node 8 -u -l python test_install.py --mpi --cuda

echo "-------------------------------------------------------------------------"
echo "DDP NCCL training test"
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
srun -u -l python test_ddp.py --gpu --backend nccl --init-method slurm \
    --ranks-per-node=$SLURM_NTASKS_PER_NODE

# Disabling failing MPI test
#echo "-------------------------------------------------------------------------"
#echo "DDP MPI training test"
#srun -u -l python test_ddp.py --backend mpi --gpu \
#    --ranks-per-node=$SLURM_NTASKS_PER_NODE

echo "-------------------------------------------------------------------------"
echo "DDP Gloo training test"
srun -u -l python test_ddp.py --gpu --backend gloo --init-method slurm \
    --ranks-per-node=$SLURM_NTASKS_PER_NODE

echo "-------------------------------------------------------------------------"
echo "PyTorch Geometric training test"
srun -N 1 -n 1 -u python test_gcn.py

echo "-------------------------------------------------------------------------"
echo "MPI4Py test"
srun -l -u python -m mpi4py.bench helloworld
