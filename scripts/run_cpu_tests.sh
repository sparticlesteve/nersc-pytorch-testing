#!/bin/bash -e
#SBATCH -J cpu-tests
#SBATCH -C haswell
#SBATCH -N 2
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

# PyTorch summary dump
srun -N 1 -n 1 -u python utils/pytorch_info.py

# CPU unit tests
./scripts/run_cpu_unit_tests.sh

# DDP MPI training test
./scripts/run_ddp_test.sh --backend mpi --init-method slurm

# PyTorch Geometric training test
./scripts/run_gcn_test.sh

# MPI4Py test
./scripts/run_mpi4py_test.sh

# Cray plugin training test - not working
#exampleScript=/opt/cray/pe/craype-dl-plugin-py3/19.06.1/examples/torch_mnist/pytorch_mnist.py
#srun -u -l python $exampleScript --epochs 1 --log-interval 50
