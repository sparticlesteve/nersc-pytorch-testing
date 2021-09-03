#!/bin/bash -e
#SBATCH -J gcn-test
#SBATCH -C gpu
#SBATCH -t 30
#SBATCH -N 1 -n 1 -G 1 -c 10
#SBATCH -o logs/%x-%j.out

# To run with NCCL debug, submit like this:
#   NCCL_DEBUG=INFO sbatch ...

echo "---------------------------------------------------------------"
date
echo "PyTorch Geometric GCN test"
echo "Cluster: $SLURM_CLUSTER_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Extra args: $@"
module list

set -x
cd integration-tests
srun -N 1 -n 1 -u python test_gcn.py $@
