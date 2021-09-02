#!/bin/bash -e
#SBATCH -J pytest
#SBATCH -C gpu
#SBATCH -N 1 -n 1 -c 4 -G 1
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

echo "---------------------------------------------------------------"
date
echo "PyTorch unit tests"
echo "Cluster: $SLURM_CLUSTER_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Extra args: $@"
module list
echo ""

set -x
srun -N 1 -n 1 -u pytest unit-tests/test_common.py unit-tests/test_gpu.py $@
