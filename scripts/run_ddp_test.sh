#!/bin/bash -e
#SBATCH -J ddp-test
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -t 30
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 10
#SBATCH --gpus-per-task 1
#SBATCH -o logs/%x-%j.out

# To run with NCCL debug, submit like this:
#   NCCL_DEBUG=INFO sbatch ...

echo "---------------------------------------------------------------"
date
echo "PyTorch DDP test"
echo "Cluster: $SLURM_CLUSTER_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks/node: $SLURM_NTASKS_PER_NODE"
echo "Extra args: $@"
module list

set -x
cd integration-tests
srun -u -l python test_ddp.py --ranks-per-node $SLURM_NTASKS_PER_NODE --gpu $@
