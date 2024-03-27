#!/bin/bash -e
#SBATCH -J ddp-test
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -t 10
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH --gpus-per-node 4
#SBATCH -o logs/%x-%j.out

# To run with NCCL debug, submit like this:
#   NCCL_DEBUG=INFO sbatch ...

echo "---------------------------------------------------------------"
date
echo "PyTorch distributed launch DDP test"
echo "Cluster: $SLURM_CLUSTER_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks/node: $SLURM_NTASKS_PER_NODE"
echo "Image: $SLURM_SPANK_SHIFTER_IMAGEREQUEST"
echo "Extra args: $@"
module list

[ -z $SLURM_SPANK_SHIFTER_IMAGE ] || SHIFTER=shifter
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29507

set -x
cd integration-tests
srun ${SHIFTER} torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc-per-node=$SLURM_GPUS_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    test_ddp.py --gpu $@
