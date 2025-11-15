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
echo "Extra args: $@"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29507
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

# Container command
CONT_CMD=(
    podman-hpc run
    --rm --gpu --nccl
    --net=host --ipc=host #--pid=host --privileged
    --env SLURM_* --env MASTER_* --env NCCL_DEBUG --env OMP*
    -v .:/workspace
    nersc/pytorch:25.06.01
)

# Launch command
LAUNCH_CMD=(
    torchrun
    --nnodes=$SLURM_JOB_NUM_NODES
    --nproc-per-node=$SLURM_GPUS_PER_NODE
    --rdzv-backend=c10d
    #--rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT
    test_ddp.py --gpu
    $@
)

set -x
cd integration-tests
srun ${CONT_CMD[@]} ${LAUNCH_CMD[@]}
