#!/bin/bash -e
#SBATCH -J cpu-pytest
#SBATCH -C haswell
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

echo "---------------------------------------------------------------"
date
echo "PyTorch CPU unit tests"
echo "Cluster: $SLURM_CLUSTER_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Image: $SLURM_SPANK_SHIFTER_IMAGEREQUEST"
echo "Extra args: $@"
module list

[ -z $SLURM_SPANK_SHIFTER_IMAGE ] || SHIFTER=shifter

set -x
srun -N 1 -n 1 -u ${SHIFTER} pytest \
    unit-tests/test_common.py unit-tests/test_cpu.py $@
