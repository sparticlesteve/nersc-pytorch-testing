#!/bin/bash -e
#SBATCH -J mpi4py-test
#SBATCH -C gpu
#SBATCH -t 30
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --module=gpu
#SBATCH -o logs/%x-%j.out

echo "---------------------------------------------------------------"
date
echo "MPI4Py test"
echo "Cluster: $SLURM_CLUSTER_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Image: $SLURM_SPANK_SHIFTER_IMAGEREQUEST"
echo "Extra args: $@"
module list

[ -z $SLURM_SPANK_SHIFTER_IMAGE ] || SHIFTER=shifter
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29507
export MPICH_GPU_SUPPORT_ENABLED=0

set -x
srun -l -u --mpi=pmi2 ${SHIFTER} python -m mpi4py.bench helloworld $@
