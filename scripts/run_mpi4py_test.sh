#!/bin/bash -e
#SBATCH -J mpi4py-test
#SBATCH -C gpu
#SBATCH -t 30
#SBATCH -N 1 -n 1 -G 1 -c 10
#SBATCH -o logs/%x-%j.out

echo "---------------------------------------------------------------"
date
echo "MPI4Py test"
echo "Cluster: $SLURM_CLUSTER_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Extra args: $@"
module list

set -x
srun -l -u python -m mpi4py.bench helloworld $@
