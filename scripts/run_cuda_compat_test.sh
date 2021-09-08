#!/bin/bash -e
#SBATCH -J compat-test
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH -t 5
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-task 1
#SBATCH -o logs/%x-%j.out

# For this test for now you have to check that the printed CUDA version
# is what you expect from the module load or shifter image.

echo "---------------------------------------------------------------"
date
echo "CUDA compat test"
echo "Cluster: $SLURM_CLUSTER_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Image: $SLURM_SPANK_SHIFTER_IMAGEREQUEST"
module list

[ -z $SLURM_SPANK_SHIFTER_IMAGE ] || SHIFTER=shifter
srun -u -l ${SHIFTER} bash -c 'nvidia-smi | grep CUDA'
#srun -u -l ${SHIFTER} echo `hostname` cuda_compat_status $_CUDA_COMPAT_STATUS
