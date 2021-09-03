# PyTorch tests for NERSC systems

This repo contains unit tests and integration tests for PyTorch at NERSC.

## Examples

Run unit tests on Perlmutter:

```bash
module load pytorch/1.9.0
sbatch scripts/run_gpu_unit_tests.sh
```

Run all CPU tests on Cori in a conda environment:

```
conda activate my_env
sbatch scripts/run_cpu_tests.sh
```

Run the DDP test on 2 full nodes on Cori-GPU with the gloo backend:

```bash
module purge
module load cgpu pytorch/1.8.0-gpu
sbatch -N 2 --ntasks-per-node 8 scripts/run_ddp_test.sh --backend gloo
```
