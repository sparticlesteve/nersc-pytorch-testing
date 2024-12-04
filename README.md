# PyTorch tests for NERSC systems

This repo contains unit tests and integration tests for PyTorch at NERSC.

## Examples

Run unit tests on Perlmutter:

```bash
module load pytorch/2.3.1
sbatch -A $youraccount scripts/run_gpu_unit_tests.sh
```

Run all CPU tests on Cori in a conda environment:

```bash
conda activate my_env
sbatch scripts/run_cpu_tests.sh
```

Run the DDP test on 2 full nodes on Perlmutter with the NCCL backend:

```bash
module load pytorch/2.3.1
sbatch -A $youraccount -N 2 --ntasks-per-node 4 scripts/run_ddp_test.sh --backend nccl --init-method file --ranks-per-node 4
```
