#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/%j.log

export OMP_NUM_THREADS=64

source $SCRATCH/benchmarks/SpKNN/GrassRMA/venv/bin/activate

python3 grassRMA_ex.py -n 1000000 -d 16 -num_threads 64 -input $SCRATCH/datasets/SpKNN/grassRMA/base_1M.csr -query $SCRATCH/datasets/SpKNN/grassRMA/queries.dev.csr -gt $SCRATCH/datasets/SpKNN/grassRMA/base_1M.dev.gt -output $SCRATCH/datasets/SpKNN/grassRMA/indices/base_1M
