#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/%j.log

export OMP_NUM_THREADS=64
export PYTHONUNBUFFERED=1

source $SCRATCH/benchmarks/SpKNN/vsag-venv/bin/activate

python3 sindi_ex.py \
    -input $SCRATCH/datasets/SpKNN/grassRMA/base_full.csr \
    -query $SCRATCH/datasets/SpKNN/grassRMA/queries.dev.csr \
    -gt $SCRATCH/datasets/SpKNN/grassRMA/base_full.dev.gt \
    -output $SCRATCH/datasets/SpKNN/sindi/indices/base_full.index \
    -csv $SCRATCH/datasets/SpKNN/sindi/sindi_full.csv
