#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/%j.log

THREADS="${1:-64}"

export PYTHONUNBUFFERED=1

source $SCRATCH/benchmarks/SpKNN/mt-venv/bin/activate

DATA=$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_full
OUT=$SCRATCH/datasets/SpKNN/kannolo
mkdir -p $OUT/indices

python3 kannolo_mt_ex.py \
    -input $DATA/base_full.csr \
    -query $DATA/queries.dev.csr \
    -gt $DATA/base_full.dev.gt \
    -index $OUT/indices/msmarco_full_m32_efc200.index \
    -num_threads $THREADS \
    -csv $OUT/kannolo_mt_results.csv
