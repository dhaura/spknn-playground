#!/bin/bash
#SBATCH --job-name=kannolo_grace
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=350G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j.log

THREADS="${1:-48}"

export PYTHONUNBUFFERED=1

module load GCCcore/12.2.0
module load Python/3.10.8
source /scratch/user/dhaura/benchmarks/SpKNN/mt-venv/bin/activate

DATA=/scratch/user/dhaura/repos/minimal_hnsw/sparse/data/msmarco_full
OUT=/scratch/user/dhaura/datasets/SpKNN/kannolo
mkdir -p $OUT/indices

python3 kannolo_mt_ex.py \
    -input $DATA/base_full.csr \
    -query $DATA/queries.dev.csr \
    -gt $DATA/base_full.dev.gt \
    -index $OUT/indices/msmarco_full_m32_efc200.index \
    -num_threads $THREADS \
    -csv $OUT/kannolo_mt_results_grace.csv
