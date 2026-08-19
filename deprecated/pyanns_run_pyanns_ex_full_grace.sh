#!/bin/bash
#SBATCH --job-name=pyanns_grace
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=350G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j.log

module load GCCcore/14.2.0
module load Python/3.13.1

source /scratch/user/dhaura/repos/pyanns/venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=48

DATA=/scratch/user/dhaura/repos/minimal_hnsw/sparse/data/msmarco_full
OUT=/scratch/user/dhaura/datasets/SpKNN/pyanns
mkdir -p $OUT/indices

python3 pyanns_ex.py -n 8841823 -nq 6980 \
    -input $DATA/base_full.csr \
    -query $DATA/queries.dev.csr \
    -gt $DATA/base_full.dev.gt \
    -output $OUT/indices/base_full \
    -csv $OUT/pyanns_results_grace.csv
