#!/bin/bash
#SBATCH --job-name=grassRMA_grace
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=350G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j.log

export OMP_NUM_THREADS=48

module load GCCcore/12.2.0
module load Python/3.10.8
source /scratch/user/dhaura/benchmarks/SpKNN/mt-venv/bin/activate

DATA=/scratch/user/dhaura/repos/minimal_hnsw/sparse/data/msmarco_full
OUT=/scratch/user/dhaura/datasets/SpKNN/grassRMA
mkdir -p $OUT/indices

python3 grassRMA_ex.py -n 8841823 -d 16 -num_threads 48 \
    -input $DATA/base_full.csr \
    -query $DATA/queries.dev.csr \
    -gt $DATA/base_full.dev.gt \
    -output $OUT/indices/base_full \
    -csv $OUT/grassRMA_results_grace.csv
