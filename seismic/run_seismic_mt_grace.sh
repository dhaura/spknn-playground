#!/bin/bash
#SBATCH --job-name=seismic_grace
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
OUT=/scratch/user/dhaura/datasets/SpKNN/seismic
mkdir -p $OUT/indices $OUT/bin

for pair in "base_full.csr base_full.bin" "queries.dev.csr queries.dev.bin"; do
    set -- $pair
    if [ ! -f $OUT/bin/$2 ]; then
        echo "Converting $1 -> $2..."
        python3 ../utilities/convert_csr_to_bin.py -file_path $DATA/$1 -output_path $OUT/bin/$2
    fi
done

python3 seismic_mt_ex.py \
    -input $OUT/bin/base_full.bin \
    -query $OUT/bin/queries.dev.bin \
    -gt $DATA/base_full.dev.gt \
    -index $OUT/indices/msmarco_full_np3500.index \
    -num_threads $THREADS \
    -csv $OUT/seismic_mt_results_grace.csv
