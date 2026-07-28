#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/%j.log

THREADS="${1:-64}"

export PYTHONUNBUFFERED=1

source $SCRATCH/benchmarks/SpKNN/mt-venv/bin/activate

DATA=$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_full
OUT=$SCRATCH/datasets/SpKNN/seismic
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
    -csv $OUT/seismic_mt_results.csv
