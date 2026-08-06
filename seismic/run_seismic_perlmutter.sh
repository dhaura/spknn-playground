#!/bin/bash
#SBATCH --job-name=seismic
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 physical cores x 2 hyperthreads -> BENCH_THREADS=64
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=12:00:00

set -euo pipefail
PG=${SPKNN_PLAYGROUND_ROOT:-/global/homes/d/dhaura/repos/SpKNN/spknn-playground}
cd "$PG/seismic"

module load python/3.11-24.1.0

source "$PG/common/bench_env_perlmutter.sh"
bench_provenance
source "$SPKNN_VENV/bin/activate"

OUT=${SPKNN_OUT:-$SCRATCH/datasets/SpKNN/seismic}
mkdir -p "$OUT/indices" "$OUT/bin"
rm -f "$OUT/seismic_results.csv"

for NPOST in ${NPOST_LIST:-2000 3500 6000}; do
  for SENERGY in ${SENERGY_LIST:-0.4 0.5}; do
    echo
    echo "######## seismic index: n_postings=$NPOST summary_energy=$SENERGY ########"
    $BENCH_LAUNCH stdbuf -oL -eL python3 seismic_mt_ex.py \
        -n_postings "$NPOST" -summary_energy "$SENERGY" \
        -repeats "${REPEATS:-5}" -warmup "${WARMUP:-1}" \
        -input   "$SPKNN_DATA/base_full.csr" \
        -query   "$SPKNN_DATA/queries.dev.csr" \
        -gt      "$SPKNN_DATA/base_full.dev.gt" \
        -bin_dir "$OUT/bin" \
        -index   "$OUT/indices/msmarco_full_np${NPOST}_se${SENERGY}.index" \
        -csv     "$OUT/seismic_results.csv"
  done
done
