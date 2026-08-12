#!/bin/bash
#SBATCH --job-name=grassRMA
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
cd "$PG/grassRMA"

module load python/3.11-24.1.0

source "$PG/common/bench_env_perlmutter.sh"
bench_provenance
source "$SPKNN_VENV/bin/activate"

OUT=${SPKNN_OUT:-$SPKNN_OUT_ROOT/grassRMA}
mkdir -p "$OUT"
rm -f "$OUT/grassRMA_results.csv"

$BENCH_LAUNCH stdbuf -oL -eL python3 grassRMA_ex.py \
    -n "$SPKNN_NDOCS" \
    -M "${M:-32}" -ef_construction "${EFC:-200}" \
    -ef_list "${EF_LIST:-10,20,50,100,200,400,800,1600,3200}" \
    -repeats "${REPEATS:-5}" -warmup "${WARMUP:-1}" \
    -input  "$SPKNN_BASE" \
    -query  "$SPKNN_QUERIES" \
    -gt     "$SPKNN_GT" \
    -csv    "$OUT/grassRMA_results.csv"
