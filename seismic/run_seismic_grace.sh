#!/bin/bash
#SBATCH --job-name=seismic
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=350G
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=12:00:00

set -euo pipefail

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

source /scratch/user/dhaura/repos/spknn-playground/common/bench_env.sh
bench_provenance
source "$SPKNN_VENV/bin/activate"

OUT=/scratch/user/dhaura/datasets/SpKNN/seismic
mkdir -p "$OUT/indices" "$OUT/bin"

$BENCH_LAUNCH stdbuf -oL -eL python3 seismic_mt_ex.py \
    -n_postings "${NPOST:-3500}" -summary_energy "${SENERGY:-0.4}" \
    -repeats "${REPEATS:-5}" -warmup "${WARMUP:-1}" \
    -input   "$SPKNN_DATA/base_full.csr" \
    -query   "$SPKNN_DATA/queries.dev.csr" \
    -gt      "$SPKNN_DATA/base_full.dev.gt" \
    -bin_dir "$OUT/bin" \
    -index   "$OUT/indices/msmarco_full_np${NPOST:-3500}.index" \
    -csv     "$OUT/seismic_results.csv"
