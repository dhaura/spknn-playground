#!/bin/bash
#SBATCH --job-name=pyanns
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

OUT=/scratch/user/dhaura/datasets/SpKNN/pyanns
mkdir -p "$OUT/indices"

$BENCH_LAUNCH stdbuf -oL -eL python3 pyanns_ex.py \
    -n 8841823 -ef "${EF:-80}" \
    -budgets "${BUDGETS:-0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.5,0.8}" \
    -repeats "${REPEATS:-5}" -warmup "${WARMUP:-1}" \
    -input  "$SPKNN_DATA/base_full.csr" \
    -query  "$SPKNN_DATA/queries.dev.csr" \
    -gt     "$SPKNN_DATA/base_full.dev.gt" \
    -index  "$OUT/indices/base_full" \
    -csv    "$OUT/pyanns_results.csv"
