#!/bin/bash
#SBATCH --job-name=grassRMA
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

OUT=/scratch/user/dhaura/datasets/SpKNN/grassRMA
mkdir -p "$OUT"

$BENCH_LAUNCH stdbuf -oL -eL python3 grassRMA_ex.py \
    -n 8841823 \
    -M "${M:-16}" -ef_construction "${EFC:-200}" \
    -ef_list "${EF_LIST:-10,20,50,100,200,400,800,1600,3200}" \
    -repeats "${REPEATS:-5}" -warmup "${WARMUP:-1}" \
    -input  "$SPKNN_DATA/base_full.csr" \
    -query  "$SPKNN_DATA/queries.dev.csr" \
    -gt     "$SPKNN_DATA/base_full.dev.gt" \
    -csv    "$OUT/grassRMA_results.csv"
