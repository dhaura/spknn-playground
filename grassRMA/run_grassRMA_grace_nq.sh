#!/bin/bash
#SBATCH --job-name=grassRMA_nq
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=350G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#
# GrassRMA baseline on Grace for nq_splade.  Grace port of
# run_grassRMA_v2_perlmutter.sh; the grid replicates the nq_splade Perlmutter
# run (M=32, efC 200-1600, 11-point ef sweep) so the two are comparable.
# One graph build per efC (4 builds, ~1h each on Grace).
#
#   cd grassRMA && mkdir -p logs && sbatch run_grassRMA_grace_nq.sh

set -euo pipefail
PG=/scratch/user/dhaura/repos/spknn-playground
cd "$PG/grassRMA"
mkdir -p logs

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

export SPKNN_DATASET=${SPKNN_DATASET:-nq_splade}
source "$PG/common/bench_env.sh"
source "$PG/common/dataset_env_grace.sh"
bench_provenance
source "$SPKNN_VENV/bin/activate"

OUT=${SPKNN_OUT:-$SPKNN_OUT_ROOT/grassRMA}
mkdir -p "$OUT"
EF_LIST=${EF_LIST:-10,20,50,100,200,400,800,1600,3200,6400,12800}
echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET  ndocs=$SPKNN_NDOCS"
echo "### threads : $BENCH_THREADS   launch: $BENCH_LAUNCH"
python3 -c "import sparse_hnswlib; print('### module  :', sparse_hnswlib.__file__)"

for EFC in ${EFCS:-200 400 800 1600}; do
    CSV="$OUT/grassRMA_results_${SPKNN_DATASET}_efc${EFC}_${SLURM_JOB_ID}.csv"
    echo "########## M=${M:-32} efC=$EFC -> $CSV ##########"
    $BENCH_LAUNCH stdbuf -oL -eL python3 grassRMA_ex.py \
        -n "$SPKNN_NDOCS" -M "${M:-32}" -ef_construction "$EFC" \
        -ef_list "$EF_LIST" -repeats "${REPEATS:-3}" -warmup "${WARMUP:-1}" \
        -input "$SPKNN_BASE" -query "$SPKNN_QUERIES" -gt "$SPKNN_GT" -csv "$CSV"
done
echo "########## done -> $OUT ##########"
