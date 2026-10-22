#!/bin/bash
#SBATCH --job-name=kannolo_nq
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=350G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#
# kANNolo baseline on Grace for nq_splade.  Grid replicates the nq_splade
# Perlmutter run: m=32, efC 200/400/800/1600 (one index build each),
# ef 10..25600, early-exit 0.005..0.25.
set -euo pipefail
PG=/scratch/user/dhaura/repos/spknn-playground
cd "$PG/kannolo"; mkdir -p logs
module purge; module load GCCcore/13.2.0; module load Python/3.11.5
export SPKNN_DATASET=${SPKNN_DATASET:-nq_splade}
source "$PG/common/bench_env.sh"; source "$PG/common/dataset_env_grace.sh"
bench_provenance; source "$SPKNN_VENV/bin/activate"
OUT=${SPKNN_OUT:-$SPKNN_OUT_ROOT/kannolo}; mkdir -p "$OUT/indices"
for EFC in ${EFCS:-200 400 800 1600}; do
    CSV="$OUT/kannolo_results_${SPKNN_DATASET}_efc${EFC}_${SLURM_JOB_ID}.csv"
    echo "########## m=${M:-32} efC=$EFC -> $CSV ##########"
    $BENCH_LAUNCH stdbuf -oL -eL python3 kannolo_mt_ex.py \
        -m "${M:-32}" -ef_construction "$EFC" \
        -ef_list "${EF_LIST:-10,20,50,100,200,400,800,1600,3200,6400,12800,25600}" \
        -early_exit_list "${EET_LIST:-none,0.005,0.01,0.05,0.1,0.25}" \
        -repeats "${REPEATS:-3}" -warmup "${WARMUP:-1}" \
        -input "$SPKNN_BASE" -query "$SPKNN_QUERIES" -gt "$SPKNN_GT" \
        -index "$OUT/indices/${SPKNN_DATASET}_m${M:-32}_efc${EFC}.index" -csv "$CSV"
done
echo "########## done -> $OUT ##########"
