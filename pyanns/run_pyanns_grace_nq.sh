#!/bin/bash
#SBATCH --job-name=pyanns_nq
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=350G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#
# PyANNS baseline on Grace for nq_splade.  Six graphs (R:L) as in the
# nq_splade Perlmutter run; each is built once then swept over ef x budget.
set -euo pipefail
PG=/scratch/user/dhaura/repos/spknn-playground
cd "$PG/pyanns"; mkdir -p logs
module purge; module load GCCcore/13.2.0; module load Python/3.11.5
export SPKNN_DATASET=${SPKNN_DATASET:-nq_splade}
source "$PG/common/bench_env.sh"; source "$PG/common/dataset_env_grace.sh"
bench_provenance; source "$SPKNN_VENV/bin/activate"
OUT=${SPKNN_OUT:-$SPKNN_OUT_ROOT/pyanns}; mkdir -p "$OUT/indices"
for RL in ${GRAPHS:-32:200 32:400 32:800 32:1600 48:800 64:800}; do
    R=${RL%%:*}; L=${RL##*:}
    CSV="$OUT/pyanns_results_${SPKNN_DATASET}_R${R}_L${L}_${SLURM_JOB_ID}.csv"
    echo "########## R=$R L=$L -> $CSV ##########"
    $BENCH_LAUNCH stdbuf -oL -eL python3 pyanns_ex.py \
        -n "$SPKNN_NDOCS" -R "$R" -L "$L" \
        -ef_list "${EF_LIST:-80,150,300,600,1200,2400,4800}" \
        -budgets "${BUDGETS:-0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.5,0.8,0.9,1.0}" \
        -repeats "${REPEATS:-3}" -warmup "${WARMUP:-1}" \
        -input "$SPKNN_BASE" -query "$SPKNN_QUERIES" -gt "$SPKNN_GT" \
        -index "$OUT/indices/${SPKNN_DATASET}_R${R}_L${L}" -csv "$CSV"
done
echo "########## done -> $OUT ##########"
