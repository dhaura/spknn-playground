#!/bin/bash
#
# Configuration matches the Grace run so the two are comparable:
#   graph methods   M=32, efC=200, ef in {10,20,50,100,200,400,800,1600,3200}
#   SparseHNSW      + alpha=0.8, beta=3
#   SEISMIC         n_postings=3500, summary_energy=0.4, 7-point sweep
#   PyANNS          ef=80, 12 budgets
#   SINDI           8 doc_prune x 6 query_prune x 3 n_cand = 144 points
#

set -euo pipefail
PG=${SPKNN_PLAYGROUND_ROOT:-/global/homes/d/dhaura/repos/SpKNN/spknn-playground}
cd "$PG"
PG=$PWD
HNSW_SCRIPTS=$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts

DEP=""
if [ -n "${AFTER_JOB:-}" ]; then
    DEP="--dependency=afterany:${AFTER_JOB}"
    echo "chaining: every job waits for ${AFTER_JOB} (afterany)"
fi

submit() {   # submit <dir> <script>
    local dir=$1 script=$2 jid
    mkdir -p "$dir/logs"
    if [ -n "${DRY_RUN:-}" ]; then
        echo "  DRY_RUN  (cd $dir && sbatch $DEP $script)" >&2
        echo "dry"; return
    fi
    jid=$(cd "$dir" && sbatch --parsable $DEP "$script")
    printf '  %-28s job %s\n' "$script" "$jid" >&2
    echo "$jid"
}

echo "=== submitting method jobs (64 threads each) ==="
IDS=()
IDS+=("$(submit "$HNSW_SCRIPTS" run_hnsw_sweep_perlmutter.sh)")
IDS+=("$(submit "$HNSW_SCRIPTS" run_sindi_sweep_perlmutter.sh)")
IDS+=("$(submit "$PG/grassRMA"  run_grassRMA_perlmutter.sh)")
IDS+=("$(submit "$PG/kannolo"   run_kannolo_perlmutter.sh)")
IDS+=("$(submit "$PG/seismic"   run_seismic_perlmutter.sh)")
IDS+=("$(submit "$PG/pyanns"    run_pyanns_perlmutter.sh)")

if [ -n "${DRY_RUN:-}" ]; then echo "(dry run: figures job not submitted)"; exit 0; fi

ALL=$(IFS=:; echo "${IDS[*]}")
FIG=$(cd "$PG" && sbatch --parsable --dependency=afterany:"$ALL" common/make_figures_perlmutter.sh)
echo "  make_figures_perlmutter.sh  job $FIG  (waits for ${#IDS[@]} jobs)"

echo
echo "=== submitted ==="
echo "methods : ${IDS[*]}"
echo "figures : $FIG"
echo "watch   : squeue -u \$USER"
echo "output  : results/msmarco_full_perlmutter/{all_points.csv,pareto.csv,figures,figures_zoom}"
