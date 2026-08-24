#!/bin/bash
#SBATCH --job-name=spknn_figures
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
PG=${SPKNN_PLAYGROUND_ROOT:-/global/homes/d/dhaura/repos/SpKNN/spknn-playground}
cd "$PG"

module load python/3.11-24.1.0 2>/dev/null
source "$PG/common/bench_env_perlmutter.sh"
source "$SPKNN_VENV/bin/activate"

DATA_ROOT=${SPKNN_DATA_ROOT:-$SPKNN_OUT_ROOT}
OUT=${FIG_OUT:-results/${SPKNN_DATASET}_perlmutter}
ZOOM_XMIN=${ZOOM_XMIN:-0.8}

mkdir -p "$OUT"

HNSW_DIR="$DATA_ROOT/sparse_hnsw"
if compgen -G "$HNSW_DIR/sparse_hnsw_results_*.csv" >/dev/null; then
    echo "=== collapsing SparseHNSW alpha/beta configurations to their envelope ==="
    # ENVELOPE_TARGETS thins the front to one point per recall target, so the
    # SparseHNSW series has a comparable marker density to the other methods
    # instead of ~100 points smeared along the curve.
    python3 "$SPKNN_HNSW_REPO/sparse/scripts/combine_hnsw_envelope.py" "$HNSW_DIR" \
        ${ENVELOPE_TARGETS:+--targets "$ENVELOPE_TARGETS"}
    echo
fi

CSVS=()
for pat in "$DATA_ROOT"/sparse_hnsw/sparse_hnsw_results*.csv \
           "$DATA_ROOT"/sindi/sindi_results*.csv \
           "$DATA_ROOT"/grassRMA/grassRMA_results*.csv \
           "$DATA_ROOT"/kannolo/kannolo_results*.csv \
           "$DATA_ROOT"/seismic/seismic_results*.csv \
           "$DATA_ROOT"/pyanns/pyanns_results*.csv; do
    found=0
    for f in $pat; do
        [ -s "$f" ] || continue
        CSVS+=("$f"); found=1
        echo "  found  $f  ($(($(wc -l < "$f") - 1)) rows)"
    done
    [ "$found" -eq 1 ] || echo "  MISSING $pat  (that method will be absent from the figures)" >&2
done

if [ ${#CSVS[@]} -eq 0 ]; then
    echo "FATAL: no result CSVs found under $DATA_ROOT" >&2
    exit 1
fi

echo
echo "=== merging ${#CSVS[@]} result files ==="
python3 common/merge_results.py -o "$OUT/all_points.csv"          "${CSVS[@]}"
python3 common/merge_results.py -o "$OUT/pareto.csv"    --pareto  "${CSVS[@]}"

echo
echo "=== figures (auto recall axis) ==="
python3 common/plot_results.py -i "$OUT/pareto.csv" -o "$OUT/figures"

echo
echo "=== figures_zoom (recall axis from $ZOOM_XMIN) ==="
python3 common/plot_results.py -i "$OUT/pareto.csv" -o "$OUT/figures_zoom" --xmin "$ZOOM_XMIN"

echo
echo "=== produced ==="
find "$OUT" -type f | sort
