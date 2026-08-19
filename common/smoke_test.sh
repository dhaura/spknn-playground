#!/bin/bash
#SBATCH --job-name=spknn_smoke
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=350G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail   # NOT -e: one failing method should not hide the others

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

source /scratch/user/dhaura/repos/spknn-playground/common/bench_env.sh
bench_provenance

echo
echo "---- thread affinity check ----"
"$SPKNN_VENV/bin/python3" - <<'PY' || exit 1
import ctypes, os, sys
ctypes.CDLL("libgomp.so.1").omp_get_max_threads()
n = len(os.sched_getaffinity(0))
want = int(os.environ.get("BENCH_THREADS", "48"))
print(f"affinity mask after OpenMP init: {n} CPUs (expected {want})")
if n < want:
    print("FATAL: affinity collapsed. Non-OpenMP threads (GrassRMA bindings,\n"
          "rayon) would all share these CPUs. Check OMP_PROC_BIND/OMP_PLACES.",
          file=sys.stderr)
    sys.exit(1)
PY

PG=$SPKNN_PLAYGROUND
SMALL=$SPKNN_HNSW_REPO/sparse/data/msmarco_small
OUT=${SMOKE_OUT:-/scratch/user/dhaura/datasets/SpKNN/smoke}
rm -rf "$OUT"; mkdir -p "$OUT/bin" "$OUT/idx"

BASE=$SMALL/base_small.csr
QUERY=$SMALL/queries.dev.csr
GT=$SMALL/base_small.dev.gt
N=100000
EFS=50,200
REPEATS=3

pass=0; fail=0
run_step () {
    local name="$1"; shift
    echo
    echo "################ $name ################"
    if "$@"; then
        echo "---- $name: OK"
        pass=$((pass + 1))
    else
        echo "---- $name: FAILED (exit $?)" >&2
        fail=$((fail + 1))
    fi
}

# ---- C++ ----------------------------------------------------------------
bench_assert_optimized "$SPKNN_BIN/sparse_hnsw_sweep"
run_step SparseHNSW $BENCH_LAUNCH stdbuf -oL -eL "$SPKNN_BIN/sparse_hnsw_sweep" \
    16 200 "$EFS" 1 0 0 0.8 3 "$BASE" "$QUERY" "$GT" \
    "$OUT/results.csv" SparseHNSW "$REPEATS" 1

if [ -x "$SPKNN_BIN/sindi_sweep" ]; then
    bench_assert_optimized "$SPKNN_BIN/sindi_sweep"
    run_step SINDI $BENCH_LAUNCH stdbuf -oL -eL "$SPKNN_BIN/sindi_sweep" \
        0.3 0.5 20 "$BASE" "$QUERY" "$GT" \
        "$OUT/results.csv" 0 50000 1 0 SINDI "$REPEATS" 1
fi

# ---- Python -------------------------------------------------------------
source "$SPKNN_VENV/bin/activate"

run_step GrassRMA $BENCH_LAUNCH stdbuf -oL -eL python3 "$PG/grassRMA/grassRMA_ex.py" \
    -n $N -M 16 -ef_construction 200 -ef_list "$EFS" -repeats "$REPEATS" \
    -input "$BASE" -query "$QUERY" -gt "$GT" -csv "$OUT/results.csv"

run_step kANNolo $BENCH_LAUNCH stdbuf -oL -eL python3 "$PG/kannolo/kannolo_mt_ex.py" \
    -m 16 -ef_construction 200 -ef_list "$EFS" -repeats "$REPEATS" \
    -input "$BASE" -query "$QUERY" -gt "$GT" \
    -index "$OUT/idx/kannolo.idx" -csv "$OUT/results.csv"

run_step SEISMIC $BENCH_LAUNCH stdbuf -oL -eL python3 "$PG/seismic/seismic_mt_ex.py" \
    -n_postings 3500 -summary_energy 0.4 -repeats "$REPEATS" \
    -sweep "10:0.9,20:0.8" \
    -input "$BASE" -query "$QUERY" -gt "$GT" \
    -bin_dir "$OUT/bin" -index "$OUT/idx/seismic.idx" -csv "$OUT/results.csv"

run_step PyANNS $BENCH_LAUNCH stdbuf -oL -eL python3 "$PG/pyanns/pyanns_ex.py" \
    -n $N -ef 80 -budgets "0.05,0.2" -repeats "$REPEATS" \
    -input "$BASE" -query "$QUERY" -gt "$GT" \
    -index "$OUT/idx/pyanns" -csv "$OUT/results.csv"

# ---- verdict ------------------------------------------------------------
echo
echo "######################## SUMMARY ########################"
echo "steps passed: $pass   failed: $fail"
echo
if [ -f "$OUT/results.csv" ]; then
    python3 - "$OUT/results.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(f"{'Model':<12}{'Params':<44}{'Recall':>8}{'RR@10':>8}{'QPS':>11}  ord  RR>=rec")
bad = 0
for r in rows:
    rec, rr = float(r["Recall"]), float(r["RR@10"])
    ok = rr >= rec - 1e-9
    bad += not ok
    print(f"{r['Model']:<12}{r['Params'][:43]:<44}{rec:8.4f}{rr:8.4f}"
          f"{float(r['QPS']):11.1f}  {r['ResultsOrdered'][:1]}    {'ok' if ok else 'VIOLATION'}")
print()
print(f"{len(rows)} rows, {len({r['Model'] for r in rows})} methods, "
      f"{bad} RR<recall violations")
if bad:
    print("FAIL: RR<recall means that method returns unranked results.")
PY
else
    echo "no results.csv produced" >&2
fi
echo "#########################################################"
[ "$fail" -eq 0 ]
