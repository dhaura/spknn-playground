#!/bin/bash
# Shared benchmark environment
#
# --- thread count ----------------------------------------------------------
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    BENCH_THREADS="${BENCH_THREADS:-$SLURM_CPUS_PER_TASK}"
else
    BENCH_THREADS="${BENCH_THREADS:-48}"
fi

export OMP_NUM_THREADS="$BENCH_THREADS"
export RAYON_NUM_THREADS="$BENCH_THREADS"   # Rust
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- pinning ---------------------------------------------------------------
export OMP_PROC_BIND=false
unset OMP_PLACES

# --- NUMA ------------------------------------------------------------------
# Grace c-nodes are 2 sockets x 24 cores (Xeon Gold 6248R).
if command -v numactl >/dev/null 2>&1; then
    BENCH_LAUNCH="numactl --cpunodebind=0-1 --interleave=0-1"
else
    BENCH_LAUNCH=""
    echo "bench_env: WARNING numactl not found; NUMA policy is NOT controlled." >&2
fi
export BENCH_LAUNCH

export PYTHONUNBUFFERED=1

# --- canonical paths -------------------------------------------------------
export SPKNN_PLAYGROUND=/scratch/user/dhaura/repos/spknn-playground
export SPKNN_HNSW_REPO=/scratch/user/dhaura/repos/minimal_hnsw
export SPKNN_BIN="$SPKNN_HNSW_REPO/build-release/bin"
export SPKNN_DATA="$SPKNN_HNSW_REPO/sparse/data/msmarco_full"
export SPKNN_VENV=/scratch/user/dhaura/benchmarks/SpKNN/bench-venv

export SPKNN_MODULES="GCCcore/13.2.0 Python/3.11.5"

bench_load_modules() {
    module purge
    module load GCCcore/13.2.0
    module load Python/3.11.5
}

if [ -d "$SPKNN_HNSW_REPO/build" ]; then
    echo "bench_env: FATAL a '$SPKNN_HNSW_REPO/build' tree exists." >&2
    echo "           Remove it and use build-release only." >&2
    exit 1
fi

# --- provenance ------------------------------------------------------------
bench_provenance() {
    echo "=================== BENCH PROVENANCE ==================="
    echo "date          : $(date -Is)"
    echo "host          : $(hostname)"
    echo "slurm job     : ${SLURM_JOB_ID:-<none>}  partition=${SLURM_JOB_PARTITION:-<none>}"
    echo "nodelist      : ${SLURM_JOB_NODELIST:-<none>}"
    echo "cpus-per-task : ${SLURM_CPUS_PER_TASK:-<unset>}"
    echo "bench threads : $BENCH_THREADS"
    echo "launcher      : ${BENCH_LAUNCH:-<none>}"
    echo "cpu           : $(lscpu | sed -n 's/^Model name: *//p')"
    echo "sockets/cores : $(lscpu | sed -n 's/^Socket(s): *//p') sockets x $(lscpu | sed -n 's/^Core(s) per socket: *//p') cores"
    echo "-- numactl -H --"
    numactl -H 2>/dev/null | head -12
    echo "-- git --"
    for r in "$SPKNN_PLAYGROUND" "$SPKNN_HNSW_REPO"; do
        printf '%-28s %s %s\n' "$(basename "$r")" \
            "$(git -C "$r" rev-parse --short HEAD 2>/dev/null || echo '?')" \
            "$(git -C "$r" diff --quiet 2>/dev/null && echo clean || echo DIRTY)"
    done
    echo "-- modules --"
    (module list) 2>&1 | head -12
    echo "-- binaries --"
    for b in "$SPKNN_BIN"/*; do
        [ -f "$b" ] || continue
        printf '%-24s %s  %s\n' "$(basename "$b")" \
            "$(date -r "$b" '+%Y-%m-%d %H:%M')" "$(sha256sum "$b" | cut -c1-16)"
    done
    echo "========================================================"
}

# Prove from the log alone that a benchmarked binary was optimized.
bench_assert_optimized() {
    local bin="$1"
    local ymm
    ymm=$(objdump -d --no-show-raw-insn "$bin" 2>/dev/null | grep -coE '%ymm[0-9]+')
    echo "bench_env: $(basename "$bin") vector-register refs (ymm): $ymm"
    if [ "${ymm:-0}" -eq 0 ]; then
        echo "bench_env: FATAL $bin contains no vector instructions -- it is an" >&2
        echo "           unoptimized build. Refusing to benchmark it." >&2
        exit 1
    fi
}
