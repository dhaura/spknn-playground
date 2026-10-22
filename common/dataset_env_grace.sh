#!/bin/bash
# Grace per-dataset environment.  Source AFTER common/bench_env.sh.
#
# Grace's bench_env.sh only exports SPKNN_DATA (msmarco_full).  The dataset-
# generic playground runners (the *_v2_* scripts and everything derived from
# them) expect the Perlmutter variables SPKNN_DATASET / SPKNN_QSET /
# SPKNN_BASE / SPKNN_QUERIES / SPKNN_GT / SPKNN_OUT_ROOT.  This shim supplies
# them for Grace without touching bench_env.sh.
#
#   export SPKNN_DATASET=nq_splade
#   source common/bench_env.sh
#   source common/dataset_env_grace.sh
#
# SPKNN_OUT_ROOT is per-dataset (datasets/SpKNN/<dataset>/<method>/), matching
# the layout make_bloom_playground_figures*.sh expects.  Note the older Grace
# runners wrote to a FLAT datasets/SpKNN/<method>/ -- those are msmarco_full
# results and are left exactly where they are.

export SPKNN_DATASET=${SPKNN_DATASET:-msmarco_full}
case "$SPKNN_DATASET" in
    msmarco_full)
        export SPKNN_DATA="$SPKNN_HNSW_REPO/sparse/data/msmarco_full"
        export SPKNN_QSET="${SPKNN_QSET:-dev}"
        export SPKNN_BASE="$SPKNN_DATA/base_full.csr"
        export SPKNN_GT="$SPKNN_DATA/base_full.$SPKNN_QSET.gt"
        export SPKNN_NDOCS=8841823 SPKNN_NNZ=1121199371 ;;
    nq_splade)
        export SPKNN_DATA="$SPKNN_HNSW_REPO/sparse/data/nq_splade"
        export SPKNN_QSET="${SPKNN_QSET:-test}"
        export SPKNN_BASE="$SPKNN_DATA/base_nq.csr"
        export SPKNN_GT="$SPKNN_DATA/base_nq.$SPKNN_QSET.gt"
        export SPKNN_NDOCS=2680893 SPKNN_NNZ=411886108 ;;
    msmarco_small)
        export SPKNN_DATA="$SPKNN_HNSW_REPO/sparse/data/msmarco_small"
        export SPKNN_QSET="${SPKNN_QSET:-dev}"
        export SPKNN_BASE="$SPKNN_DATA/base_small.csr"
        export SPKNN_GT="$SPKNN_DATA/base_small.$SPKNN_QSET.gt"
        export SPKNN_NDOCS=1000000 SPKNN_NNZ=0 ;;
    *)
        echo "dataset_env_grace: unknown SPKNN_DATASET='$SPKNN_DATASET'" >&2
        echo "           known: msmarco_full, nq_splade, msmarco_small" >&2
        return 1 2>/dev/null || exit 1 ;;
esac
export SPKNN_QUERIES="$SPKNN_DATA/queries.$SPKNN_QSET.csr"
export SPKNN_BIG=0
[ "${SPKNN_NNZ:-0}" -gt 4294967295 ] && export SPKNN_BIG=1
export SPKNN_OUT_ROOT=${SPKNN_OUT_ROOT:-/scratch/user/dhaura/datasets/SpKNN/$SPKNN_DATASET}

for _f in "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT"; do
    [ -r "$_f" ] || { echo "dataset_env_grace: FATAL missing data file: $_f" >&2
                      return 1 2>/dev/null || exit 1; }
done
echo "dataset_env_grace: $SPKNN_DATASET/$SPKNN_QSET  out=$SPKNN_OUT_ROOT"
