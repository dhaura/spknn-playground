#!/bin/bash
#SBATCH --job-name=seismic_nq
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=350G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#
# SEISMIC baseline on Grace for nq_splade.  Grace port of
# run_seismic_v2_perlmutter.sh.  TRIPLES are the 61 (n_postings,
# summary_energy, centroid_fraction) index configurations of the nq_splade
# Perlmutter run; the 19 query_cut:heap_factor sweep pairs are the v2 default
# (they match that run exactly).  One index build per triple, swept in-memory.
#
#   cd seismic && mkdir -p logs && sbatch run_seismic_grace_nq.sh
# Split across jobs with e.g. TRIPLES="500:0.6:0.2 500:0.6:0.3".

set -euo pipefail
PG=/scratch/user/dhaura/repos/spknn-playground
cd "$PG/seismic"; mkdir -p logs
module purge; module load GCCcore/13.2.0; module load Python/3.11.5
export SPKNN_DATASET=${SPKNN_DATASET:-nq_splade}
source "$PG/common/bench_env.sh"; source "$PG/common/dataset_env_grace.sh"
bench_provenance; source "$SPKNN_VENV/bin/activate"

OUT=${SPKNN_OUT:-$SPKNN_OUT_ROOT/seismic}; mkdir -p "$OUT/indices" "$OUT/bin"
RM_INDEX=${RM_INDEX-1}   # bare flag; unset RM_INDEX to keep indices
SWEEP=${SEISMIC_SWEEP:-1:1.0,2:1.0,3:1.0,3:0.9,4:0.9,5:0.9,5:1.0,6:0.9,7:0.9,8:0.9,10:0.9,10:0.8,12:0.8,14:0.8,20:0.8,20:0.7,30:0.7,30:0.8,50:0.7}
TRIPLES=${TRIPLES:-"500:0.6:0.2 500:0.6:0.3 500:0.6:0.4 500:0.7:0.2 500:0.7:0.3 500:0.7:0.4 500:0.8:0.2 500:0.8:0.3 500:0.8:0.4 1000:0.6:0.2 1000:0.6:0.3 1000:0.6:0.4 1000:0.7:0.2 1000:0.7:0.3 1000:0.7:0.4 1000:0.8:0.2 1000:0.8:0.3 1000:0.8:0.4 2000:0.4:0.05 2000:0.4:0.1 2000:0.4:0.2 2000:0.5:0.05 2000:0.5:0.1 2000:0.5:0.2 2000:0.6:0.05 2000:0.6:0.1 2000:0.6:0.2 2000:0.6:0.3 2000:0.6:0.4 2000:0.7:0.2 2000:0.7:0.3 2000:0.7:0.4 2000:0.8:0.2 2000:0.8:0.3 2000:0.8:0.4 3000:0.4:0.05 3000:0.4:0.1 3000:0.4:0.2 3000:0.5:0.05 3000:0.5:0.1 3000:0.5:0.2 3000:0.6:0.05 3000:0.6:0.1 3000:0.6:0.2 6000:0.4:0.05 6000:0.4:0.1 6000:0.4:0.2 6000:0.5:0.05 6000:0.5:0.1 6000:0.5:0.2 6000:0.6:0.05 6000:0.6:0.1 6000:0.6:0.2 6000:0.6:0.3 6000:0.6:0.4 6000:0.7:0.2 6000:0.7:0.3 6000:0.7:0.4 6000:0.8:0.2 6000:0.8:0.3 6000:0.8:0.4"}

echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET   threads=$BENCH_THREADS"
echo "### triples : $(echo $TRIPLES | wc -w)   sweep pairs: $(echo $SWEEP | tr ',' ' ' | wc -w)"
for T in $TRIPLES; do
    NP=${T%%:*}; REST=${T#*:}; SE=${REST%%:*}; CF=${REST##*:}
    CSV="$OUT/seismic_results_${SPKNN_DATASET}_np${NP}_se${SE}_cf${CF}_${SLURM_JOB_ID}.csv"
    echo "########## n_postings=$NP summary_energy=$SE centroid_fraction=$CF ##########"
    $BENCH_LAUNCH stdbuf -oL -eL python3 seismic_mt_ex.py \
        -n_postings "$NP" -summary_energy "$SE" -centroid_fraction "$CF" \
        -min_cluster_size "${MIN_CLUSTER:-2}" -doc_cut "${DOC_CUT:-15}" \
        -max_fraction "${MAX_FRACTION:-6}" -sweep "$SWEEP" \
        -nknn "${NKNN:-0}" -n_knn_list "${N_KNN_LIST:-0}" -sorted "${SORTED:-both}" \
        -repeats "${REPEATS:-3}" -warmup "${WARMUP:-1}" \
        -input "$SPKNN_BASE" -query "$SPKNN_QUERIES" -gt "$SPKNN_GT" \
        -bin_dir "$OUT/bin" -index "$OUT/indices/${SPKNN_DATASET}_np${NP}_se${SE}_cf${CF}.index" \
        ${RM_INDEX:+-rm_index} -csv "$CSV"
done
echo "########## done -> $OUT ##########"
