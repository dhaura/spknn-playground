#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/%j.log

export OMP_NUM_THREADS=64

source ~/repos/SpKNN/spknn-playground/venv/bin/activate

export PYTHONUNBUFFERED=1 

python3 kannolo_sp_ex.py -input $SCRATCH/datasets/SpKNN/kannolo/experiments/msmarco-splade/dataset.bin \
    -query $SCRATCH/datasets/SpKNN/kannolo/experiments/msmarco-splade/queries.bin \
    -index $SCRATCH/datasets/SpKNN/kannolo/example/msmarco-splade-index \
    -gt $SCRATCH/datasets/SpKNN/kannolo/experiments/msmarco-splade/groundtruth.tsv \
    -output $SCRATCH/datasets/SpKNN/kannolo/example/msmarco-splade-results.tsv \
    -query_ids $SCRATCH/datasets/SpKNN/kannolo/experiments/msmarco-splade/queries_ids.npy \
    -doc_ids $SCRATCH/datasets/SpKNN/kannolo/experiments/msmarco-splade/doc_ids.npy \
    -qrels $SCRATCH/datasets/SpKNN/kannolo/experiments/msmarco-splade/qrels.dev.small.tsv
