#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/%j.log

export OMP_NUM_THREADS=64

source ~/repos/SpKNN/spknn-playground/venv/bin/activate

python3 seismic_sp_ex.py -input $SCRATCH/datasets/SpKNN/seismic/example/docs_anserini.jsonl -query $SCRATCH/datasets/SpKNN/seismic/example/queries_anserini.tsv
