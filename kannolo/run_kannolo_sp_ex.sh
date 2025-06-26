#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/%j.log

export OMP_NUM_THREADS=64

source ~/repos/SpKNN/spknn-playground/venv/bin/activate

export PYTHONUNBUFFERED=1 

python3 kannolo_sp_ex.py -input $SCRATCH/datasets/SpKNN/kannolo/experiments/msmarco-splade/dataset.bin -query $SCRATCH/datasets/SpKNN/kannolo/experiments/msmarco-splade/queries.bin -index $SCRATCH/datasets/SpKNN/kannolo/example/msmarco-splade-index
