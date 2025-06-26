#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/%j.log

export OMP_NUM_THREADS=64

source ~/repos/SpKNN/spknn-playground/venv/bin/activate

export PYTHONUNBUFFERED=1 

python3 kannolo_sp_ex_load.py -query $SCRATCH/datasets/SpKNN/kannolo/experiments/msmarco-splade/queries.bin -index $SCRATCH/datasets/SpKNN/kannolo/example/msmarco-splade-index
