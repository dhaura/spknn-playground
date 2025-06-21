#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/%j.log

export OMP_NUM_THREADS=64

source ~/repos/SpKNN/spknn-playground/venv/bin/activate

python3 umap_sp_math_ex.py
