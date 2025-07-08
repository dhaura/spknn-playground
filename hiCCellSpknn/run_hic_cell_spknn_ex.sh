#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/%j.log

export OMP_NUM_THREADS=64

module load conda
conda activate hicspknn

export PYTHONUNBUFFERED=1 

python3 hic_cell_spknn_ex.py -input $SCRATCH/datasets/SpKNN/hiCCellSpknn/nagano_1mb_intermediate.npz \
    -num_threads 64
