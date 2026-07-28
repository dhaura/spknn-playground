#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=02:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=pyanns_small
#SBATCH --output=logs/%j.log

module load GCCcore/14.2.0
module load Python/3.13.1

source $SCRATCH/repos/pyanns/venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32

python3 pyanns_ex.py -n 100000 -nq 6980 -budget 0.1 -input $SCRATCH/data/SpKNN/pyanns/base_small.csr \
	-query $SCRATCH/data/SpKNN/pyanns/queries.dev.csr \
	-gt $SCRATCH/data/SpKNN/pyanns/base_small.dev.gt \
	-output $SCRATCH/datasets/SpKNN/pyanns/indices/base_small
