#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=pyanns_small
#SBATCH --output=logs/%j.log

module load GCCcore/14.2.0
module load Python/3.13.1

source $SCRATCH/repos/pyanns/venv/bin/activate

export PYTHONUNBUFFERED=1

python3 pyanns_ex.py -n 100000 -d 16 -num_threads 2 -input $SCRATCH/data/SpKNN/pyanns/base_small.csr \
	-query $SCRATCH/data/SpKNN/pyanns/queries.dev.csr \
	-gt $SCRATCH/data/SpKNN/pyanns/base_small.dev.gt \
	-output $SCRATCH/datasets/SpKNN/pyanns/indices/base_small
