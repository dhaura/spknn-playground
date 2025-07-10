# spknn-playground

```bash
module load python/3.10
python3 -m venv venv
source venv/bin/activate
```

## UMAP

Sparse Data Format - LIL (List of Lists) => CSR 

```bash
pip install numpy scipy sympy scikit-learn matplotlib umap-learn umap-learn[plot]
```

## SEISMIC

```bash
pip install pyseismic-lsr ir_datasets ir_measures
```

Git repo experiments - https://github.com/TusKANNy/seismic/blob/main/docs/RunExperiments.md
Fixes - https://github.com/dhaura/seismic/tree/exp_dtp

## GrassRMA

Build GrassRMA module.
```bash
git clone https://github.com/Leslie-Chung/GrassRMA.git
cd GrassRMA
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
module load python/3.13
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Download test dataset.
```bash
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_small.csr.gz
gunzip base_small.csr.gz
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/queries.dev.csr.gz
gunzip queries.dev.csr.gz
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_small.dev.gt
```

Execute GrassRMA example.
```bash
cd grassRMA
source $SCRATCH/benchmarks/SpKNN/GrassRMA/venv/bin/activate
pip install numpy
```

## PyANNS

> Requires AVX-512 instruction which is only availbale in intel CPUs and not available with AMD CPUs and thus, cannot run in NERSC.

Build PyANNS module in TAMU Grace Cluster.
```bash
git clone https://github.com/hhy3/pyanns.git
cd pyanns
module load GCCcore/14.2.0
module load Python/3.13.1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
chmod +x build.sh
pip install setuptools
./build.sh
```

## kANNolo

Run these from ` $SCRATCH/benchmarks/SpKNN/forks/kannolo`.

```bash
rustup install nightly
rustup override set nightly
rustup default nightly
module load cray-hdf5/1.12.2.9
source venv/bin/activate
pip install -r scripts/requirements.txt
RUSTFLAGS="-C target-cpu=native" pip install --no-binary :all: kannolo
```

Git repo experiments - https://github.com/TusKANNy/kannolo/blob/main/docs/RunExperiments.md

## HiC Cell SpKNN

```bash
module load conda
conda create -n hicspknn python=3.10
conda activate hicspknn
conda install sparse-neighbors-search -c bioconda
conda install pytest
```


### NeurIPS BigANN 23 Datasets
https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/dataset_preparation/sparse_dataset.md

