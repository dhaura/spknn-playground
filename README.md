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

### NeurIPS BigANN 23 Datasets
https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/dataset_preparation/sparse_dataset.md
