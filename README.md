# spknn-playground

Two clusters are supported. Files are suffixed by machine; **nothing is
shared except the Python harness** (`common/bench_harness.py`,
`plot_results.py`, `merge_results.py`), which is environment-driven and works
on both.

| | TAMU Grace | **NERSC Perlmutter** |
|---|---|---|
| CPU | Intel Xeon Gold 6248R (Cascade Lake) | AMD EPYC 7763 (Zen3 "Milan") |
| cores | 48 (2 x 24, SMT off) | 128 (2 x 64, SMT on -> 256) |
| NUMA | 2 domains | **8 domains (NPS4)**, 16 cores + ~51 GB/s each |
| AVX-512 | yes | **no** — Zen3 is AVX2-only |
| target ISA | `-march=cascadelake` | `-march=znver3` |
| `perf` | not installed | available (`/usr/bin/perf`) |
| env file | `common/bench_env.sh` | `common/bench_env_perlmutter.sh` |

---

## Running the benchmark — Perlmutter

One command submits everything (6 method jobs + a figures job that waits on
them). All jobs run **64 threads**.

```bash
bash common/submit_full_benchmark_perlmutter.sh
# DRY_RUN=1 ...            to print what would be submitted
# AFTER_JOB=<jobid> ...     to chain behind another job
```

Individually:

```bash
bash common/build_bench_venv_perlmutter.sh        # ONE-TIME, login node (needs network)
sbatch common/smoke_test_perlmutter.sh            # correctness, msmarco_small
sbatch grassRMA/run_grassRMA_perlmutter.sh
sbatch kannolo/run_kannolo_perlmutter.sh
sbatch seismic/run_seismic_perlmutter.sh
sbatch pyanns/run_pyanns_perlmutter.sh
cd $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts
sbatch run_hnsw_sweep_perlmutter.sh               # SPARSE_HNSW (C++)
sbatch run_sindi_sweep_perlmutter.sh              # SINDI (C++)
```

Figures (also submitted automatically by `submit_full_benchmark_perlmutter.sh`):

```bash
bash common/make_figures_perlmutter.sh            # cheap enough for a login node
# -> results/msmarco_full_perlmutter/{all_points.csv,pareto.csv,figures,figures_zoom}
```

### Running on Grace (original)

```bash
module purge && module load GCCcore/13.2.0 Python/3.11.5
source /scratch/user/dhaura/benchmarks/SpKNN/bench-venv/bin/activate
sbatch common/smoke_test.sh
sbatch grassRMA/run_grassRMA_grace.sh
sbatch kannolo/run_kannolo_grace.sh
sbatch seismic/run_seismic_grace.sh
sbatch pyanns/run_pyanns_grace.sh
```

---

## Compilation & optimization (Perlmutter)

Everything is built for **Zen3 / AVX2**. Nothing here uses AVX-512, because
the hardware has none — see "AVX-512" below.

| method | toolchain | flags |
|---|---|---|
| SparseHNSW, grassRMA (C++), SINDI driver | **icpx 2025.3** | `-O3 -march=znver3 -mtune=znver3` |
| GrassRMA Python bindings (`sparse_hnswlib`) | **icpx 2025.3** | `-O3 -march=znver3 -mtune=znver3` |
| SINDI engine (`libvsag`) | gcc 14.3 | `-O3 -march=znver3 -mtune=znver3` |
| PyANNS | gcc 14.3 | `-Ofast -march=native` (= znver3 here) |
| kANNolo | rustc nightly | `-C target-cpu=znver3`, `lto = "fat"` |
| SEISMIC | rustc nightly | `-C target-cpu=znver3`, `lto = true` |

---

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

```bash
module swap PrgEnv-gnu PrgEnv-intel && module load python/3.11-24.1.0
source $SCRATCH/benchmarks/SpKNN/bench-venv-perlmutter/bin/activate
cd $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/GrassRMA
export CC=icx CXX=icpx HNSWLIB_NO_NATIVE=1
export CFLAGS="-march=znver3 -mtune=znver3" CXXFLAGS="$CFLAGS"
rm -rf build && pip install --no-deps --force-reinstall --no-cache-dir .
```

### Upstream instructions (Grace)

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

## SINDI (vsag)

```bash
python3 -m venv $SCRATCH/benchmarks/SpKNN/vsag-venv
source $SCRATCH/benchmarks/SpKNN/vsag-venv/bin/activate
pip install pyvsag numpy scipy
```

Run it:

```bash
cd sindi
sbatch run_sindi_full.sh
```

## Multithreaded benchmarks (Seismic + kANNolo)

```bash
module load python/3.10
python3 -m venv $SCRATCH/benchmarks/SpKNN/mt-venv
source $SCRATCH/benchmarks/SpKNN/mt-venv/bin/activate
pip install kannolo pyseismic-lsr numpy scipy
```

```bash
cd kannolo && sbatch run_kannolo_mt.sh 64 && sbatch run_kannolo_mt.sh 128
cd seismic && sbatch run_seismic_mt.sh 64 && sbatch run_seismic_mt.sh 128
```

## kANNolo


> ```bash
> module load python/3.11-24.1.0 rust/stable
> export RUSTUP_HOME=$SCRATCH/.rustup CARGO_HOME=$SCRATCH/.cargo   # AFTER module load:
> export PATH=$CARGO_HOME/bin:$PATH                                # the module's are read-only
> rustup toolchain install nightly
> export RUSTUP_TOOLCHAIN=nightly RUSTFLAGS="-C target-cpu=znver3"
> source $SCRATCH/benchmarks/SpKNN/bench-venv-perlmutter/bin/activate
> pip install maturin
> cd $SCRATCH/build/rustpkgs/kannolo-0.7.1   # sdist, Cargo.lock deleted
> maturin build --release
> pip install --no-deps --force-reinstall target/wheels/*.whl
> ```


### Upstream instructions (Grace)

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

