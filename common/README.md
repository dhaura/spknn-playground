# Benchmark Environment

## Files

| file | machine | purpose |
|---|---|---|
| `bench_env_perlmutter.sh` | Perlmutter | Sourced by every job. Threads, pinning, NUMA, ISA target, **dataset selection**, provenance, and the optimized-build guard. |
| `build_bench_venv_perlmutter.sh` | Perlmutter | Builds the venv holding all four Python-run methods, at `-march=znver3`. **Login node only** — compute nodes have no outbound network. |
| `smoke_test_perlmutter.sh` | Perlmutter | End-to-end correctness on msmarco_small, all six methods. Run before any full sweep. |
| `submit_full_benchmark_perlmutter.sh` | Perlmutter | Submits the whole msmarco_full benchmark: 6 method jobs + a figures job chained behind them. |
| `make_figures_perlmutter.sh` | Perlmutter | Merges every method CSV, computes the Pareto frontier, writes both figure sets. |
| `bench_env.sh` | Grace | Original. **Do not source on Perlmutter** — see the NUMA warning below. |
| `build_bench_venv.sh` | Grace | Original, `ARCH=cascadelake`. |
| `smoke_test.sh` | Grace | Original. |
| `bench_harness.py` | both | Shared measurement definitions: median-of-N timing, recall/RR@10, the CSV schema. |
| `merge_results.py` | both | Merge per-method CSVs; `--pareto` keeps only each method's (recall, QPS) frontier. |
| `plot_results.py` | both | The five figures, PNG + PDF, plus `figure_data.tsv`. |
| `sbatch_header.template` | Grace | The canonical Grace `#SBATCH` block. |

