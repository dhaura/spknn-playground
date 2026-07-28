# Benchmark Environment

## Files

| file | purpose |
|---|---|
| `bench_env.sh` | Sourced by every job. Fixes threads, pinning and NUMA; prints provenance; refuses to run against an unoptimized build. |
| `bench_harness.py` | Shared measurement definitions for every Python runner. |
| `sbatch_header.template` | The canonical `#SBATCH` block. |
| `build_bench_venv.sh` | Builds the single venv holding all four Python-run methods. |
| `smoke_test.sh` | End-to-end correctness check on msmarco_small. Run before any full sweep. |
