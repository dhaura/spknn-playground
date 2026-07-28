# Deprecated job scripts (superseded 2026-07-27)

These are the July 2026 job scripts. They are kept only so the provenance of the
old results is auditable. **Do not run them.** Every one of them:

* points at a venv that no longer exists (`mt-venv`, `vsag-venv`, `pyanns/venv`);
* uses `--ntasks-per-node=48`, which requests 48 one-CPU tasks rather than one
  48-CPU task;
* sets no NUMA policy, so the 9 GB base vectors land on one socket and half the
  threads pay remote-memory latency;
* calls runner scripts whose arguments and CSV schema have both changed.

Replacements:

| old | new |
|---|---|
| `grassRMA/run_grassRMA_full_grace.sh` | `grassRMA/run_grassRMA_grace.sh` |
| `kannolo/run_kannolo_mt_grace.sh` | `kannolo/run_kannolo_grace.sh` |
| `seismic/run_seismic_mt_grace.sh` | `seismic/run_seismic_grace.sh` |
| `pyanns/run_pyanns_ex_full_grace.sh` | `pyanns/run_pyanns_grace.sh` |
| `sindi/run_sindi_full.sh` | `minimal_hnsw/sparse/scripts/run_sindi_sweep.sh` (C++ driver) |

See `common/README.md` for what changed and why.
