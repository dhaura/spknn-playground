"""SEISMIC benchmark.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import bench_harness as bh

parser = argparse.ArgumentParser(description="SEISMIC sparse benchmark.")
parser.add_argument("-input", required=True, help="Base CSR file (BigANN sparse format).")
parser.add_argument("-query", required=True, help="Query CSR file (BigANN sparse format).")
parser.add_argument("-gt", required=True, help="Ground truth file.")
parser.add_argument("-csv", required=True, help="Results CSV to append to.")
parser.add_argument("-bin_dir", required=True, help="Where the converted .bin files live.")
parser.add_argument("-index", default=None,
                    help="If it exists, load it (search-only run). Else build and save here.")
parser.add_argument("-n_postings", type=int, default=3500)
parser.add_argument("-summary_energy", type=float, default=0.4)
parser.add_argument("-k", type=int, default=10)
parser.add_argument("-repeats", type=int, default=5)
parser.add_argument("-warmup", type=int, default=1)
parser.add_argument("-sweep", default="3:0.9,5:0.9,10:0.9,10:0.8,20:0.8,20:0.7,30:0.7",
                    help="Comma-separated query_cut:heap_factor pairs.")
args = parser.parse_args()

threads = bh.env_threads()
os.environ.setdefault("RAYON_NUM_THREADS", str(threads))

import numpy as np
from seismic import SeismicIndexRaw

k = args.k
sweep = []
for pair in args.sweep.split(","):
    qc, hf = pair.split(":")
    sweep.append((int(qc), float(hf)))

print(f"SEISMIC | n_postings={args.n_postings} summary_energy={args.summary_energy} "
      f"threads={threads} (RAYON_NUM_THREADS={os.environ['RAYON_NUM_THREADS']})", flush=True)

os.makedirs(args.bin_dir, exist_ok=True)
base_bin = os.path.join(args.bin_dir, "base.bin")
query_bin = os.path.join(args.bin_dir, "queries.bin")

# --- convert -------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utilities"))
convert_sec = 0.0
todo = [(args.input, base_bin), (args.query, query_bin)]
missing = [(src, dst) for src, dst in todo if not os.path.exists(dst)]
if missing:
    from convert_csr_to_bin import convert_csr_to_bin  # noqa: E402
    with bh.phase("convert csr -> bin") as p_conv:
        for src, dst in missing:
            print(f"    {src} -> {dst}", flush=True)
            convert_csr_to_bin(src, dst)
    convert_sec = p_conv.sec
else:
    print("  .bin files already present; ConvertSec reported as 0 "
          "(re-run with a clean -bin_dir to measure it)", flush=True)

# --- load + index --------------------------------------------------------
load_sec = 0.0
index_sec = 0.0

if args.index and os.path.exists(args.index):
    print(f"  loading existing index {args.index} (search-only run)", flush=True)
    with bh.phase("load index"):
        index = SeismicIndexRaw.load(args.index)
else:
    # build() reads base_bin itself; warm the cache and charge it to LoadSec.
    with bh.phase("load (page-cache warm)") as p_load:
        with open(base_bin, "rb") as f:
            while f.read(1 << 24):
                pass
    load_sec = p_load.sec

    with bh.phase("index") as p_index:
        index = SeismicIndexRaw.build(
            base_bin, n_postings=args.n_postings, summary_energy=args.summary_energy
        )
    index_sec = p_index.sec

    if args.index:
        with bh.phase("save index"):
            index.save(args.index)

print(f"  peak RSS after indexing: {bh.peak_rss_gb():.1f} GB", flush=True)

gt = bh.read_gt(args.gt)
n_queries = gt.shape[0]
print(f"  {n_queries} queries, k={k}, {len(sweep)} sweep points\n", flush=True)

index_bytes = bh.path_bytes(args.index)

rows = []
for query_cut, heap_factor in sweep:
    def run():
        # The num_threads argument is a no-op in this build; RAYON_NUM_THREADS rules.
        return index.batch_search(
            query_bin, k, query_cut, heap_factor, 0, True, num_threads=threads
        )

    results, med, times = bh.timed_search(
        run, repeats=args.repeats, warmup=args.warmup,
        label=f"qc={query_cut} hf={heap_factor}",
    )

    # batch_search returns per-query lists of (score, doc_id), best first.
    pred = np.array(
        [[doc for _, doc in r] + [-1] * (k - len(r)) for r in results], dtype=np.int64
    )
    # Pad with -inf so short result lists still read as descending-by-score.
    scores = np.array(
        [[s for s, _ in r] + [-np.inf] * (k - len(r)) for r in results], dtype=np.float64
    )
    ordered = bh.check_sorted_by_score(scores, "SEISMIC")

    row = bh.make_row(
        model="SEISMIC",
        params=f"n_postings={args.n_postings} summary_energy={args.summary_energy} "
               f"query_cut={query_cut} heap_factor={heap_factor}",
        threads=threads,
        gt=gt,
        pred=pred,
        k=k,
        index_sec=index_sec,
        load_sec=load_sec,
        convert_sec=convert_sec,
        search_times=times,
        index_bytes=index_bytes,
        results_ordered=ordered,
    )
    bh.print_point(row)
    rows.append(row)

bh.write_rows(args.csv, rows)
