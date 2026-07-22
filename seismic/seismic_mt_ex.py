import argparse
import csv
import os
import sys
import time

parser = argparse.ArgumentParser(description="Multithreaded Seismic raw-index benchmark.")
parser.add_argument("-input", type=str, help="Path to the base .bin file (Seismic inner format).")
parser.add_argument("-query", type=str, help="Path to the query .bin file (Seismic inner format).")
parser.add_argument("-gt", type=str, help="Path to the BigANN ground truth file.")
parser.add_argument("-index", type=str, help="Index path: loaded if it exists, else built and saved here.")
parser.add_argument("-num_threads", type=int, default=64, help="Threads for build and batch search.")
parser.add_argument("-k", type=int, default=10)
parser.add_argument("-n_postings", type=int, default=3500)
parser.add_argument("-summary_energy", type=float, default=0.4)
parser.add_argument("-csv", type=str, help="Optional path to append results as CSV rows.")
args = parser.parse_args()

os.environ["RAYON_NUM_THREADS"] = str(args.num_threads)

import numpy as np  # noqa: E402
from seismic import SeismicIndexRaw  # noqa: E402


def knn_result_read(fname):
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    with open(fname, "rb") as f:
        f.seek(8)
        I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    return I


def rr_at_k(gt, pred, k):
    total = 0.0
    for g, p in zip(gt, pred):
        g_set = set(g.tolist())
        for rank, item in enumerate(p[:k], start=1):
            if item in g_set:
                total += 1.0 / rank
                break
    return total / gt.shape[0]


indexing_time = 0.0
if args.index and os.path.exists(args.index):
    print(f"Loading index from {args.index}...")
    t0 = time.time()
    index = SeismicIndexRaw.load(args.index)
    print(f"Index loaded in {time.time() - t0:.2f} seconds.")
else:
    print(f"Building index from {args.input} (n_postings={args.n_postings}, "
          f"summary_energy={args.summary_energy}, {args.num_threads} rayon threads)...")
    t0 = time.time()
    index = SeismicIndexRaw.build(
        args.input, n_postings=args.n_postings, summary_energy=args.summary_energy
    )
    indexing_time = time.time() - t0
    print(f"Index built in {indexing_time:.2f} seconds.")
    if args.index:
        index.save(args.index)
        print(f"Index saved to {args.index}")

I = knn_result_read(args.gt)
n_queries = I.shape[0]
k = args.k
nt = args.num_threads
print(f"{n_queries} queries, k={k}, RAYON_NUM_THREADS={nt}")


def run(query_cut, heap_factor):
    start = time.time()
    # num_threads below is the broken no-op parameter; RAYON_NUM_THREADS rules.
    results = index.batch_search(
        args.query, k, query_cut, heap_factor, 0, True, num_threads=nt
    )
    elapsed = time.time() - start
    ids = np.array([[doc for _, doc in r] + [-1] * (k - len(r)) for r in results], dtype=np.int64)
    return ids, elapsed


print("Warm-up pass...")
run(10, 0.8)

rows = []
for query_cut, heap_factor in [
    (3, 0.9), (5, 0.9), (10, 0.9), (10, 0.8), (20, 0.8), (20, 0.7), (30, 0.7),
]:
    print(f"Setting query_cut={query_cut}, heap_factor={heap_factor}...")
    res, elapsed = run(query_cut, heap_factor)
    intersections = np.array([np.intersect1d(a, b).size for a, b in zip(I[:, :k], res)])
    recall = intersections.sum() / (I.shape[0] * k)
    rr = rr_at_k(I[:, :k], res, k)
    qps = n_queries / elapsed
    print(f"Elapsed: {elapsed:.4f}s; {qps:.2f} QPS at {nt} threads")
    print(f"Recall@{k}: {recall * 100:.4f}")
    print(f"RR@{k} (vs exact-NN gt): {rr:.4f}\n")
    rows.append(("SEISMIC", nt, query_cut, heap_factor, recall, indexing_time, elapsed, qps, rr))

if args.csv:
    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                ["Model", "Threads", "query_cut", "heap_factor", "Recall",
                 "Indexing Time", "Searching Time (Seconds)", "QPS", "RR@10 (vs exact-NN gt)"]
            )
        w.writerows(rows)
    print(f"Results appended to {args.csv}")
