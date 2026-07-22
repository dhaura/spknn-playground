import argparse
import csv
import os
import time

import numpy as np
from kannolo import SparsePlainHNSW

parser = argparse.ArgumentParser(description="Multithreaded kANNolo sparse benchmark.")
parser.add_argument("-input", type=str, help="Path to the base CSR file (BigANN sparse format).")
parser.add_argument("-query", type=str, help="Path to the query CSR file.")
parser.add_argument("-gt", type=str, help="Path to the ground truth file.")
parser.add_argument("-index", type=str, help="Index path: loaded if it exists, else built and saved here.")
parser.add_argument("-num_threads", type=int, default=64, help="Threads for batch query search.")
parser.add_argument("-k", type=int, default=10)
parser.add_argument("-m", type=int, default=32, help="HNSW neighbors per node.")
parser.add_argument("-ef_construction", type=int, default=200)
parser.add_argument("-csv", type=str, help="Optional path to append results as CSV rows.")
args = parser.parse_args()


def read_bigann_csr(fname):
    with open(fname, "rb") as f:
        nrow, ncol, nnz = np.fromfile(f, dtype="int64", count=3)
        indptr = np.fromfile(f, dtype="int64", count=nrow + 1)
        indices = np.fromfile(f, dtype="int32", count=nnz)
        data = np.fromfile(f, dtype="float32", count=nnz)
    return indptr, indices, data


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
    index = SparsePlainHNSW.load(args.index)
    print(f"Index loaded in {time.time() - t0:.2f} seconds.")
else:
    print(f"Reading base vectors from {args.input}...")
    b_indptr, b_indices, b_data = read_bigann_csr(args.input)
    print(f"Building index (m={args.m}, ef_construction={args.ef_construction})...")
    t0 = time.time()
    index = SparsePlainHNSW.build_from_arrays(
        b_indices, b_data, b_indptr, m=args.m, ef_construction=args.ef_construction, metric="ip"
    )
    indexing_time = time.time() - t0
    print(f"Index built in {indexing_time:.2f} seconds.")
    if args.index:
        index.save(args.index)
        print(f"Index saved to {args.index}")

q_indptr, q_indices, q_data = read_bigann_csr(args.query)
n_queries = len(q_indptr) - 1
I = knn_result_read(args.gt)
k = args.k
nt = args.num_threads
print(f"{n_queries} queries, k={k}, num_threads={nt}")


def run(ef_search):
    start = time.time()
    dists, ids = index.batch_search(
        q_indices, q_data, q_indptr, k=k, ef_search=ef_search, num_threads=nt
    )
    elapsed = time.time() - start
    return ids.reshape(n_queries, k), elapsed


print("Warm-up pass...")
run(100)

rows = []
for ef_search in [50, 100, 200, 400, 800, 1600, 3200]:
    print(f"Setting ef_search to {ef_search}...")
    res, elapsed = run(ef_search)
    intersections = np.array([np.intersect1d(a, b).size for a, b in zip(I[:, :k], res)])
    recall = intersections.sum() / (I.shape[0] * k)
    rr = rr_at_k(I[:, :k], res, k)
    qps = n_queries / elapsed
    print(f"Elapsed: {elapsed:.4f}s; {qps:.2f} QPS at {nt} threads")
    print(f"Recall@{k}: {recall * 100:.4f}")
    print(f"RR@{k} (vs exact-NN gt): {rr:.4f}\n")
    rows.append(("kANNolo", nt, ef_search, recall, indexing_time, elapsed, qps, rr))

if args.csv:
    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                ["Model", "Threads", "ef_search", "Recall", "Indexing Time",
                 "Searching Time (Seconds)", "QPS", "RR@10 (vs exact-NN gt)"]
            )
        w.writerows(rows)
    print(f"Results appended to {args.csv}")
