import argparse
import json
import time

import numpy as np
import pyvsag

parser = argparse.ArgumentParser(description="Run kNN search on a SINDI (vsag) sparse index.")
parser.add_argument("-input", type=str, help="Path to the base CSR file (BigANN sparse format).")
parser.add_argument("-query", type=str, help="Path to the query CSR file.")
parser.add_argument("-gt", type=str, help="Path to the ground truth file.")
parser.add_argument("-output", type=str, help="Path to save the index.")
parser.add_argument("-k", type=int, default=10, help="Number of neighbours to retrieve.")
parser.add_argument("-window_size", type=int, default=50000, help="Documents per window (10000-60000).")
parser.add_argument("-doc_prune_ratio", type=float, default=0.0, help="Build-time doc pruning (0.0-0.9).")
parser.add_argument("-query_prune_ratio", type=float, default=0.0, help="Search-time query pruning (0.0-0.9).")
parser.add_argument("-use_quantization", action="store_true", help="SQ8-quantize stored term values.")
parser.add_argument("-use_reorder", action="store_true", help="Keep a flat copy and rescore candidates.")
parser.add_argument("-csv", type=str, help="Optional path to append results as CSV rows.")
args = parser.parse_args()


def mmap_sparse_matrix_fields(fname):
    """mmap the fields of a CSR matrix without instanciating it"""
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype="int64", count=3)
        nrow, ncol, nnz = sizes
    ofs = sizes.nbytes
    indptr = np.memmap(fname, dtype="int64", mode="r", offset=ofs, shape=nrow + 1)
    ofs += indptr.nbytes
    indices = np.memmap(fname, dtype="int32", mode="r", offset=ofs, shape=nnz)
    ofs += indices.nbytes
    data = np.memmap(fname, dtype="float32", mode="r", offset=ofs, shape=nnz)
    return data, indices, indptr, ncol


def knn_result_read(fname):
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    f = open(fname, "rb")
    f.seek(4 + 4)
    I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return I, D


def to_vsag_csr(data, indices, indptr):
    """BigANN CSR (int64 indptr, int32 indices) -> vsag CSR (uint32 indptr, uint32 indices)."""
    nnz = int(indptr[-1])
    if nnz > np.iinfo(np.uint32).max:
        raise ValueError(f"nnz={nnz} overflows the uint32 index_pointers vsag requires.")
    return (
        np.ascontiguousarray(indptr, dtype=np.uint32),
        np.ascontiguousarray(indices, dtype=np.uint32),
        np.ascontiguousarray(data, dtype=np.float32),
    )


print(f"Loading base vectors from {args.input}...")
base_data, base_indices, base_indptr, _ = mmap_sparse_matrix_fields(args.input)
b_indptr, b_indices, b_values = to_vsag_csr(base_data, base_indices, base_indptr)

n = len(b_indptr) - 1
ids = np.arange(n, dtype=np.int64)

max_nnz = int(np.diff(b_indptr.astype(np.int64)).max())
term_id_limit = int(b_indices.max()) + 1
print(f"n = {n}, nnz = {len(b_indices)}, max nnz/vec = {max_nnz}, term_id_limit = {term_id_limit}")

index_params = json.dumps(
    {
        "dtype": "sparse",
        "metric_type": "ip",
        "dim": max_nnz,
        "index_param": {
            "term_id_limit": term_id_limit,
            "window_size": args.window_size,
            "doc_prune_ratio": args.doc_prune_ratio,
            "use_quantization": args.use_quantization,
            "use_reorder": args.use_reorder,
        },
    }
)
print(f"Index params: {index_params}")

index = pyvsag.Index("sindi", index_params)

t0 = time.time()
print("Building index...")
index.build(index_pointers=b_indptr, indices=b_indices, values=b_values, ids=ids)
t1 = time.time()
indexing_time = t1 - t0
print(f"Index built in {indexing_time:.2f} seconds.")

if args.output:
    index.save(args.output)
    print(f"Index saved to {args.output}")

print(f"Loading queries from {args.query}...")
q_data, q_indices, q_indptr, _ = mmap_sparse_matrix_fields(args.query)
q_indptr, q_indices, q_values = to_vsag_csr(q_data, q_indices, q_indptr)
n_queries = len(q_indptr) - 1

I, _ = knn_result_read(args.gt)
k = args.k

def run(n_candidate, query_prune_ratio):
    params = json.dumps(
        {"sindi": {"n_candidate": n_candidate, "query_prune_ratio": query_prune_ratio}}
    )
    start = time.time()
    res, _ = index.knn_search(q_indptr, q_indices, q_values, k, params)
    elapsed = time.time() - start
    return res, elapsed


print("Warm-up pass...")
run(10 * k, args.query_prune_ratio)

sweep = [
    (n_candidate * k // 10, qpr)
    for qpr in [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    for n_candidate in [20, 50, 100, 200]
]

rows = []
for n_candidate, query_prune_ratio in sweep:
    print(f"Setting n_candidate={n_candidate}, query_prune_ratio={query_prune_ratio}...")
    res, elapsed = run(n_candidate, query_prune_ratio)

    intersection_sizes = np.array(
        [np.intersect1d(row1, row2).size for row1, row2 in zip(I[:, :k], res)]
    )
    recall = np.sum(intersection_sizes) / (I.shape[0] * k)
    latency_us = elapsed / n_queries * 1e6
    qps_1t = n_queries / elapsed

    rr = 0.0
    for gt, pred in zip(I[:, :k], res):
        gt_set = set(gt.tolist())
        for rank, item in enumerate(pred[:k], start=1):
            if item in gt_set:
                rr += 1.0 / rank
                break
    rr /= I.shape[0]

    print(f"Elapsed: {elapsed:.4f}s; single-thread {qps_1t:.2f} QPS; {latency_us:.2f} us/query")
    print(f"Recall@{k}: {recall * 100:.4f}")
    print(f"RR@{k} (vs exact-NN gt, not qrels): {rr:.4f}\n")

    rows.append(
        (
            "SINDI",
            recall,
            indexing_time,
            latency_us,
            elapsed,
            qps_1t,
            rr,
            args.doc_prune_ratio,
            query_prune_ratio,
            n_candidate,
        )
    )

if args.csv:
    import csv
    import os

    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                [
                    "Model",
                    "Recall",
                    "Indexing Time",
                    "Single Query Time (microseconds)",
                    "Searching Time (Seconds)",
                    "QPS (single-thread)",
                    "RR@10 (vs exact-NN gt)",
                    "doc_prune_ratio",
                    "query_prune_ratio",
                    "n_candidate",
                ]
            )
        w.writerows(rows)
    print(f"Results appended to {args.csv}")
