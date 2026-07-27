import pyanns
import numpy as np
import argparse
import csv
import os
import time

print("Before arg parser")
parser = argparse.ArgumentParser(description="Run kNN search on a PyANNS index.")
parser.add_argument("-n", type=int, default=100000, help="Number of elements in the index.")
parser.add_argument("-nq", type=int, default=6980, help="Number of queries.")
parser.add_argument("-input", type=str, help="Path to the input CSR file.")
parser.add_argument("-query", type=str, help="Path to the query CSR file.")
parser.add_argument("-gt", type=str, help="Path to the ground truth file.")
parser.add_argument("-output", type=str, help="Path to the output directory.")
parser.add_argument("-csv", type=str, help="Optional path to append results as CSV rows.")
args = parser.parse_args()

print("After arg parser")
n = args.n
nq = args.nq
input_path = args.input
query_path = args.query
gt_path = args.gt
output_path = args.output
num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))


print("Initializing index with n =", n)

t0 = time.time()
searcher = pyanns.SparseGrapSearcher(input_path, output_path)
t1 = time.time()

print(f"Index initialized in {t1 - t0:.2f} seconds.")


searcher.set_ef(80)

def knn_result_read(fname):
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    f = open(fname, "rb")
    f.seek(4+4)
    I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return I, D
  
def mmap_sparse_matrix_fields(fname):
    """ mmap the fields of a CSR matrix without instanciating it """
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
    ofs = sizes.nbytes
    indptr = np.memmap(fname, dtype='int64', mode='r', offset=ofs, shape=nrow + 1)
    ofs += indptr.nbytes
    indices = np.memmap(fname, dtype='int32', mode='r', offset=ofs, shape=nnz)
    ofs += indices.nbytes
    data = np.memmap(fname, dtype='float32', mode='r', offset=ofs, shape=nnz)
    return data, indices, indptr, ncol

data, indices, indptr, _ = mmap_sparse_matrix_fields(query_path)

I, _ = knn_result_read(gt_path)

def rr_at_k(gt, pred, k=10):
    total = 0.0
    for g, p in zip(gt, pred):
        g_set = set(g.tolist())
        for rank, item in enumerate(p[:k], start=1):
            if item in g_set:
                total += 1.0 / rank
                break
    return total / gt.shape[0]

print("Warm-up pass...")
searcher.search_batch(nq, indptr, indices, data, 10, 0.1)

rows = []
for budget in [0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01]:
    print(f"Setting budget={budget}...")
    start = time.time()
    res = searcher.search_batch(nq, indptr, indices, data, 10, budget).reshape(-1, 10)
    elapsed = time.time() - start

    intersection_sizes = np.array([np.intersect1d(row1, row2).size for row1, row2 in zip(I, res)])
    recall = np.sum(intersection_sizes) / (I.shape[0] * I.shape[1])
    qps = I.shape[0] / elapsed
    rr = rr_at_k(I, res, 10)
    print(f'Elapsed time: {elapsed}; {qps:.2f} QPS')
    print(f'Recall: {recall * 100:.4f}')
    print(f'RR@10: {rr:.4f}\n')
    rows.append(("PyANNS", num_threads, budget, recall, t1 - t0, elapsed, qps, rr))

if args.csv:
    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                ["Model", "Threads", "budget", "Recall",
                 "Indexing Time", "Searching Time (Seconds)", "QPS", "RR@10"]
            )
        w.writerows(rows)
    print(f"Results appended to {args.csv}")

