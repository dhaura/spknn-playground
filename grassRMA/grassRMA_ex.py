import sparse_hnswlib
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description="Run kNN search on a sparse HNSW index.")
parser.add_argument("-n", type=int, default=100000, help="Number of elements in the index.")
parser.add_argument("-d", type=int, default=16, help="Dimensionality of the vectors.")
parser.add_argument("-num_threads", type=int, default=8, help="Number of threads to use for parallel processing.")
parser.add_argument("-input", type=str, help="Path to the input CSR file.")
parser.add_argument("-query", type=str, help="Path to the query CSR file.")
parser.add_argument("-gt", type=str, help="Path to the ground truth file.")
parser.add_argument("-output", type=str, help="Path to the output directory.")
args = parser.parse_args()

n = args.n
d = args.d
num_threads = args.num_threads
input_path = args.input
query_path = args.query
gt_path = args.gt
output_path = args.output

print("Initializing index with n =", n)

t0 = time.time()
p = sparse_hnswlib.Index(space="ip", dim=d)
p.init_index(
    max_elements=n,
    csr_path=input_path,
    ef_construction=1000,
    M=16,
)
p.add_items(num_threads=num_threads)
t1 = time.time()
print(f"Index initialized in {t1 - t0:.2f} seconds.")

print("Saving index to disk...")
p.save_index(output_path)
print("Index saved.")

# print("Loading index from disk...")
# p.load_index(output_path, n)
# print("Index loaded.")

p.set_ef(48)

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

print("Running kNN query...")
start = time.time()
res, distances = p.knn_query(indptr, indices, data, k=10, num_threads=num_threads)
end = time.time()
print("kNN query completed.")

elapsed = end - start
intersection_sizes = np.array([np.intersect1d(row1, row2).size for row1, row2 in zip(I, res)])
print(f'Elapsed time: {elapsed}; {round(I.shape[0] /elapsed, 2)} QPS')
print(f'Recall: {np.sum(intersection_sizes) / (I.shape[0] * I.shape[1]) * 100}')
