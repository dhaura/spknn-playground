import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)
import pytest
import os
from tempfile import NamedTemporaryFile, mkdtemp

from sparse_neighbors_search import MinHash
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz

import argparse
import time

argparse = argparse.ArgumentParser(description='Run MinHash on a sparse matrix.')
argparse.add_argument('-input', type=str, required=True, help='Path to the input sparse matrix file in .npz format.')
argparse.add_argument('-num_threads', type=int, default=4, help='Number of threads to use for computation.')
args = argparse.parse_args()

def test_minHash():
    neighborhood_matrix = load_npz(args.input)
    minHash_object = MinHash(n_neighbors=500, number_of_hash_functions=20, number_of_cores=args.num_threads,
                             shingle_size=5, fast=True, maxFeatures=int(max(neighborhood_matrix.getnnz(1))), 
                             absolute_numbers=False, max_bin_size=100000, minimal_blocks_in_common=400, 
                             excess_factor=1, prune_inverse_index=False)

    minHash_object.fit(neighborhood_matrix)

    knn_graph = minHash_object.kneighbors_graph(mode='distance')

t0 = time.time()
test_minHash()
t1 = time.time()
print(f"Time taken for MinHash: {t1 - t0} seconds")
