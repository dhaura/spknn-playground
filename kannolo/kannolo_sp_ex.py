from kannolo import SparsePlainHNSW
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description="Run Seismic index example.")
parser.add_argument("-input", type=str, help="Path to the JSON input file.")
parser.add_argument("-query", type=str, help="Path to the query file.")
parser.add_argument("-index", type=str, help="Path to save the index file.")
args = parser.parse_args()

input_file = args.input
query_file = args.query
index_path = args.index

d = 30522
efConstruction = 200
m = 32 # n. neighbors per node
metric = "ip" # Inner product. Alternatively, you can use "l2" for squared L2 metric

"""
Binary File Format:
- First 4 bytes: Unsigned 32-bit integer (little-endian) indicating the total number of sparse vectors.
- For each vector:
    - 4 bytes: Unsigned 32-bit integer (little-endian) representing the number of nonzero components.
    - Next (4 * n) bytes: Array of n unsigned 32-bit integers (little-endian) for component indices (cast to int32).
    - Following (4 * n) bytes: Array of n 32-bit floating point values (little-endian) for the nonzero components.
"""

t0 = time.time()
print("Building index...")
index = SparsePlainHNSW.build_from_file(input_file, d, m, efConstruction, metric)
t1 = time.time()
print(f"Index built in {t1 - t0:.2f} seconds.")

index.save(index_path)

k = 10 # Number of results to be retrieved
efSearch = 200 # Search parameter for regulating the accuracy

t2 = time.time()
print("Processing queries...")
dists, ids = index.search_batch(query_file, d, k, efSearch)
t3 = time.time()
print(f"Search completed in {t3 - t2:.2f} seconds.")
