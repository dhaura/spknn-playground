from kannolo import SparsePlainHNSW
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description="Run Seismic index example.")
parser.add_argument("-query", type=str, help="Path to the query file.")
parser.add_argument("-index", type=str, help="Path to save the index file.")
args = parser.parse_args()

query_file = args.query
index_path = args.index

"""
Binary File Format:
- First 4 bytes: Unsigned 32-bit integer (little-endian) indicating the total number of sparse vectors.
- For each vector:
    - 4 bytes: Unsigned 32-bit integer (little-endian) representing the number of nonzero components.
    - Next (4 * n) bytes: Array of n unsigned 32-bit integers (little-endian) for component indices (cast to int32).
    - Following (4 * n) bytes: Array of n 32-bit floating point values (little-endian) for the nonzero components.
"""

t0 = time.time()
print("Loading index...")
index = SparsePlainHNSW.load(index_path)
t1 = time.time()
print(f"Index loaded in {t1 - t0:.2f} seconds.")

index.save(index_path)

d = 30522
k = 10 # Number of results to be retrieved
efSearch = 200 # Search parameter for regulating the accuracy

t2 = time.time()
print("Processing queries...")
dists, ids = index.search_batch(query_file, d, k, efSearch)
t3 = time.time()
print(f"Search completed in {t3 - t2:.2f} seconds.")
