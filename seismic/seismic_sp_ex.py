from seismic import SeismicIndex
import numpy as np
import json
import ir_datasets
import ir_measures
from ir_measures import *
import time
import argparse

parser = argparse.ArgumentParser(description="Run Seismic index example.")
parser.add_argument("-input", type=str, help="Path to the JSON input file.")
parser.add_argument("-query", type=str, help="Path to the query file.")
args = parser.parse_args()

# Splade embeddings for MSMARCO passage.
json_input_file = args.input

# Build Seismic index.
t0 = time.time()
index = SeismicIndex.build(json_input_file)
t1 = time.time()
print(f"Index built in {t1 - t0:.2f} seconds.")
print("Number of documents:", index.len)
print("Avg number of non-zero components:", index.nnz / index.len)
print("Dimensionality of the vectors:", index.dim)

index.print_space_usage_byte()

# Load queries from a TSV file.
queries_path = args.query

print("Loading queries...")
count = 0
queries = []
with open(queries_path, 'r') as f:
    print("Reading queries from file...")
    for line in f:
        queries.append(json.loads(line))
        count += 1
        if count >= 3000:
            break

print(f"Number of queries: {len(queries)}")

MAX_TOKEN_LEN = 30
string_type  = f'U{MAX_TOKEN_LEN}'

queries_ids = np.array([q['id'] for q in queries], dtype=string_type)

query_components = []
query_values = []

print("Preparing queries for batch search...")

# Perform batch query search.
t2 = time.time()
for query in queries:
    vector = query['vector']
    query_components.append(np.array(list(vector.keys()), dtype=string_type))
    query_values.append(np.array(list(vector.values()), dtype=np.float32))

    results = index.batch_search(
    queries_ids=queries_ids,
        query_components=query_components,
        query_values=query_values,
        k=10,
        query_cut=20,
        heap_factor=0.7,
        sorted=True,
        n_knn=0,
    )
t3 = time.time()
print(f"Search completed in {t3 - t2:.2f} seconds.")

# Process results and calculate Reciprocal Rank at 10 (RR@10).
ir_results = [ir_measures.ScoredDoc(query_id, doc_id, score) for r in results for (query_id, score, doc_id) in r]
qrels = ir_datasets.load('msmarco-passage/dev/small').qrels

# rr10 = ir_measures.calc_aggregate([RR@10], qrels, ir_results)
# print(f"RR@10: {rr10:.4f}")