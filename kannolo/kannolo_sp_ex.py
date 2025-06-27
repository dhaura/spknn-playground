from kannolo import SparsePlainHNSW
import numpy as np
import argparse
import time

def compute_accuracy(output_file, gt_file):
    # if files are csv
    if gt_file.endswith(".csv") or gt_file.endswith(".tsv"):
        column_names = ["query_id", "doc_id", "rank", "score"]
        if gt_file.endswith(".csv"):
            gt_pd = pd.read_csv(gt_file, sep=',', names=column_names)
            res_pd = pd.read_csv(output_file, sep=',', names=column_names)
        else:
            gt_pd = pd.read_csv(gt_file, sep='\t', names=column_names)
            res_pd = pd.read_csv(output_file, sep='\t', names=column_names)

        # Group both dataframes by 'query_id' and get unique 'doc_id' sets
        gt_pd_groups = gt_pd.groupby('query_id')['doc_id'].apply(set)
        res_pd_groups = res_pd.groupby('query_id')['doc_id'].apply(set)

        # Compute the intersection size for each query_id in both dataframes
        intersections_size = {
            query_id: len(gt_pd_groups[query_id] & res_pd_groups[query_id]) if query_id in res_pd_groups else 0
            for query_id in gt_pd_groups.index
        }

        # Computes total number of results in the groundtruth
        total_results = len(gt_pd)
        total_intersections = sum(intersections_size.values())

    elif gt_file.endswith(".npy"):
        # Read csv results and transform to numpy array
        column_names = ["query_id", "doc_id", "rank", "score"]
        res_pd = pd.read_csv(output_file, sep='\t', names=column_names)
        res_npy = res_pd['doc_id'].to_numpy()
        # Group results by query id and transform to npy array with shape (num_queries, num_results)
        res_npy = res_npy.reshape(-1, res_pd.groupby('query_id').size().max())
        k = res_npy.shape[1]

        # Read npy groundtruth
        doc_ids = np.load(gt_file, allow_pickle=True)
        
        # compute total results and total intersections
        total_results = res_npy.shape[0] * res_npy.shape[1]
        total_intersections = 0
        for i in range(res_npy.shape[0]):
            total_intersections += len(np.intersect1d(res_npy[i], doc_ids[i][:k]))
    else:
        raise ValueError("Groundtruth file must be in csv, tsv or numpy format!!!")
        
    return round((total_intersections/total_results) * 100, 3)

def compute_metric(output_file, gt_file, query_ids_path, doc_ids_path, qrels_path, metric): 

    if metric == None or metric == "":
        print("No metric specified. Skipping evaluation.")
        return None

    column_names = ["query_id", "doc_id", "rank", "score"]
    gt_pd = pd.read_csv(gt_file, sep='\t', names=column_names)
    res_pd = pd.read_csv(output_file, sep='\t', names=column_names)
    
    queries_ids = np.load(query_ids_path, allow_pickle=True)
    doc_ids = np.load(doc_ids_path, allow_pickle=True)

    gt_pd['query_id'] = gt_pd['query_id'].apply(lambda x: queries_ids[x])
    res_pd['query_id'] = res_pd['query_id'].apply(lambda x: queries_ids[x])
    
    gt_pd['doc_id'] = gt_pd['doc_id'].apply(lambda x: doc_ids[x])
    res_pd['doc_id'] = res_pd['doc_id'].apply(lambda x: doc_ids[x])
    
    
    df_qrels = pd.read_csv(qrels_path, sep="\t", names=["query_id", "useless", "doc_id", "relevance"])
    if len(pd.unique(df_qrels['useless'])) != 1:
        df_qrels = pd.read_csv(qrels_path, sep="\t", names=["query_id", "doc_id", "relevance", "useless"])

    gt_pd['doc_id'] = gt_pd['doc_id'].astype(df_qrels.doc_id.dtype)
    res_pd['doc_id'] = res_pd['doc_id'].astype(df_qrels.doc_id.dtype)
    
    gt_pd['query_id'] = gt_pd['query_id'].astype(df_qrels.query_id.dtype)
    res_pd['query_id'] = res_pd['query_id'].astype(df_qrels.query_id.dtype)
    
    ir_metric = ir_measures.parse_measure(metric)
    
    metric_val = ir_measures.calc_aggregate([ir_metric], df_qrels, res_pd)[ir_metric]
    metric_gt = ir_measures.calc_aggregate([ir_metric], df_qrels, gt_pd)[ir_metric]
    
    print(f"Metric of the run: {ir_metric}: {metric_val}")
    print(f"Metric of the gt : {ir_metric}: {metric_gt}")
    
    return metric_val

parser = argparse.ArgumentParser(description="Run Seismic index example.")
parser.add_argument("-input", type=str, help="Path to the JSON input file.")
parser.add_argument("-query", type=str, help="Path to the query file.")
parser.add_argument("-index", type=str, help="Path to save the index file.")
parser.add_argument("-gt", type=str, help="Path to the groundtruth file for accuracy computation.")
parser.add_argument("-query_ids", type=str, help="Path to the query ids file for evaluation.")
parser.add_argument("-doc_ids", type=str, help="Path to the document ids file for evaluation.")
parser.add_argument("-qrels", type=str, help="Path to the qrels file for evaluation.")
parser.add_argument("-output", type=str, help="Path to save the output file with results.")
args = parser.parse_args()

input_file = args.input
query_file = args.query
index_path = args.index
gt_file = args.gt
output_file = args.output
query_ids_path = args.query_ids
doc_ids_path = args.doc_ids
qrels_path = args.qrels

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

dists = dists.tolist()
ids = ids.tolist()

with open(output_file, 'w') as f:
    for i in range(len(ids)//k):
        for j in range(k):
            f.write(f"{i}\t{ids[i * k + j]}\t{j+1}\t{dists[i * k + j]}\n")

print(f"Results saved to {output_file}")

accuracy = compute_accuracy(output_file, gt_file)
print(f"Accuracy: {accuracy}%")

metric = "RR@10"
metric_val = compute_metric(output_file, gt_file, query_ids_path, doc_ids_path, qrels_path, metric)
print(f"{metric}: {metric_val}")
