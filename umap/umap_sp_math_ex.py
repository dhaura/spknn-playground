import numpy as np
import scipy.sparse
import sympy
import sklearn.datasets
import sklearn.feature_extraction.text
import umap
import umap.plot
import matplotlib.pyplot as plt

import time

primes = list(sympy.primerange(2, 110000))
prime_to_column = {p:i for i, p in enumerate(primes)}

# Prepare training data.
lil_matrix_rows = []
lil_matrix_data = []
for n in range(100000):
    prime_factors = sympy.primefactors(n)
    lil_matrix_rows.append([prime_to_column[p] for p in prime_factors])
    lil_matrix_data.append([1] * len(prime_factors))

factor_matrix = scipy.sparse.lil_matrix((len(lil_matrix_rows), len(primes)), dtype=np.float32)
factor_matrix.rows = np.array(lil_matrix_rows, dtype=object)
factor_matrix.data = np.array(lil_matrix_data, dtype=object)

# Prepare test data.
test_lil_matrix_rows = []
test_lil_matrix_data = []
for n in range(100000, 110000):
    prime_factors = sympy.primefactors(n)
    test_lil_matrix_rows.append([prime_to_column[p] for p in prime_factors])
    test_lil_matrix_data.append([1] * len(prime_factors))

test_data = scipy.sparse.lil_matrix((len(test_lil_matrix_rows), len(primes)), dtype=np.float32)
test_data.rows = np.array(test_lil_matrix_rows, dtype=object)
test_data.data = np.array(test_lil_matrix_data, dtype=object)

t0 = time.time()
umap_mapper = umap.UMAP(metric='cosine', random_state=42, low_memory=True, n_jobs=64)
embedding = umap_mapper.fit_transform(factor_matrix)
test_embedding = umap_mapper.transform(test_data)
t1 = time.time()

print(f"UMAP embedding completed in {t1 - t0:.2f} seconds.")

# umap.plot.points(embedding, values=np.arange(100000), theme='viridis')

# Plot UMAP embedding of data.
train_labels = np.zeros(embedding.shape[0])
test_labels = np.ones(test_embedding.shape[0])

all_embeddings = np.vstack([embedding, test_embedding])
all_labels = np.concatenate([train_labels, test_labels])

plt.figure(figsize=(10, 8))
plt.scatter(
    all_embeddings[all_labels == 0][:, 0], 
    all_embeddings[all_labels == 0][:, 1],
    c='blue', s=1, alpha=0.5, label='Train'
)
plt.scatter(
    all_embeddings[all_labels == 1][:, 0], 
    all_embeddings[all_labels == 1][:, 1],
    c='red', s=1, alpha=0.5, label='Test'
)
plt.legend(loc='upper right')
plt.axis('off')
plt.tight_layout()

plt.savefig("output/umap_math.png", dpi=300)
plt.close()




