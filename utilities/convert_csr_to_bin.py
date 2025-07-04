import struct
import numpy as np
from scipy.sparse import csr_matrix
import argparse

parser = argparse.ArgumentParser(description="Convert .csr files to .bin files.")
parser.add_argument("-file_path", type=str)
parser.add_argument("-output_path", type=str)
args = parser.parse_args()

file_path = args.file_path
output_path = args.output_path

def convertz_csr_to_bin(csr_file_path, bin_file_path):
    with open(csr_file_path, "rb") as f:
        # Read header: 3 uint64 values
        n_rows = np.fromfile(f, dtype=np.uint64, count=1)[0]
        n_cols = np.fromfile(f, dtype=np.uint64, count=1)[0]
        nnz = np.fromfile(f, dtype=np.uint64, count=1)[0]

        print(f"n_rows = {n_rows}, n_cols = {n_cols}, nnz = {nnz}")

        # Read indptr: (n_rows + 1) uint64
        indptr = np.fromfile(f, dtype=np.uint64, count=n_rows + 1)

        # Read indices: nnz uint32
        indices = np.fromfile(f, dtype=np.uint32, count=nnz)

        # Read data: nnz float32
        data = np.fromfile(f, dtype=np.float32, count=nnz)

    # Reconstruct CSR matrix
    csr = csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))

    # Write to binary file in specified format
    with open(bin_file_path, 'wb') as out:
        # Write total number of vectors (rows) as uint32
        out.write(struct.pack('<I', csr.shape[0]))

        for i in range(csr.shape[0]):
            start = csr.indptr[i]
            end = csr.indptr[i + 1]
            row_indices = csr.indices[start:end]
            row_data = csr.data[start:end]

            # Write number of nonzero components
            out.write(struct.pack('<I', len(row_indices)))

            # Write indices (uint32)
            out.write(struct.pack('<' + 'I' * len(row_indices), *row_indices))

            # Write values (float32)
            out.write(struct.pack('<' + 'f' * len(row_data), *row_data))

convert_csr_to_bin(file_path, output_path)
 