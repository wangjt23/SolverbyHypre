import numpy as np
import scipy
from scipy.sparse import csr_matrix
import scipy.sparse

def write_csr_for_hypre(csr_matrix, filename):
    with open(filename, 'w') as f:
        f.write(f"{csr_matrix.shape[0]}\n")
        for i in csr_matrix.indptr:
            f.write(f"{i+1}\n")  # +1 是因为HYPRE使用1-based索引
        for j in csr_matrix.indices:
            f.write(f"{j+1}\n")  # +1 是因为HYPRE使用1-based索引
        for d in csr_matrix.data:
            f.write(f"{d}\n")

def write_vector_for_hypre(vector, filename):
    with open(filename, 'w') as f:
        f.write(f"{vector.size}\n")
        for value in vector:
            f.write(f"{value:.16e}\n")  # 使用科学计数法，保留16位小数

if __name__ == "__main__":

    data = scipy.sparse.load_npz('./data/csr1.npz')
    if data.format == 'csr':
        print(data.shape)
        write_csr_for_hypre(data, './data/csr1.hpcsr')
    else:
        raise ValueError("not csr matrix")

    # test vector 
    vec = np.load('./data/b1.npy')
    print(vec.size)
    write_vector_for_hypre(vec, './data/b1.hpcsr')