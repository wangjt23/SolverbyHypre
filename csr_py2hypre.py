import numpy as np
import scipy
from scipy.sparse import csr_matrix
import scipy.sparse

def write_csr_for_hypre(csr_matrix, filename):
    rows = csr_matrix.shape[0]
    nnz = csr_matrix.nnz
    with open(filename, 'w') as f:
        f.write(f"{rows}\n")
        for count,i in enumerate(csr_matrix.indptr):
            f.write(f"{i+1}\n")  # +1 是因为HYPRE使用1-based索引
            if (count % 1000000) == 0:
                print(f"1.Reading indptr at [{count}/{rows+1}]")
        for count,j in enumerate(csr_matrix.indices):
            f.write(f"{j+1}\n")  # +1 是因为HYPRE使用1-based索引
            if (count % 1000000) == 0:
                print(f"2.Reading indices at [{count}/{nnz}]")
        for count,d in enumerate(csr_matrix.data):
            f.write(f"{d}\n")
            if (count % 1000000) == 0:
                print(f"3.Reading data at [{count}/{nnz}]")

def write_vector_for_hypre(vector, filename):
    vec_len = vector.size
    with open(filename, 'w') as f:
        f.write(f"{vec_len}\n")
        for count,value in enumerate(vector):
            f.write(f"{value:.16e}\n")  # 使用科学计数法，保留16位小数
            if (count % 1000000) == 0:
                print(f"4.Reading vec at [{count}/{vec_len}]")

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