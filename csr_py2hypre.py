import numpy as np
import scipy
from scipy.sparse import csr_matrix
import scipy.sparse

def write_csr_for_hypre(csr_matrix, filename):
    rows = csr_matrix.shape[0]
    nnz = csr_matrix.nnz
    with open(filename, 'w') as f:
        f.write(f"{rows}\n")
        indptr = csr_matrix.indptr
        for count,i in enumerate(indptr):
            # HYPRE use 1-based index in csr file loading so need plus 1
            f.write(f"{i+1}\n")  
            if (count % 100000) == 0:
                print(f"1.Reading indptr at [{count}/{rows+1}]")
        
        for i in range(rows):
            if (count % 100000) == 0:
                print(f"2.Reading indices at [{count}/{rows}]")
            # Hypre place diagnal element to the first place in csr record
            # HYPRE use 1-based index in csr file loading so need plus 1
            tmp_indices = csr_matrix.indices[indptr[i] : indptr[i+1]]
            if i in tmp_indices:
                f.write(f"{i+1}\n")
                tmp_indices = tmp_indices[tmp_indices != i]
            for ind in tmp_indices:
                f.write(f"{ind+1}\n")

        for i in range(rows):
            if (count % 100000) == 0:
                print(f"3.Reading data at [{count}/{rows}]")
            # Hypre place diagnal element to the first place in csr record
            # HYPRE use 1-based index in csr file loading so need plus 1
            tmp_indices = csr_matrix.indices[indptr[i] : indptr[i+1]]
            tmp_data = csr_matrix.data[indptr[i] : indptr[i+1]]
            if i in tmp_indices:
                diag_index = np.where(tmp_indices == i)[0][0]
                f.write(f"{tmp_data[diag_index]}\n")
                tmp_data = np.delete(tmp_data, diag_index)
            for ind in tmp_data:
                f.write(f"{ind}\n")

def write_vector_for_hypre(vector, filename):
    vec_len = vector.size
    with open(filename, 'w') as f:
        f.write(f"{vec_len}\n")
        for count,value in enumerate(vector):
            f.write(f"{value:.16e}\n")
            if (count % 100000) == 0:
                print(f"4.Reading vec at [{count}/{vec_len}]")

if __name__ == "__main__":

    data = scipy.sparse.load_npz('./data/csr1.npz')
    if data.format == 'csr':
        print("Start Reading matrix with shape: ", data.shape)
        write_csr_for_hypre(data, './data/csr1.hpcsr')
    else:
        raise ValueError("not csr matrix")

    vec = np.load('./data/b1.npy')
    # vec = np.ones(8)
    print(print("Start Reading matrix with shape: ", data.shape),vec.shape)
    write_vector_for_hypre(vec, './data/b1.hpcsr')