# Solve large-scale sparse matrix linear equations in parallel using hypre

## Required data format

A: *.npz file which contains a scipy.sparse.csr matrix

b: *.npy file wihch is a numpy.ndarray in 1D

## key scripts

- csr_py2hypre.py: transform A&b's format from `scipy.sparse.csr`/`np.ndarray `to `HYPRE_CSRMatrix`/ `HYPRE_Vector `, make sure Hypre's buildin function `BuildParFromOneFile` can load data correctlly.
- mysolver.c: contains AMG, PCG+AMG, Gmres+AMG, ILU...  to solve input systems.
  - `-solver <ID>`: Solver ID
    - `0`: AMG (default)
    - `1`: PCG-AMG
    - `3`: GMRES-AMG
    - `80`: ILU
  - `-matPath <FILEDIR>`: CSR matrix file path.
  - `-rhsPath <FILEDIR>`: RHS vector file path.
  - `-maxIters <n>`: Set solver's maximum iterations (default: 100).
  - `-tol <value>`: Set solver's tolerance (default: 1e-6).
  - `-SolPath <FILEDIR>` : whether save solution in one file(given the file dir)(default: not)
- ij.c: source code in HYPRE, contains a large number of solver templates

## Requirements

- python3.9+
  - scipy
  - numpy
- c
  - HYPRE
  - openmpi/mpich

## Usage

### transform data

1. change  the proper data loaction of yours in `csr_py2hypre.py`.
2. run

```bash
python csr_py2hypre.py
```

### Configure Makefile

configure you own mpi&hypre path in `Makefile`

```makefile
CC        = mpicc
HYPRE_DIR = /usr/local/hypre
```

### Complie

```bash
make mysolver
```

### run

```bash
mpirun -np 4 ./mysolver -matPath ./data/csr1.hpcsr -rhsPath ./data/b1.hpcsr -solver 3 -maxIters 10 -tol 1e-3 -SolPath ./data/x.sol
```

## update

[2024.8.28] first commit

[2024.10.22] fix bugs that hypre can not load scipy csr format matrix correctly(since hypre's HYPRE_ParCSRMatrix format place the diag element to the first poisition at each row).
