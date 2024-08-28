# Solve large-scale sparse matrix linear equations in parallel using hypre
## Required data format
A: *.npz file which contains a scipy.sparse.csr matrix 

b: *.npy file wihch is a numpy.ndarray in 1D

## key scripts
- csr_py2hypre.py: transform A&b's format to make sure Hypre's func `BuildParFromOneFile` can load.
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
- ij.c: source code in HYPRE, contains a large number of solver templates
## Requirements
- python3.9+
    - scipy
    - numpy
- c
    - HYPRE
    - openmpi
## Usage
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
mpirun -np 16 ./mysolver -matPath ./data/csr1.hpcsr -rhsPath ./data/b1.hpcsr -solver 3 -maxIters 10 -tol 1e-3
```
## update
[2024.8.28] Add mysolver.c, 
