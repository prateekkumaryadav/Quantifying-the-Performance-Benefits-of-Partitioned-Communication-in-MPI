# mpicc -O3 -Xpreprocessor -fopenmp \
#   -I$(brew --prefix libomp)/include \
#   -L$(brew --prefix libomp)/lib -lomp -Wl,-w \
#   pcomm_m4_bench.c -o pcomm_m4_bench

# mpirun -np 2 --bind-to none ./pcomm_m4_bench


# brew install libomp
mpicc -O2 -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include \
  pcomm_bench.c -o pcomm_bench -L$(brew --prefix libomp)/lib -lomp -lm
export OMP_WAIT_POLICY=active
mpirun --bind-to none -np 2 ./pcomm_bench 4 8     # threads=4, theta for Exp 1 = 8