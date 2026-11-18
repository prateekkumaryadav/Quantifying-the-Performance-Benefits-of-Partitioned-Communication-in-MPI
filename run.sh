mpicc -O3 -Xpreprocessor -fopenmp \
  -I$(brew --prefix libomp)/include \
  -L$(brew --prefix libomp)/lib -lomp -Wl,-w \
  pcomm_m4_bench.c -o pcomm_m4_bench

mpirun -np 2 --bind-to none ./pcomm_m4_bench