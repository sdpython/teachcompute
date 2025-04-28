#include "mmat_impl.h"

#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

typedef vector<double> Matrix;

// Accès à un élément (i, j)
template <typename DTYPE> inline DTYPE &at(DTYPE *p, int cols, int i, int j) {
  return p[i * cols + j];
}

// Multiplication de matrices plates par blocs
template <typename DTYPE>
void BlockMatrixMultiply(const DTYPE *A, const DTYPE *B, DTYPE *C, int n, int m,
                         int p, int block_size) {
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < n; i += block_size) {
    for (int j = 0; j < p; j += block_size) {
      for (int k = 0; k < m; k += block_size) {
        // Sous-bloc
        for (int ii = i; ii < min(i + block_size, n); ++ii) {
          for (int jj = j; jj < min(j + block_size, p); ++jj) {
            double sum = 0.0;
            for (int kk = k; kk < min(k + block_size, m); ++kk) {
              sum += at(A, m, ii, kk) * at(B, p, kk, jj);
            }
            at(C, p, ii, jj) += sum;
          }
        }
      }
    }
  }
}

void mmat_impl_cpp(int n_row, int n_col, int k, const float *p1,
                   const float *p2, float *res, int block_size) {
  BlockMatrixMultiply(p1, p2, res, n_row, k, n_col, block_size);
}

void mmat_impl_cpp(int n_row, int n_col, int k, const double *p1,
                   const double *p2, double *res, int block_size) {
  BlockMatrixMultiply(p1, p2, res, n_row, k, n_col, block_size);
}
