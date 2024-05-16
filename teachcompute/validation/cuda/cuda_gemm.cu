#include "teachcompute_helpers.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

namespace cuda_example {

enum TransType { FalseFalse = 0, FalseTrue = 1, TrueFalse = 2, TrueTrue = 3 };

#define BLOCK_SIZE 32
#define CEIL_DIV(N, DEN) ((N + (DEN - 1)) / DEN)

template <typename T, TransType trans_type>
__global__ void matmul_v1(int M, int N, int K, T *A, T *B, T *C) {

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (trans_type == TransType::FalseFalse) {
    if (x < M && y < N) {
      T tmp = 0.0;
      for (int i = 0; i < K; ++i) {
        tmp += A[x * K + i] * B[i * N + y];
      }
      C[x * N + y] += tmp;
    }
  }
}

template <typename T>
void _matmul_v1(int n_rows1, int n_cols1, T *A, int n_rows2, int n_cols2, T *B,
                T *C, bool transA, bool transB) {
  int M, N, K;
  TransType tt;
  if (transA) {
    if (transB) {
      tt = TransType::TrueTrue;
    } else {
      tt = TransType::FalseTrue;
    }
  } else {
    if (transB) {
      tt = TransType::TrueFalse;
    } else {
      EXT_ENFORCE(n_cols1 == n_rows2, "Dimensions do not match.");
      M = n_rows1;
      N = n_cols2;
      K = n_cols1;
    }
  }
  dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE), 1);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
  switch (tt) {
  case TransType::FalseFalse:
    matmul_v1<T, TransType::FalseFalse>
        <<<gridDim, blockDim>>>(M, N, K, A, B, C);
    break;
  default:
    throw std::runtime_error("Not implemented yet.");
  }
}

void matmul_v1(int n_rows1, int n_cols1, float *A, int n_rows2, int n_cols2,
               float *B, float *C, bool transA, bool transB) {
  _matmul_v1(n_rows1, n_cols1, A, n_rows2, n_cols2, B, C, transA, transB);
}

} // namespace cuda_example
