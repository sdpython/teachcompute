#include "teachcompute_helpers.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

namespace cuda_example {

enum TransType { FalseFalse = 0, FalseTrue = 1, TrueFalse = 2, TrueTrue = 3 };

static void _set_mnk(int n_rows1, int n_cols1, int n_rows2, int n_cols2,
                     bool transA, bool transB, int &M, int &N, int &K,
                     TransType &tt) {
  if (transA) {
    if (transB) {
      tt = TransType::TrueTrue;
      EXT_ENFORCE(n_rows1 == n_cols2, "Dimensions do not match.");
      M = n_cols1;
      N = n_rows2;
      K = n_rows1;
    } else {
      tt = TransType::TrueFalse;
      EXT_ENFORCE(n_rows1 == n_rows2, "Dimensions do not match.");
      M = n_cols1;
      N = n_cols2;
      K = n_rows1;
    }
  } else {
    if (transB) {
      tt = TransType::FalseTrue;
      EXT_ENFORCE(n_cols1 == n_cols2, "Dimensions do not match.");
      M = n_rows1;
      N = n_rows2;
      K = n_cols1;
    } else {
      tt = TransType::FalseFalse;
      EXT_ENFORCE(n_cols1 == n_rows2, "Dimensions do not match.");
      M = n_rows1;
      N = n_cols2;
      K = n_cols1;
    }
  }
}

#define BLOCK_SIZE1 32
#define CEIL_DIV(N, DEN) ((N + (DEN - 1)) / DEN)

struct Access {
  __inline__ __device__ int get(int i, int j, int nc) { return i * nc + j; }
};

struct AccessT {
  __inline__ __device__ int get(int i, int j, int nc) { return j * nc + i; }
};

template <typename T, typename T1, typename T2>
__global__ void kernel_matmul_v1(int M, int N, int K, int nc1, int nc2,
                                 const T *A, const T *B, T *C) {

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    T tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[T1().get(x, i, nc1)] * B[T2().get(i, y, nc2)];
    }
    C[x * N + y] += tmp;
  }
}

template <typename T>
int _matmul_v1(int n_rows1, int n_cols1, const T *A, int n_rows2, int n_cols2,
               const T *B, T *C, bool transA, bool transB) {
  int M, N, K;
  TransType tt;
  _set_mnk(n_rows1, n_cols1, n_rows2, n_cols2, transA, transB, M, N, K, tt);
  dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE1), CEIL_DIV(N, BLOCK_SIZE1));
  dim3 blockDim(BLOCK_SIZE1, BLOCK_SIZE1);
  switch (tt) {
  case TransType::FalseFalse:
    kernel_matmul_v1<T, Access, Access>
        <<<gridDim, blockDim>>>(M, N, K, n_cols1, n_cols2, A, B, C);
    break;
  case TransType::TrueFalse:
    kernel_matmul_v1<T, AccessT, Access>
        <<<gridDim, blockDim>>>(M, N, K, n_cols1, n_cols2, A, B, C);
    break;
  case TransType::FalseTrue:
    kernel_matmul_v1<T, Access, AccessT>
        <<<gridDim, blockDim>>>(M, N, K, n_cols1, n_cols2, A, B, C);
    break;
  case TransType::TrueTrue:
    kernel_matmul_v1<T, AccessT, AccessT>
        <<<gridDim, blockDim>>>(M, N, K, n_cols1, n_cols2, A, B, C);
    break;
  default:
    EXT_THROW("Not implemented yet for trans*=", (int)tt, ".");
  }
  cudaDeviceSynchronize();
  return K;
}

int matmul_v1(int n_rows1, int n_cols1, const float *A, int n_rows2,
              int n_cols2, const float *B, float *C, bool transA, bool transB) {
  return _matmul_v1(n_rows1, n_cols1, A, n_rows2, n_cols2, B, C, transA,
                    transB);
}

///////
// TILE
///////

template <typename T, typename T1, typename T2, int TILE_ROW, int TILE_COL>
__global__ void kernel_matmul_v2(int M, int N, int K, int nc1, int nc2,
                                 const T *A, const T *B, T *C) {

  __shared__ float tile_A[TILE_ROW][TILE_COL];
  __shared__ float tile_B[TILE_ROW][TILE_COL];

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  T Cvalue = 0;
  T1 t1;
  T2 t2;

  for (int t = 0; t < K; t += TILE_ROW) {
    auto ind_a = t1.get(x, t + threadIdx.y, nc1);
    tile_A[threadIdx.x][threadIdx.y] = A[ind_a];
    auto ind_b = t2.get(t + threadIdx.x, y, nc2);
    tile_B[threadIdx.x][threadIdx.y] = B[ind_b];
    __syncthreads();

    for (int i = 0; i < TILE_ROW; ++i) {
      Cvalue += tile_A[threadIdx.x][i] * tile_B[i][threadIdx.y];
    }
    __syncthreads();
  }

  C[x * N + y] += Cvalue;
}

#define BLOCK_SIZE2 32
#define BLOCK_SIZE2_1 33

template <typename T>
int _matmul_v2(int n_rows1, int n_cols1, const T *A, int n_rows2, int n_cols2,
               const T *B, T *C, bool transA, bool transB) {
  int M, N, K;
  TransType tt;
  _set_mnk(n_rows1, n_cols1, n_rows2, n_cols2, transA, transB, M, N, K, tt);
  EXT_ENFORCE(n_rows1 % BLOCK_SIZE2 == 0 && n_cols1 % BLOCK_SIZE2 == 0 &&
                  n_rows2 % BLOCK_SIZE2 == 0 && n_cols2 % BLOCK_SIZE2 == 0,
              "_matmul_v2 only work with dimensions multiple 32.");

  dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE2), CEIL_DIV(N, BLOCK_SIZE2));
  dim3 blockDim(BLOCK_SIZE2, BLOCK_SIZE2);
  switch (tt) {
  case TransType::FalseFalse:
    kernel_matmul_v2<T, Access, Access, BLOCK_SIZE2, BLOCK_SIZE2_1>
        <<<gridDim, blockDim>>>(M, N, K, n_cols1, n_cols2, A, B, C);
    break;
  case TransType::TrueFalse:
    kernel_matmul_v2<T, AccessT, Access, BLOCK_SIZE2, BLOCK_SIZE2_1>
        <<<gridDim, blockDim>>>(M, N, K, n_cols1, n_cols2, A, B, C);
    break;
  case TransType::FalseTrue:
    kernel_matmul_v2<T, Access, AccessT, BLOCK_SIZE2, BLOCK_SIZE2_1>
        <<<gridDim, blockDim>>>(M, N, K, n_cols1, n_cols2, A, B, C);
    break;
  case TransType::TrueTrue:
    kernel_matmul_v2<T, AccessT, AccessT, BLOCK_SIZE2, BLOCK_SIZE2_1>
        <<<gridDim, blockDim>>>(M, N, K, n_cols1, n_cols2, A, B, C);
    break;
  default:
    EXT_THROW("Not implemented yet for trans*=", (int)tt, ".");
  }
  cudaDeviceSynchronize();
  return K;
}

int matmul_v2(int n_rows1, int n_cols1, const float *A, int n_rows2,
              int n_cols2, const float *B, float *C, bool transA, bool transB) {
  return _matmul_v2(n_rows1, n_cols1, A, n_rows2, n_cols2, B, C, transA,
                    transB);
}

} // namespace cuda_example
