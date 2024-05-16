#pragma once

namespace cuda_example {

int matmul_v1(int n_rows1, int n_cols1, const float *A, int n_rows2,
              int n_cols2, const float *B, float *C, bool transA, bool transB);

} // namespace cuda_example
