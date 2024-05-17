#include "cuda_gemm.cuh"

#include "cuda_runtime.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace cuda_example;

PYBIND11_MODULE(cuda_gemm, m) {
  m.doc() = R"pbdoc(C++ experimental implementations with CUDA and GEMM.)pbdoc";

  m.def(
      "matmul_v1_cuda",
      [](int n_rows1, int n_cols1, int64_t A, int n_rows2, int n_cols2,
         int64_t B, int64_t C, bool transA, bool transB) -> int {
        return matmul_v1(n_rows1, n_cols1, (const float *)A, n_rows2, n_cols2,
                         (const float *)B, (float *)C, transA, transB);
      },
      py::arg("n_rows1"), py::arg("n_cols1"), py::arg("A"), py::arg("n_rows2"),
      py::arg("n_cols2"), py::arg("B"), py::arg("C"), py::arg("transA") = false,
      py::arg("transB") = false,
      R"pbdoc(Naive Implementation doing a Matrix Multplication
supporting transposition on CUDA.

:param n_rows1: number of rows for A
:param n_cols1: number of rows for A
:param A: pointer on CUDA
:param n_rows2: number of rows for B
:param n_cols2: number of rows for B
:param B: pointer on CUDA
:param C: allocated pointer on CUDA
:param transA: A needs to be transposed?
:param transB: B needs to be transposed?
)pbdoc");

  m.def(
      "matmul_v2_cuda",
      [](int n_rows1, int n_cols1, int64_t A, int n_rows2, int n_cols2,
         int64_t B, int64_t C, bool transA, bool transB) -> int {
        return matmul_v2(n_rows1, n_cols1, (const float *)A, n_rows2, n_cols2,
                         (const float *)B, (float *)C, transA, transB);
      },
      py::arg("n_rows1"), py::arg("n_cols1"), py::arg("A"), py::arg("n_rows2"),
      py::arg("n_cols2"), py::arg("B"), py::arg("C"), py::arg("transA") = false,
      py::arg("transB") = false,
      R"pbdoc(Naive Implementation with tiles
doing a Matrix Multplication supporting transposition on CUDA.

:param n_rows1: number of rows for A
:param n_cols1: number of rows for A
:param A: pointer on CUDA
:param n_rows2: number of rows for B
:param n_cols2: number of rows for B
:param B: pointer on CUDA
:param C: allocated pointer on CUDA
:param transA: A needs to be transposed?
:param transB: B needs to be transposed?
)pbdoc");

  m.def(
      "matmul_v3_cuda",
      [](int n_rows1, int n_cols1, int64_t A, int n_rows2, int n_cols2,
         int64_t B, int64_t C, bool transA, bool transB) -> int {
        return matmul_v3(n_rows1, n_cols1, (const float *)A, n_rows2, n_cols2,
                         (const float *)B, (float *)C, transA, transB);
      },
      py::arg("n_rows1"), py::arg("n_cols1"), py::arg("A"), py::arg("n_rows2"),
      py::arg("n_cols2"), py::arg("B"), py::arg("C"), py::arg("transA") = false,
      py::arg("transB") = false,
      R"pbdoc(Implementation doing a Matrix Multplication
supporting transposition on CUDA. It proceeds by blocks
within tiles.

:param n_rows1: number of rows for A
:param n_cols1: number of rows for A
:param A: pointer on CUDA
:param n_rows2: number of rows for B
:param n_cols2: number of rows for B
:param B: pointer on CUDA
:param C: allocated pointer on CUDA
:param transA: A needs to be transposed?
:param transB: B needs to be transposed?
)pbdoc");
}
