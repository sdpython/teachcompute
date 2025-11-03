#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vector_sum.h"
#include "thread_sum.h"

namespace py = pybind11;
using namespace validation;

PYBIND11_MODULE(_validation, m) {
  m.doc() =
#if defined(__APPLE__)
      "C++ experimental implementations."
#else
      R"pbdoc(C++ experimental implementations.)pbdoc"
#endif
      ;

  m.def("vector_sum", &vector_sum, py::arg("n_columns"), py::arg("values"),
        py::arg("by_rows"),
        R"pbdoc(Computes the sum of all elements in an array
by rows or by columns. This function is slower than
:func:`vector_sum_array <teachcompute.validation.cpu._validation.vector_sum_array>`
as this function copies the data from an array to a `std::vector`.
This copy (and allocation) is bigger than the compution itself.

:param n_columns: number of columns
:param values: all values in an array
:param by_rows: by rows or by columns
:return: sum of all elements

See `vector_sum.cpp <https://github.com/sdpython/teachcompute/blob/main/teachcompute/validation/cpu/vector_sum.cpp>`_.
)pbdoc");

  m.def("vector_sum_array", &vector_sum_array, py::arg("n_columns"),
        py::arg("values"), py::arg("by_rows"),
        R"pbdoc(Computes the sum of all elements in an array
by rows or by columns.

:param n_columns: number of columns
:param values: all values in an array
:param by_rows: by rows or by columns
:return: sum of all elements

See `vector_sum.cpp <https://github.com/sdpython/teachcompute/blob/main/teachcompute/validation/cpu/vector_sum.cpp>`_.
)pbdoc");

  m.def("vector_sum_array_parallel", &vector_sum_array_parallel,
        py::arg("n_columns"), py::arg("values"), py::arg("by_rows"),
        R"pbdoc(Computes the sum of all elements in an array
by rows or by columns. The computation is parallelized.

:param n_columns: number of columns
:param values: all values in an array
:param by_rows: by rows or by columns
:return: sum of all elements

See `vector_sum.cpp <https://github.com/sdpython/teachcompute/blob/main/teachcompute/validation/cpu/vector_sum.cpp>`_.
)pbdoc");

  m.def("vector_sum_array_avx", &vector_sum_array_avx, py::arg("n_columns"),
        py::arg("values"),
        R"pbdoc(Computes the sum of all elements in an array
by rows or by columns. The computation uses AVX instructions
(see `AVX API
<https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html>`_).

:param n_columns: number of columns
:param values: all values in an array
:return: sum of all elements

See `vector_sum.cpp <https://github.com/sdpython/teachcompute/blob/main/teachcompute/validation/cpu/vector_sum.cpp>`_.
)pbdoc");

  m.def("vector_sum_array_avx_parallel", &vector_sum_array_avx_parallel,
        py::arg("n_columns"), py::arg("values"),
        R"pbdoc(Computes the sum of all elements in an array
by rows or by columns. The computation uses AVX instructions
and parallelization (see `AVX API
<https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html>`_).

:param n_columns: number of columns
:param values: all values in an array
:return: sum of all elements

See `vector_sum.cpp <https://github.com/sdpython/teachcompute/blob/main/teachcompute/validation/cpu/vector_sum.cpp>`_.
)pbdoc");

  m.def("vector_add", &vector_add, py::arg("v1"), py::arg("v2"),
        R"pbdoc(Computes the addition of 2 vectors of any dimensions.
It assumes both vectors have the same dimensions (no broadcast).

:param v1: first vector
:param v2: second vector
:return: new vector

See `vector_sum.cpp <https://github.com/sdpython/teachcompute/blob/main/teachcompute/validation/cpu/vector_sum.cpp>`_.
)pbdoc");

  m.def("test_sum_no_mutex", &test_sum_no_mutex, py::arg("N"),
        R"pbdoc(Computes a parallelized sum with no mutex.

:param N: number of 1 to sum
:return: result
)pbdoc");
}
