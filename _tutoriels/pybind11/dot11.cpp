#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>  // <- OpenMP

namespace py = pybind11;

py::array_t<double> add_scaled_omp(py::array_t<double> a, py::array_t<double> b) {
    auto buf_a = a.unchecked<1>();
    auto buf_b = b.unchecked<1>();
    ssize_t n = buf_a.shape(0);

    py::array_t<double> result(n);
    auto buf_out = result.mutable_unchecked<1>();

    #pragma omp parallel for
    for (ssize_t i = 0; i < n; ++i) {
        buf_out(i) = buf_a(i) + 2.0 * buf_b(i);
    }

    return result;
}

double dot_product_omp(py::array_t<double> a, py::array_t<double> b) {
    auto buf_a = a.unchecked<1>();
    auto buf_b = b.unchecked<1>();
    ssize_t n = buf_a.shape(0);

    double total = 0.0;

    #pragma omp parallel for reduction(+:total)
    for (ssize_t i = 0; i < n; ++i) {
        total += buf_a(i) * buf_b(i);
    }

    return total;
}

PYBIND11_MODULE(dot11, m) {
    m.def("add_scaled", &add_scaled_omp, "Add a + 2*b using OpenMP");
    m.def("dot_product", &dot_product_omp, "Dot product using OpenMP");
}
