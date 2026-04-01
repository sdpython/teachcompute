#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <memory>
#include <omp.h>  // <- OpenMP

namespace nb = nanobind;

using Array1D = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

nb::ndarray<nb::numpy, double, nb::ndim<1>> add_scaled_omp(Array1D a, Array1D b) {
    size_t n = a.shape(0);
    const double *pa = a.data();
    const double *pb = b.data();

    auto result = std::make_unique<double[]>(n);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        result[i] = pa[i] + 2.0 * pb[i];
    }

    double *raw = result.release();
    nb::capsule owner(raw, [](void *p) noexcept { delete[] (double *)p; });
    size_t shape[1] = {n};
    return nb::ndarray<nb::numpy, double, nb::ndim<1>>(raw, 1, shape, owner);
}

double dot_product_omp(Array1D a, Array1D b) {
    size_t n = a.shape(0);
    const double *pa = a.data();
    const double *pb = b.data();

    double total = 0.0;

    #pragma omp parallel for reduction(+:total)
    for (size_t i = 0; i < n; ++i) {
        total += pa[i] * pb[i];
    }

    return total;
}

NB_MODULE(dotnanobind, m) {
    m.def("add_scaled", &add_scaled_omp, "Add a + 2*b using OpenMP");
    m.def("dot_product", &dot_product_omp, "Dot product using OpenMP");
}
