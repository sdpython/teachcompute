#include "dot_cython_omp_.h"
#include <omp.h>

int get_omp_max_threads_cpp() { return omp_get_max_threads(); }

double vector_ddot_openmp(const double *p1, const double *p2, int size,
                          int nthreads) {
  if (nthreads <= 0)
    nthreads = ::omp_get_max_threads();
  double sum = 0;
#pragma omp parallel for reduction(+ : sum) num_threads(nthreads)
  for (int i = 0; i < size; ++i)
    sum += (p1[i] * p2[i]);
  return sum;
}

double vector_ddot_product_pointer16(const double *p1, const double *p2) {
  // Branching optimization must be done in a separate function.
  double sum = 0;

  sum += *(p1++) * *(p2++);
  sum += *(p1++) * *(p2++);
  sum += *(p1++) * *(p2++);
  sum += *(p1++) * *(p2++);
  sum += *(p1++) * *(p2++);
  sum += *(p1++) * *(p2++);
  sum += *(p1++) * *(p2++);
  sum += *(p1++) * *(p2++);

  sum += *(p1++) * *(p2++);
  sum += *(p1++) * *(p2++);
  sum += *(p1++) * *(p2++);
  sum += *(p1++) * *(p2++);
  sum += *(p1++) * *(p2++);
  sum += *(p1++) * *(p2++);
  sum += *(p1++) * *(p2++);
  sum += *(p1++) * *(p2++);

  return sum;
}

double vector_ddot_openmp_16(const double *p1, const double *p2, int size,
                             int nthreads) {
  if (nthreads <= 0)
    nthreads = ::omp_get_max_threads();

  double sum = 0;
  if (size >= 16) {
    int size_ = size / 16;
#pragma omp parallel for reduction(+ : sum) num_threads(nthreads)
    for (int i = 0; i < size_; ++i)
      sum += vector_ddot_product_pointer16(&p1[i * 16], &p2[i * 16]);
  }
  int mod = size % 16;
  size -= mod;
  p1 += size;
  p2 += size;
  for (; mod > 0; ++p1, ++p2, --mod)
    sum += *p1 * *p2;
  return sum;
}
