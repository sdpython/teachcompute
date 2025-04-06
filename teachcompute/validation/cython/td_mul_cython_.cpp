#include "mul_cython_omp_.h"
#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h> // for double m128d
#else
#include <arm_neon.h>
// Les versions ARM NEON ont été écrite avec un modèle de langage (LLM).
#endif
#include <stdio.h>

#if defined(__x86_64__) || defined(_M_X64)

double vector_ddot_product_pointer16_sse(const double *p1, const double *p2,
                                         int size) {
  if (size == 0)
    return 0;
  if (size == 1)
    return *p1 * *p2;
  double sum = 0;
  const double *end = p1 + size;
  if (size % 2 == 1) {
    sum += *p1 * *p2;
    ++p1;
    ++p2;
  }
  __m128d c1 = _mm_loadu_pd(p1);
  __m128d c2 = _mm_loadu_pd(p2);
  __m128d r1 = _mm_mul_pd(c1, c2);
  p1 += 2;
  p2 += 2;
  for (; p1 != end; p1 += 2, p2 += 2) {
    c1 = _mm_loadu_pd(p1);
    c2 = _mm_loadu_pd(p2);
    r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));
  }
  double r[2]; // r is not necessary aligned.
  _mm_storeu_pd(r, r1);
  sum += r[0] + r[1];
  return sum;
}

#else

double vector_ddot_product_pointer16_neon(const double *p1, const double *p2,
                                          int size) {
  if (size == 0)
    return 0;
  if (size == 1)
    return *p1 * *p2;

  double sum = 0;
  const double *end = p1 + size;

  // Gérer le cas impair
  if (size % 2 == 1) {
    sum += *p1 * *p2;
    ++p1;
    ++p2;
  }

  float64x2_t r1 = vmovq_n_f64(0.0); // Initialise à zéro

  for (; p1 != end; p1 += 2, p2 += 2) {
    float64x2_t c1 = vld1q_f64(p1); // Charge 2 doubles
    float64x2_t c2 = vld1q_f64(p2);
    r1 = vmlaq_f64(r1, c1, c2); // Multiplie et accumule
  }

  // Réduction finale : somme des 2 éléments du registre
  sum += vgetq_lane_f64(r1, 0) + vgetq_lane_f64(r1, 1);
  return sum;
}

#endif
