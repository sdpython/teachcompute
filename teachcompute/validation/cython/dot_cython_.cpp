#include "dot_cython_.h"

//////////////////////////
// branching
//////////////////////////

#define BYN 16

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

double vector_ddot_product_pointer16(const double *p1, const double *p2,
                                     int size) {
  double sum = 0;
  int i = 0;
  if (size >= BYN) {
    int size_ = size - BYN;
    for (; i < size_; i += BYN, p1 += BYN, p2 += BYN)
      sum += vector_ddot_product_pointer16(p1, p2);
  }
  size -= i;
  for (; size > 0; ++p1, ++p2, --size)
    sum += *p1 * *p2;
  return sum;
}

float vector_sdot_product_pointer16(const float *p1, const float *p2) {
  // Branching optimization must be done in a separate function.
  float sum = 0;

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

float vector_sdot_product_pointer16(const float *p1, const float *p2,
                                    int size) {
  float sum = 0;
  int i = 0;
  if (size >= BYN) {
    int size_ = size - BYN;
    for (; i < size_; i += BYN, p1 += BYN, p2 += BYN)
      sum += vector_sdot_product_pointer16(p1, p2);
  }
  size -= i;
  for (; size > 0; ++p1, ++p2, --size)
    sum += *p1 * *p2;
  return sum;
}

//////////////////////////
// branching + AVX
//////////////////////////

#ifdef _WIN32
// Not available on all machines
// It should depends on AVX constant not WIN32.
#include <immintrin.h> // double double m256d

double vector_ddot_product_pointer16_sse(const double *p1, const double *p2) {
  __m256d c1 = _mm256_load_pd(p1);
  __m256d c2 = _mm256_load_pd(p2);
  __m256d r1 = _mm256_mul_pd(c1, c2);

  p1 += 4;
  p2 += 4;

  c1 = _mm256_load_pd(p1);
  c2 = _mm256_load_pd(p2);
  r1 = _mm256_add_pd(r1, _mm256_mul_pd(c1, c2));

  p1 += 4;
  p2 += 4;

  c1 = _mm256_load_pd(p1);
  c2 = _mm256_load_pd(p2);
  r1 = _mm256_add_pd(r1, _mm256_mul_pd(c1, c2));

  p1 += 4;
  p2 += 4;

  c1 = _mm256_load_pd(p1);
  c2 = _mm256_load_pd(p2);
  r1 = _mm256_add_pd(r1, _mm256_mul_pd(c1, c2));

  double r[4];
  _mm256_store_pd(r, r1);

  return r[0] + r[1] + r[2] + r[3];
}

#else

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h> // for double m128d
#else
#include <arm_neon.h>
// Les versions ARM NEON ont été écrite avec un modèle de langage (LLM).
#endif

#if defined(__x86_64__) || defined(_M_X64)

double vector_ddot_product_pointer16_sse(const double *p1, const double *p2) {
  __m128d c1 = _mm_load_pd(p1);
  __m128d c2 = _mm_load_pd(p2);
  __m128d r1 = _mm_mul_pd(c1, c2);

  p1 += 2;
  p2 += 2;

  c1 = _mm_load_pd(p1);
  c2 = _mm_load_pd(p2);
  r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));

  p1 += 2;
  p2 += 2;

  c1 = _mm_load_pd(p1);
  c2 = _mm_load_pd(p2);
  r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));

  p1 += 2;
  p2 += 2;

  c1 = _mm_load_pd(p1);
  c2 = _mm_load_pd(p2);
  r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));

  // 8

  p1 += 2;
  p2 += 2;

  c1 = _mm_load_pd(p1);
  c2 = _mm_load_pd(p2);
  r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));

  p1 += 2;
  p2 += 2;

  c1 = _mm_load_pd(p1);
  c2 = _mm_load_pd(p2);
  r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));

  p1 += 2;
  p2 += 2;

  c1 = _mm_load_pd(p1);
  c2 = _mm_load_pd(p2);
  r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));

  p1 += 2;
  p2 += 2;

  c1 = _mm_load_pd(p1);
  c2 = _mm_load_pd(p2);
  r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));

  double r[2];
  _mm_store_pd(r, r1);

  return r[0] + r[1];
}

#else

double vector_ddot_product_pointer16_sse(const double *p1, const double *p2) {
  float64x2_t r1 =
      vmovq_n_f64(0.0); // Initialise le registre accumulateur à zéro

  for (int i = 0; i < 16;
       i += 2) { // Traite 2 éléments par itération (SIMD 128-bit)
    float64x2_t c1 = vld1q_f64(p1 + i); // Charge 2 doubles
    float64x2_t c2 = vld1q_f64(p2 + i);
    r1 = vmlaq_f64(r1, c1, c2); // Multiplie et accumule
  }

  // Réduction finale : somme les éléments du registre SIMD
  return vgetq_lane_f64(r1, 0) + vgetq_lane_f64(r1, 1);
}

#endif

#endif

double vector_ddot_product_pointer16_sse(const double *p1, const double *p2,
                                         int size) {
  double sum = 0;
  int i = 0;
  if (size >= BYN) {
    int size_ = size - BYN;
    for (; i < size_; i += BYN, p1 += BYN, p2 += BYN)
      sum += vector_ddot_product_pointer16_sse(p1, p2);
  }
  size -= i;
  for (; size > 0; ++p1, ++p2, --size)
    sum += *p1 * *p2;
  return sum;
}

#if defined(__x86_64__) || defined(_M_X64)

#include <xmmintrin.h> // for float m128

float vector_sdot_product_pointer16_sse(const float *p1, const float *p2) {
  __m128 c1 = _mm_load_ps(p1);
  __m128 c2 = _mm_load_ps(p2);
  __m128 r1 = _mm_mul_ps(c1, c2);

  p1 += 4;
  p2 += 4;

  c1 = _mm_load_ps(p1);
  c2 = _mm_load_ps(p2);
  r1 = _mm_add_ps(r1, _mm_mul_ps(c1, c2));

  p1 += 4;
  p2 += 4;

  c1 = _mm_load_ps(p1);
  c2 = _mm_load_ps(p2);
  r1 = _mm_add_ps(r1, _mm_mul_ps(c1, c2));

  p1 += 4;
  p2 += 4;

  c1 = _mm_load_ps(p1);
  c2 = _mm_load_ps(p2);
  r1 = _mm_add_ps(r1, _mm_mul_ps(c1, c2));

  float r[4];
  _mm_store_ps(r, r1);

  return r[0] + r[1] + r[2] + r[3];
}

#else

float vector_sdot_product_pointer16_neon(const float *p1, const float *p2) {
  float32x4_t r1 =
      vmovq_n_f32(0.0f); // Initialise le registre accumulateur à zéro

  for (int i = 0; i < 16;
       i += 4) { // Traite 4 éléments par itération (SIMD 128-bit)
    float32x4_t c1 = vld1q_f32(p1 + i); // Charge 4 floats
    float32x4_t c2 = vld1q_f32(p2 + i);
    r1 = vmlaq_f32(r1, c1, c2); // Multiplie et accumule
  }

  // Réduction finale : somme des éléments du registre SIMD
  return vaddvq_f32(r1);
}

#endif

float vector_sdot_product_pointer16_sse(const float *p1, const float *p2,
                                        int size) {
  float sum = 0;
  int i = 0;
  if (size >= BYN) {
    int size_ = size - BYN;
    for (; i < size_; i += BYN, p1 += BYN, p2 += BYN)
      sum += vector_sdot_product_pointer16_sse(p1, p2);
  }
  size -= i;
  for (; size > 0; ++p1, ++p2, --size)
    sum += *p1 * *p2;
  return sum;
}
