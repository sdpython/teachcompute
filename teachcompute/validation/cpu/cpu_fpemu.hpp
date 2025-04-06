// This file includes some pieces taken from
// https://github.com/IntelLabs/FP8-Emulation-Toolkit/blob/main/mpemu/pytquant/cuda/fpemu_kernels.cu
// with the following license.
//
/*----------------------------------------------------------------------------*
 * Copyright (c) 2023, Intel Corporation - All rights reserved.
 * This file is part of FP8-Emulation-Toolkit
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *----------------------------------------------------------------------------*
 * Naveen Mellempudi (Intel Corporation)
 *----------------------------------------------------------------------------*/
// Les versions ARM NEON ont été écrite avec un modèle de langage (LLM).

#pragma once

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#else
#include <arm_neon.h>
#endif

namespace cpu_fpemu {

#if defined(__x86_64__) || defined(_M_X64)

inline float __double2float_rn(double inval) {
  float out[4] = {0};
  __m128 vout = _mm_cvtpd_ps(_mm_set1_pd(inval));

  _mm_store_ps(&out[0], vout);
  return out[0];
}

#else

inline float __double2float_rn(double inval) {
  float64x1_t dval =
      vdup_n_f64(inval); // Duplique le double en un vecteur 64 bits
  float32x2_t fval = vcvt_f32_f64(dval); // Convertit le double en float
  return vget_lane_f32(fval, 0);         // Récupère la première valeur float
}

#endif

//////////////////
// __float2half_rn
//////////////////

#ifdef _WIN32

inline unsigned short __float2half_rn(float inval) {
  __m128i m = _mm_cvtps_ph(_mm_set_ss(inval),
                           (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  return _mm_extract_epi16(m, 0);
}

inline float __half2float(unsigned short h_val) {
  __m128i m = _mm_cvtsi32_si128(h_val);
  return _mm_cvtss_f32(_mm_cvtph_ps(m));
}

#else

#if defined(__x86_64__) || defined(_M_X64)

inline unsigned short __float2half_rn(float inval) {
  return _cvtss_sh(inval, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

#else

inline float __half2float(uint16_t h_val) {
  float32_t result;
  float16_t h = vdup_n_f16(*(reinterpret_cast<float16_t *>(&h_val)));
  vst1q_f32(&result, vcvt_f32_f16(vcombine_f16(h, h)));
  return result;
}

#endif

///////////////
// __half2float
///////////////

#if defined(__x86_64__) || defined(_M_X64)

inline float __half2float(unsigned short h_val) { return _cvtsh_ss(h_val); }

#else

inline float __half2float(uint16_t h_val) {
  float16_t h = vdup_n_f16(*(reinterpret_cast<float16_t *>(&h_val)));
  return vcvt_f32_f16(h);
}

#endif

#endif

} // namespace cpu_fpemu
