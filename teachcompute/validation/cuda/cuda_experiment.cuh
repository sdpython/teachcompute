#pragma once

#include <tuple>

namespace cuda_example {

std::tuple<double, double>
measure_vector_add_half(unsigned int size, const short *ptr1, const short *ptr2,
                        short *ptr3, int cudaDevice, int repeat = 1);

} // namespace cuda_example
