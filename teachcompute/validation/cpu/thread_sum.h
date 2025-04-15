#pragma once

#include <cstddef>
#include <stdint.h>
#include <vector>

namespace validation {

void thread_sum(const std::vector<double> &values, int start, int end, double &result);

double sum_no_mutex(const std::vector<double> &values);

double test_sum_no_mutex(int N);
        
} // namespace validation
