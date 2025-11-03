#include <thread>
#include "thread_sum.h"

namespace validation {

void thread_sum(const std::vector<double> &values, int start, int end,
                double &result) {
  for (auto i = start; i < end; ++i) {
    result += values[i];
  }
}

double sum_no_mutex(const std::vector<double> &values) {
  int N = values.size();

  const int num_threads = 4;
  int chunk_size = N / num_threads;

  std::vector<std::thread> threads(num_threads);

  double somme = 0;
  for (int i = 0; i < num_threads; ++i) {
    int start = i * chunk_size;
    int end = (i == num_threads - 1) ? N : (i + 1) * chunk_size;
    threads[i] =
        std::thread(thread_sum, std::cref(values), start, end, std::ref(somme));
  }

  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }

  return somme;
}

double test_sum_no_mutex(int N) {
  std::vector<double> values(N, 1);
  return sum_no_mutex(values);
}

} // namespace validation
