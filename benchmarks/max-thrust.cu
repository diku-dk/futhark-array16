#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>

#include "common.hpp"

int main(int argc, char **argv) {
  int runs, n;
  runs_and_n(argc, argv, &runs, &n);

  thrust::device_vector<int> d(n);
  init_vector(&d);

  int result;
  // Warmup
  result = thrust::reduce(d.begin(), d.end(),
                          0, thrust::maximum<int>());
  // Do it!
  start_timing();
  for (size_t i = 0; i < runs; ++i) {
    result = thrust::reduce(d.begin(), d.end(),
                            0, thrust::maximum<int>());
  }
  end_timing();

  std::cout << "Result: " << result << std::endl;
  report_time(runs);
}
