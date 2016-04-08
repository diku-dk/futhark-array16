#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <iostream>
#include <sys/time.h>

#include "common.hpp"

int main(int argc, char **argv) {
  int runs, n;
  runs_and_n(argc, argv, &runs, &n);

  thrust::device_vector<int> d(n);
  init_vector(&d);

  thrust::device_vector<int>::iterator res;
  unsigned int position = 0;
  
  // Warmup
  res = thrust::max_element(d.begin(), d.end());

  start_timing();
  for (size_t i = 0; i < runs; ++i) {
    res = thrust::max_element(d.begin(), d.end());
    position = res - d.begin();
  }
  end_timing();

  std::cout << "Result: " << position << std::endl;
  report_time(runs);
}
