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

#define SEED 1337
#define MAX 10
#define MIN (-10)

class iom {
public:
  __host__ __device__ long long operator()(long long x, long long y) const {
    int xv = (x >> 32) & 0xFFFFFFFF;
    int xi = x & 0xFFFFFFFF;

    int yv = (y >> 32) & 0xFFFFFFFF;
    int yi = y & 0xFFFFFFFF;

    if (xv == yv) {
      return xi < yi ? x : y;
    } else {
      return xv < yv ? y : x;
    }
  }
};

int main(int argc, char **argv) {
  int runs, n;
  runs_and_n(argc, argv, &runs, &n);

  thrust::host_vector<int> h(n);
  thrust::device_vector<int> d(n);

  for (int i = 0; i < n; i++) {
    h[i] = ((long long)((std::rand()%(MAX-MIN))+MIN) << 32) | i;
  }
  d = h;
  cudaDeviceSynchronize();

  long long res;
  // Warmup
  res = thrust::reduce(d.begin(), d.end(), 0, iom());

  start_timing();
  for(size_t i = 0; i < runs; ++i) {
    res = thrust::reduce(d.begin(), d.end(), 0, iom());
  }
  end_timing();

  std::cout << "Result: " << int(res) << std::endl;
  report_time(runs);
}
