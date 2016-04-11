#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_scan.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>

#include "common.hpp"

class prepare {
public:
  prepare(int s) : m_s(s) {}

  __host__ __device__ int operator()(const int x) const {
    return x + m_s;
  }
private:
  int m_s;
};

class combine {
public:
  __host__ __device__ int operator()(int x, int y) const {
    char x11 = x>>24;
    char x12 = x>>16;
    char x21 = x>>8;
    char x22 = x>>0;

    char y11 = y>>24;
    char y12 = y>>16;
    char y21 = y>>8;
    char y22 = y>>0;

    char z11 = x11 * y11 + x12 * y21;
    char z12 = x11 * y12 + x12 * y22;
    char z21 = x21 * y11 + x22 * y21;
    char z22 = x21 * y12 + x22 * y22;

    return (((int(z11)<<24) & 0xFF) |
            ((int(z12)<<16) & 0xFF) |
            ((int(z21)<<8)  & 0xFF) |
            ((int(z22)<<8)  & 0xFF));
  }
};

int main(int argc, char **argv) {
  int runs, n;
  runs_and_n(argc, argv, &runs, &n);

  thrust::device_vector<int> d(n);
  thrust::device_vector<int> dres(n);

  init_vector(&d);

  // Warmup
  int s = 1;
  thrust::transform_inclusive_scan
    (d.begin(), d.end(),
     dres.begin(),
     prepare(s),
     combine());

  start_timing();
  for (size_t run = 0; run < runs; ++run) {
    for (int i = 0; i < 42; i++) {
      thrust::transform_inclusive_scan
        (d.begin(), d.end(),
         dres.begin(),
         prepare(s),
         combine());
      s = dres[n-1];
    }
  }
  end_timing();

  std::cout << "Result: " << s << std::endl;
  report_time(runs);
}
