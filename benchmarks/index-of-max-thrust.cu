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

typedef thrust::tuple<int, int> IOMTuple;

class iom {
public:
  __host__ __device__ IOMTuple operator()(const IOMTuple &x,
                                          const IOMTuple &y) const {
    int xv = thrust::get<0>(x);
    int xi = thrust::get<1>(x);

    int yv = thrust::get<0>(y);
    int yi = thrust::get<1>(y);

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

  thrust::device_vector<int> d(n);
  init_vector(&d);

  IOMTuple res;
  thrust::counting_iterator<int> begin = thrust::make_counting_iterator(0);
  thrust::counting_iterator<int> end = thrust::make_counting_iterator(n);

  // Warmup
  res = thrust::reduce(thrust::make_zip_iterator
                       (thrust::make_tuple(d.begin(), begin)),
                       thrust::make_zip_iterator
                       (thrust::make_tuple(d.end(), end)),
                       IOMTuple(0,0), iom());

  start_timing();
  for(size_t i = 0; i < runs; ++i) {
    res = thrust::reduce(thrust::make_zip_iterator
                         (thrust::make_tuple(d.begin(), begin)),
                         thrust::make_zip_iterator
                         (thrust::make_tuple(d.end(), end)),
                         IOMTuple(0,0), iom());
  }
  end_timing();

  std::cout << "Result: " << thrust::get<1>(res) << std::endl;
  report_time(runs);
}
