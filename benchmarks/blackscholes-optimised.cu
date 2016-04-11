#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>

#include "common.hpp"

__host__ __device__ inline static
float horner(float x) {
  float c1 = 0.31938153, c2 = -0.356563782, c3 = 1.781477937, c4 = -1.821255978, c5 = 1.330274429;
  return x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5))));
}


__host__ __device__ inline static
float cnd0(float d) {
  float k        = 1.0 / (1.0 + 0.2316419 * abs(d));
  float p        = horner(k);
  float rsqrt2pi = 0.39894228040143267793994605993438;
  return rsqrt2pi * exp(-0.5*d*d) * p;
}

__host__ __device__ inline static
float cnd(float d) {
  float c = cnd0(d);
  return 0.0 < d ? 1.0 - c : c;
}

typedef thrust::tuple<bool, float, float, float> Option;

class make_option : public thrust::unary_function<const int, Option> {
public:
  make_option(int years) : days(years*365.0) {}
  
  __host__ __device__
  Option operator()(const int x) const {
    return Option(x % 2 == 0, 58.0 + 4.0 * x / days, 65.0, x / 365.0);
  }

private:
  const float days;
};

class blackscholes_op : public thrust::unary_function<const Option&, float> {
public:
    
  __host__ __device__
  float operator()(const Option &opt) const {
    bool call    = thrust::get<0>(opt);
    float price  = thrust::get<1>(opt);
    float strike = thrust::get<2>(opt);
    float years  = thrust::get<3>(opt);

    float r       = 0.08; // riskfree
    float v       = 0.30;  // volatility
    float v_sqrtT = v * sqrt(years);
    float d1      = (log(price / strike) + (r + 0.5 * v * v) * years) / v_sqrtT;
    float d2      = d1 - v_sqrtT;
    float cndD1   = cnd(d1);
    float cndD2   = cnd(d2);
    float x_expRT = strike * exp(-r * years);

    if (call)
      return price * cndD1 - x_expRT * cndD2;
    else
      return x_expRT * (1.0 - cndD2) - price * (1.0 - cndD1);
  }

};

float benchmark(const int years) {
  int days = years * 365;
  return thrust::transform_reduce(
             thrust::make_transform_iterator(thrust::make_counting_iterator(1), make_option(years)),
             thrust::make_transform_iterator(thrust::make_counting_iterator(days+1), make_option(years)),
             blackscholes_op(),
             0.0,
             thrust::plus<float>());
}


int main(int argc, char **argv) {
  int runs, years;
  runs_and_n(argc, argv, &runs, &years);

  /* Warmup */
  float res = benchmark(years);

  start_timing();
  for(size_t i = 0; i < runs; ++i) {
    std::cout << benchmark(years) << std::endl;
  }
  end_timing();

  report_time(runs);
}
