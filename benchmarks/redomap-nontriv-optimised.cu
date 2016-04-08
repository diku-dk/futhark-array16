#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <iostream>
#include <cstdlib>
#include <limits>

#include <sys/time.h>

#include "common.hpp"

/* Constants. */
static const float minInMonth = 43200.0;
static const float today = 1.683432e7;
static const float inf = std::numeric_limits<float>::infinity();
static const float r = 0.03;
static const float sw_mat  = 30.00;
static const float sw_ty   = 512.0;

__host__ __device__ float date_365(float t1, float t2) {
  return (t2 - t1) / (minInMonth*12.0);
}

__host__ __device__ float add_months(float date1, float num_months) {
  return num_months*minInMonth + date1;
}

__host__ __device__ float zc(float t) {
  return std::exp(-r * date_365(t, today));
}

class beg_dates_op {
public:
  beg_dates_op(float maturity, float sw_freq)
    : m_maturity(maturity), m_sw_freq(sw_freq)
  {}
  __host__ __device__ float operator()(int i) const {
    return add_months(m_maturity, i*m_sw_freq);
  }
private:
  float m_maturity, m_sw_freq;
};

class end_dates_op {
public:
  end_dates_op(float sw_freq)
    : m_sw_freq(sw_freq)
  {}
  __host__ __device__ float operator()(float beg_date) const {
    return add_months(beg_date, m_sw_freq);
  }
private:
  float m_sw_freq;
};

class lvls_op {
public:
  __host__ __device__ float operator()(float a1, float a2) const {
    return zc(a2) * date_365(a2, a1);
  }
};

class zc_op : thrust::unary_function<float, float> {
public:
  __host__ __device__ float operator()(float x) const {
    return zc(x);
  }
};


class pair_zc_op
  : public thrust::unary_function<const int,
                                  thrust::tuple<float, float> > {
public:
  pair_zc_op(float maturity, float sw_freq)
    : bdop(maturity, sw_freq),
      edop(sw_freq)
  {}

  __host__ __device__
  thrust::tuple<float, float> operator()(const int x) {
    float bd  = bdop(x);
    float ed  = edop(bd);

    return thrust::make_tuple(zc(bd), zc(ed));
  }

private:
  const beg_dates_op bdop;
  const end_dates_op edop;
};


class dates_lvl_op
  : public thrust::unary_function<const int,
                                  thrust::tuple<float, float, float> > {
public:
  dates_lvl_op(float maturity, float sw_freq)
    : bdop(maturity, sw_freq),
      edop(sw_freq),
      lop()
  {}

  __host__ __device__
  thrust::tuple<float, float, float> operator()(const int x) {
    float bd  = bdop(x);
    float ed  = edop(bd);
    float lvl = lop(bd, ed);

    return thrust::make_tuple(lvl, bd, ed);
  }

private:
  const beg_dates_op bdop;
  const end_dates_op edop;
  const lvls_op lop;
};


struct combined_red_op
  : public thrust::binary_function< const thrust::tuple<float, float, float> &,
                                    const thrust::tuple<float, float, float> &,
                                    thrust::tuple<float, float, float> >
{
  __host__ __device__
  thrust::tuple<float, float, float> operator()(const thrust::tuple<float, float, float>& x,
                                                const thrust::tuple<float, float, float>& y) const {
    float lvl = thrust::get<0>(x) + thrust::get<0>(y);
    float t0  = thrust::min(thrust::get<1>(x), thrust::get<1>(y));
    float tn  = thrust::max(thrust::get<2>(x), thrust::get<2>(y));
    return thrust::make_tuple(lvl, t0, tn);
  }
};






void benchmark(bool report, const int n_sched) {
  float maturity = add_months(today, 12.0*sw_mat);
  float sw_freq = 12.0 * sw_ty / n_sched;

  // Part of result
  thrust::device_vector<float> vt_ends(n_sched);
  thrust::device_vector<float> fact_aicis(n_sched);

  thrust::transform(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(n_sched),
                    thrust::make_zip_iterator(thrust::make_tuple(
                            vt_ends.begin(),
                            fact_aicis.begin())),
                    pair_zc_op(maturity, sw_freq));

  thrust::tuple<float, float, float> tmp = dates_lvl_op(maturity, sw_freq)(0);
  thrust::tuple<float, float, float> init(0.0, thrust::get<1>(tmp), thrust::get<2>(tmp));
  thrust::tuple<float, float, float> lvl_t0_tn =
    thrust::transform_reduce(thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(n_sched),
                             dates_lvl_op(maturity, sw_freq),
                             init,
                             combined_red_op());

  float lvl = thrust::get<0>(lvl_t0_tn);
  float t0 = thrust::get<1>(lvl_t0_tn);
  float tn = thrust::get<2>(lvl_t0_tn);


  if (report) {
    std::cout << std::fixed
              << "lvl: " << lvl << std::endl
              << "t0: " << t0 << std::endl
              << "tn: " << tn << std::endl;
              // << "vt_ends[0]: " << vt_ends[n_sched-1] << std::endl
              // << "fact_aicis[0]: " << fact_aicis[n_sched-1] << std::endl;
  }
}

int main(int argc, char **argv) {
  int runs, n_sched;
  runs_and_n(argc, argv, &runs, &n_sched);

  /* Warmup */
  benchmark(false, n_sched);

  start_timing();
  for(size_t i = 0; i < runs; ++i) {
      benchmark(true, n_sched);
  }
  end_timing();

  report_time(runs);
}
