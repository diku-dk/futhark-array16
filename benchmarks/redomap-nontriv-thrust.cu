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
  __device__ float operator()(int i) {
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
  __device__ float operator()(float beg_date) {
    return add_months(beg_date, m_sw_freq);
  }
private:
  float m_sw_freq;
};

class lvls_op {
public:
  __device__ float operator()(float a1, float a2) {
    return zc(a2) * date_365(a2, a1);
  }
};

class zc_op {
public:
  __device__ float operator()(float x) {
    return zc(x);
  }
};


void benchmark(bool report, const int n_sched) {
  float maturity = add_months(today, 12.0*sw_mat);
  float sw_freq = 12.0 * sw_ty / n_sched;

  thrust::device_vector<float> beg_dates(n_sched);
  thrust::device_vector<float> end_dates(n_sched);
  thrust::device_vector<float> lvls(n_sched);

  thrust::transform(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(n_sched),
                    beg_dates.begin(),
                    beg_dates_op(maturity, sw_freq));

  thrust::transform(beg_dates.begin(),
                    beg_dates.end(),
                    end_dates.begin(),
                    end_dates_op(sw_freq));

  thrust::transform(beg_dates.begin(),
                    beg_dates.end(),
                    end_dates.begin(),
                    lvls.begin(),
                    lvls_op());

  float lvl = thrust::reduce(lvls.begin(), lvls.end(), 0.0, thrust::plus<float>());
  float t0 = *thrust::min_element(beg_dates.begin(), beg_dates.end());
  float tn = *thrust::max_element(end_dates.begin(), end_dates.end());

  thrust::device_vector<float> vt_ends(n_sched);
  thrust::device_vector<float> fact_aicis(n_sched);

  thrust::transform(beg_dates.begin(),
                    beg_dates.end(),
                    vt_ends.begin(),
                    zc_op());

  thrust::transform(end_dates.begin(),
                    end_dates.end(),
                    fact_aicis.begin(),
                    zc_op());

  if (report) {
    std::cout << std::fixed
              << "lvl: " << lvl << std::endl
              << "t0: " << t0 << std::endl
              << "tn: " << tn << std::endl
              << "beg_dates[0]: " << beg_dates[0] << std::endl
              << "end_dates[0]: " << end_dates[0] << std::endl;
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
