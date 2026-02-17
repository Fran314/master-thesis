#ifndef FP_UTILITY_H__
#define FP_UTILITY_H__

#include <omp.h>

#include <random>
#include <unordered_map>
#include <vector>

using namespace std;

size_t send_size;
size_t recv_size;

double s_time;
double run_time;

struct TimeStats {
  int exec_cnt;
  double tot_time;
  unsigned long transfer_sz;

  double average_time() { return tot_time / exec_cnt; }
  double percentage(double all_time) { return tot_time / all_time; }
};

std::unordered_map<std::string, TimeStats> time_stats;

#define TIME_STATS_BEG                                                 \
  std::string __func_name = __FUNCTION__;                              \
  if (time_stats.find(__func_name) == time_stats.end())                \
    time_stats.insert(                                                 \
        {__func_name,                                                  \
         TimeStats{.exec_cnt = 0, .tot_time = 0., .transfer_sz = 0}}); \
  double __start_time = omp_get_wtime();                               \
  run_time += __start_time - s_time;                                   \
  s_time = __start_time;                                               \
  double __old_run_time = run_time;                                    \
  double __old_recv_size = recv_size;

#define TIME_STATS_END                                                \
  double __end_time = omp_get_wtime();                                \
  run_time += __end_time - s_time;                                    \
  s_time = __end_time;                                                \
  time_stats[__func_name].tot_time += run_time - __old_run_time;      \
  time_stats[__func_name].transfer_sz += recv_size - __old_recv_size; \
  time_stats[__func_name].exec_cnt++;

#define TIME_STATS_BEG_NAME(name)                                      \
  std::string __##name = #name;                                        \
  if (time_stats.find(__##name) == time_stats.end())                   \
    time_stats.insert(                                                 \
        {__##name,                                                     \
         TimeStats{.exec_cnt = 0, .tot_time = 0., .transfer_sz = 0}}); \
  double __start_time1_##name = omp_get_wtime();                       \
  run_time += __start_time1_##name - s_time;                           \
  s_time = __start_time1_##name;                                       \
  double __old_run_time1_##name = run_time;                            \
  double __old_recv_size1_##name = recv_size;

#define TIME_STATS_END_NAME(name)                                          \
  double __end_time1_##name = omp_get_wtime();                             \
  run_time += __end_time1_##name - s_time;                                 \
  s_time = __end_time1_##name;                                             \
  time_stats[__##name].tot_time += run_time - __old_run_time1_##name;      \
  time_stats[__##name].transfer_sz += recv_size - __old_recv_size1_##name; \
  time_stats[__##name].exec_cnt++;

template <typename FieldT>
void uni_hash_coeff_gen(vector<FieldT>& coeff, const FieldT& seed, size_t sz) {
  coeff.resize(sz);
  coeff[0] = seed;
  for (size_t i = 1; i < sz; ++i) coeff[i] = coeff[i - 1] * seed;
}

template <typename FieldT>
void vector_inn_prdt(FieldT res, const vector<FieldT>& x,
                     const vector<FieldT>& co) {
  res = FieldT::zero();
  for (size_t i = 0; i < x.size(); i++) res += x[i] * co[i];
}

template <typename FieldT>
FieldT mods(FieldT a, FieldT n) {
  a %= n;
  if (FieldT(2) * a > n) a -= n;
  return a;
}

template <typename FieldT>
FieldT quos(FieldT a, FieldT n) {
  return FieldT::divexact(a - mods(a, n), n);
}

template <typename FieldT>
void grem(FieldT& r0, FieldT& r1, FieldT w0, FieldT w1, FieldT z0, FieldT z1) {
  FieldT n = z0 * z0 + z1 * z1;
  FieldT u0 = quos(w0 * z0 + w1 * z1, n);
  FieldT u1 = quos(w1 * z0 - w0 * z1, n);
  r0 = w0 - z0 * u0 + z1 * u1;
  r1 = w1 - z0 * u1 - z1 * u0;
  return;
}

template <typename FieldT>
void ggcd(FieldT& r0, FieldT& r1, FieldT w0, FieldT w1, FieldT z0, FieldT z1) {
  while (!z0.is_zero() || !z1.is_zero()) {
    r0 = z0;
    r1 = z1;
    grem(z0, z1, w0, w1, z0, z1);
    w0 = r0;
    w1 = r1;
  }
  return;
}

template <typename FieldT>
FieldT root4(FieldT p) {
  FieldT k = p / FieldT(4);
  FieldT j = FieldT(2);
  FieldT n = FieldT(-1);
  FieldT o = FieldT(1);
  while (true) {
    FieldT a = FieldT::pow_mod(j, k, p);
    FieldT b = mods(a * a, p);
    if (b == n) return a;
    j += o;
  }
  return k;
}

template <typename FieldT>
void two_square_decomp(FieldT& y1, FieldT& y2, FieldT x) {
  FieldT a = root4(x);
  ggcd(y1, y2, x, FieldT::zero(), a, FieldT::one());
  if (!y1.is_positive()) y1 = -y1;
  if (!y2.is_positive()) y2 = -y2;
}

template <typename FieldT>
void three_square_decomp(FieldT& y1, FieldT& y2, FieldT& y3, const FieldT& x) {
  FieldT r = FieldT::sqrt_int(x);
  FieldT o = FieldT::one();
  FieldT p = x - r * r;
  while ((!p.is_prime()) && (p != o) && (p.is_zero() == false)) {
    r -= o;
    p = x - r * r;
  }
  y1 = r;
  if (p == o) {
    y2 = o;
    y3 = FieldT::zero();
    return;
  } else if (p.is_zero()) {
    y2 = FieldT::zero();
    y3 = FieldT::zero();
    return;
  }
  two_square_decomp(y2, y3, p);
}

template <typename FieldT>
void three_square_decomp(vector<FieldT>& y1, vector<FieldT>& y2,
                         vector<FieldT>& y3, const vector<FieldT>& x) {
  TIME_STATS_BEG
  size_t num = x.size();
  y1.resize(num);
  y2.resize(num);
  y3.resize(num);

#pragma omp parallel for
  for (size_t i = 0; i < num; i++) {
    FieldT::three_square_decomp_long(y1[i], y2[i], y3[i], x[i]);
  }
  TIME_STATS_END
}

void randomize(unsigned* r, size_t num) {
  // std::random_device rd;
  // size_t num_random_words = sizeof(unsigned) * num /
  // sizeof(std::random_device::result_type); auto random_words =
  // reinterpret_cast<std::random_device::result_type*>(r);
  for (size_t i = 0; i < num; ++i) {
    r[i] = 0xd23f63ea * (i + 1);
    // random_words[i] = 0x1233435345; rd();
  }
  return;
}

void randomize(vector<unsigned>& r) {
  // std::random_device rd;
  // size_t num_random_words = sizeof(unsigned) * num /
  // sizeof(std::random_device::result_type); auto random_words =
  // reinterpret_cast<std::random_device::result_type*>(r);
  size_t num = r.size();
  for (size_t i = 0; i < num; ++i) {
    r[i] = 0xd23f63ea * (i + 1);
    // random_words[i] = 0x1233435345; rd();
  }
  return;
}

bool test_bit(unsigned* r, size_t bitno, size_t num) {
  if (bitno >= num * sizeof(unsigned)) {
    return false;
  } else {
    const std::size_t part = bitno / sizeof(unsigned);
    const std::size_t bit = bitno - (sizeof(unsigned) * part);
    const size_t one = 1;
    return (r[part] & (one << bit)) != 0;
  }
}

bool test_bit(vector<unsigned>& r, size_t bitno) {
  size_t num = r.size();
  if (bitno >= num * sizeof(unsigned)) {
    return false;
  } else {
    const std::size_t part = bitno / sizeof(unsigned);
    const std::size_t bit = bitno - (sizeof(unsigned) * part);
    const size_t one = 1;
    return (r[part] & (one << bit)) != 0;
  }
}

long get_3D_idx(long l, long w, long d_idx, long l_idx, long w_idx) {
  return d_idx * (l * w) + l_idx * w + w_idx;
}

long get_4D_idx(long d, long l, long w, long o_idx, long d_idx, long l_idx,
                long w_idx) {
  return o_idx * (d * l * w) + d_idx * (l * w) + l_idx * w + w_idx;
}

#endif