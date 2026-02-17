#ifndef FP_OS_TRIPLE_PRV_H__
#define FP_OS_TRIPLE_PRV_H__

#include <omp.h>
#include <time.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "zknn/utility.h"
#include "zknn/vole/lvole-prover.h"
#include "zknn/zknn-arith/cnn-prover.h"
#include "zknn/zknn-arith/ostriple-common.h"
#include "zknn/zknn-arith/quantized-value.h"
#include "zknn/zknn-arith/transformer-prover.h"

using namespace emp;
using namespace std;

double range_exec_time = 0;
double square_exec_time = 0;
double auth_exec_time = 0;
double other_exec_time = 0;

extern size_t send_size;
extern size_t recv_size;

extern double s_time;
extern double run_time;

template <typename IO, typename FieldT>
class FpOSTriplePrv {
 public:
  IO *io;
  IO **ios;
  size_t threads;
  VoleTriplePrv<IO, FieldT> *vole = nullptr;
  Hash hash;
  vector<FieldT> vA0;
  vector<FieldT> vA1;
  const static unsigned long Repeat = 42;
  const static unsigned long e = 48;
  const FieldT &L = FieldT(1 << 20);

  vector<FieldT> mac, x, B;

  FpOSTriplePrv(IO **ios, size_t threads) {
    this->io = ios[0];
    this->ios = ios;
    this->threads = threads;
    vole = new VoleTriplePrv<IO, FieldT>(ios, threads);
    vole->setup();
  }

  ~FpOSTriplePrv() { delete vole; }

  /* ---------------------inputs----------------------*/

  /*
   * authenticated bits for inputs of the prover
   */
  void authenticated_val_input(FieldT &mac, const FieldT w) {
    FieldT u;
    vole->extend(mac, u);
    u = u - w;
    io->send_data(&u, sizeof(FieldT));
  }

  void authenticated_val_input(vector<FieldT> &mac, const vector<FieldT> &w,
                               size_t len) {
    vector<FieldT> u(len);

    TIME_STATS_BEG_NAME(extend)
    vole->extend(mac, u, len);
    TIME_STATS_END_NAME(extend)
    

    // #pragma omp parallel for
    for (size_t i = 0; i < len; ++i) {
      u[i] = u[i] - w[i];
    }
    io->send_data(&u[0], len * sizeof(FieldT));
  }

  void inline auth_mac_add(FieldT &mac, FieldT &x, const FieldT &mac1,
                           const FieldT &x1, const FieldT &mac2,
                           const FieldT &x2) {
    mac = mac1 + mac2;
    x = x1 + x2;
  }

  void inline auth_mac_sub(FieldT &mac, FieldT &x, const FieldT &mac1,
                           const FieldT &x1, const FieldT &mac2,
                           const FieldT &x2) {
    mac = mac1 - mac2;
    x = x1 - x2;
  }

  void inline auth_pub_add(FieldT &mac, FieldT &x, const FieldT &mac1,
                           const FieldT &x1, const FieldT &c) {
    mac = mac1;
    x = x1 + c;
  }

  void inline auth_pub_mul(FieldT &mac, FieldT &x, const FieldT &mac1,
                           const FieldT &x1, const FieldT &c) {
    mac = c * mac1;
    x = c * x1;
  }

  void inline auth_mac_linear(FieldT &mac, FieldT &x,
                              const vector<FieldT> &vmac,
                              const vector<FieldT> &vx,
                              const vector<FieldT> &vc) {
    mac = FieldT::zero();
    x = FieldT::zero();
    for (size_t i = 0; i < vx.size(); i++) {
      mac += vmac[i] * vc[i];
      x += vx[i] * vc[i];
    }
  }

  void check_zero(const vector<FieldT> &mac) {
    size_t len = mac.size();
    hash.put(&mac[0], len * sizeof(FieldT));
  }

  void check_zero(const FieldT &mac) { hash.put(&mac, sizeof(FieldT)); }

  void zkp_poly_deg2(const vector<FieldT> &left_mac,
                     const vector<FieldT> &left_x,
                     const vector<FieldT> &right_mac,
                     const vector<FieldT> &right_x,
                     const vector<FieldT> &coeff) {
    size_t left_num = left_mac.size();
    size_t right_num = right_mac.size();
    FieldT A0 = FieldT::zero();
    FieldT A1 = FieldT::zero();
    for (size_t i = 0; i < left_num; ++i) {
      A0 = A0 + coeff[i] * left_mac[i] * right_mac[i];
      A1 =
          A1 + coeff[i] * (left_mac[i] * right_x[i] + right_mac[i] * left_x[i]);
    }
    for (size_t i = left_num; i < right_num; i++) {
      A1 = A1 + coeff[i] * right_mac[i];
    }
    vA0.push_back(A0);
    vA1.push_back(A1);
  }

  void zkp_poly_deg2(const vector<FieldT> &left_mac,
                     const vector<FieldT> &left_x,
                     const vector<FieldT> &right_mac,
                     const vector<FieldT> &right_x, const vector<FieldT> &coeff,
                     size_t A_idx) {
    size_t left_num = left_mac.size();
    size_t right_num = right_mac.size();
    FieldT A0 = FieldT::zero();
    FieldT A1 = FieldT::zero();
    for (size_t i = 0; i < left_num; ++i) {
      A0 = A0 + coeff[i] * left_mac[i] * right_mac[i];
      A1 =
          A1 + coeff[i] * (left_mac[i] * right_x[i] + right_mac[i] * left_x[i]);
    }
    for (size_t i = left_num; i < right_num; i++) {
      A1 = A1 + coeff[i] * right_mac[i];
    }
    vA0[A_idx] = A0;
    vA1[A_idx] = A1;
  }

  void zkp_poly_deg2_p(const vector<FieldT> &left_mac,
                       const vector<FieldT> &left_x,
                       const vector<FieldT> &right_mac,
                       const vector<FieldT> &right_x,
                       const vector<FieldT> &coeff) {
    size_t left_num = left_mac.size();
    size_t right_num = right_mac.size();
    FieldT A0 = FieldT::zero();
    FieldT A0_p = FieldT::zero();
    FieldT A1 = FieldT::zero();
    FieldT A1_p = FieldT::zero();

#pragma omp parallel private(A0_p, A1_p) shared(A0, A1)
    {
#pragma omp for
      for (size_t i = 0; i < left_num; ++i) {
        A0_p = A0_p + coeff[i] * left_mac[i] * right_mac[i];
        A1_p = A1_p +
               coeff[i] * (left_mac[i] * right_x[i] + right_mac[i] * left_x[i]);
      }
#pragma omp critical
      A0 += A0_p;
#pragma omp critical
      A1 += A1_p;
    }

    for (size_t i = left_num; i < right_num; i++) {
      A1 = A1 + coeff[i] * right_mac[i];
    }
    vA0.push_back(A0);
    vA1.push_back(A1);
  }

  void range_get_poly_co(vector<FieldT> &co, const FieldT &B) {
    co.resize(6);
    co[0] = FieldT(4);
    co[1] = FieldT(1);
    co[2] = FieldT(1);
    co[3] = FieldT(1);
    co[4] = FieldT(-4) * B;
    co[5] = FieldT(-1);
  }

  void range_get_poly_value(vector<FieldT> &left_mac, vector<FieldT> &left_x,
                            vector<FieldT> &right_mac, vector<FieldT> &right_x,
                            const FieldT &mac, const FieldT &x,
                            const FieldT &mac_y1, const FieldT &y1,
                            const FieldT &mac_y2, const FieldT &y2,
                            const FieldT &mac_y3, const FieldT &y3) {
    left_mac.resize(4);
    left_x.resize(4);
    right_mac.resize(5);
    right_x.resize(5);

    left_mac[0] = mac;
    left_mac[1] = mac_y1;
    left_mac[2] = mac_y2;
    left_mac[3] = mac_y3;
    left_x[0] = x;
    left_x[1] = y1;
    left_x[2] = y2;
    left_x[3] = y3;
    right_mac[0] = mac;
    right_mac[1] = mac_y1;
    right_mac[2] = mac_y2;
    right_mac[3] = mac_y3;
    right_mac[4] = mac;
    right_x[0] = x;
    right_x[1] = y1;
    right_x[2] = y2;
    right_x[3] = y3;
    right_x[4] = x;
  }

  void zkp_range(const vector<FieldT> &mac, const vector<FieldT> &x,
                 const FieldT &B, size_t num) {
    this->mac.insert(this->mac.end(), mac.begin(), mac.end());
    this->x.insert(this->x.end(), x.begin(), x.end());
    while (num--) {
      this->B.push_back(B);
    }
  }

  void zkp_range_batch(const FieldT &B_bound) {
    TIME_STATS_BEG
    size_t num = mac.size();

    vector<FieldT> y1, y2, y3;
    vector<FieldT> xx;
    xx.resize(num);



    TIME_STATS_BEG_NAME(init)
#pragma omp parallel for
    for (size_t i = 0; i < num; i++)
      xx[i] = FieldT(4) * x[i] * (B[i] - x[i]) + FieldT(1);
    TIME_STATS_END_NAME(init)

    three_square_decomp(y1, y2, y3, xx);


    TIME_STATS_BEG_NAME(auth)
    vector<FieldT> mac_y1, mac_y2, mac_y3;
    authenticated_val_input(mac_y1, y1, num);
    authenticated_val_input(mac_y2, y2, num);
    authenticated_val_input(mac_y3, y3, num);
    TIME_STATS_END_NAME(auth)

    // shortness test begin
    TIME_STATS_BEG_NAME(shortness_test)

    size_t random_num =
        (num * Repeat * 4 + 8 * sizeof(unsigned) - 1) / (8 * sizeof(unsigned));
    vector<unsigned> gamma(random_num);

    run_time += omp_get_wtime() - s_time;
    io->recv_data(&gamma[0], random_num * sizeof(unsigned));
    recv_size += random_num * sizeof(unsigned);
    s_time = omp_get_wtime();

    vector<FieldT> zeta, mac_zeta;
    zeta.resize(Repeat);
    mac_zeta.resize(Repeat);

  double itime, ftime, exec_time;
    itime = omp_get_wtime();

#pragma omp parallel for
    for (size_t i = 0; i < Repeat; i++) {
      for (size_t j = 0; j < num; j++)
        if (test_bit(gamma, (j * 4 + 0) * i) == true)
          auth_mac_add(mac_zeta[i], zeta[i], mac_zeta[i], zeta[i], mac[j],
                       x[j]);
      for (size_t j = 0; j < num; j++)
        if (test_bit(gamma, (j * 4 + 1) * i) == true)
          auth_mac_add(mac_zeta[i], zeta[i], mac_zeta[i], zeta[i], mac_y1[j],
                       y1[j]);
      for (size_t j = 0; j < num; j++)
        if (test_bit(gamma, (j * 4 + 2) * i) == true)
          auth_mac_add(mac_zeta[i], zeta[i], mac_zeta[i], zeta[i], mac_y2[j],
                       y2[j]);
      for (size_t j = 0; j < num; j++)
        if (test_bit(gamma, (j * 4 + 3) * i) == true)
          auth_mac_add(mac_zeta[i], zeta[i], mac_zeta[i], zeta[i], mac_y3[j],
                       y3[j]);
    }
    
    ftime = omp_get_wtime();
    square_exec_time += ftime - itime;

    printf("Square taken is %f\n", square_exec_time);

    // shortness test end
    TIME_STATS_END_NAME(shortness_test)
    
    // shortness test begin
    TIME_STATS_BEG_NAME(digit_decomposition)
    // bit decomposition
    size_t nbits = (B_bound * 4 * num).as_bigint().num_bits();
    vector<FieldT> zeta_bit(Repeat * nbits), mac_zeta_bit;
    for (int i = 0; i < Repeat; i++) {
      auto zeta_i = zeta[i].as_bigint();
      for (int j = 0; j < nbits; j++) {
        zeta_bit[i * nbits + j] = zeta_i.test_bit(j);
      }
    }
    authenticated_val_input(mac_zeta_bit, zeta_bit, Repeat * nbits);

    for (int i = 0; i < Repeat; i++) {
      FieldT mac_rhs, rhs;
      for (int j = 0; j < nbits; j++) {
        int k = i * nbits + j;
        auth_pub_mul(mac_zeta_bit[k], zeta_bit[k], mac_zeta_bit[k], zeta_bit[k], FieldT(2) ^ j);
        auth_mac_add(mac_rhs, rhs, mac_rhs, rhs, mac_zeta_bit[k], zeta_bit[k]);
      }
      auth_mac_sub(mac_zeta[i], zeta[i], mac_zeta[i], zeta[i], mac_rhs, rhs);
    }
    check_zero(mac_zeta);

  TIME_STATS_END_NAME(digit_decomposition)

    TIME_STATS_BEG_NAME(poly)
    size_t vA_oSize = vA0.size();
    vA0.resize(vA0.size() + num);
    vA1.resize(vA1.size() + num);
#pragma omp parallel for
    for (size_t i = 0; i < num; i++) {
      vector<FieldT> co;
      range_get_poly_co(co, B[i]);
      vector<FieldT> left_mac;
      vector<FieldT> left_x;
      vector<FieldT> right_mac;
      vector<FieldT> right_x;
      range_get_poly_value(left_mac, left_x, right_mac, right_x, mac[i], x[i],
                           mac_y1[i], y1[i], mac_y2[i], y2[i], mac_y3[i],
                           y3[i]);
      zkp_poly_deg2(left_mac, left_x, right_mac, right_x, co, vA_oSize + i);
    }
    TIME_STATS_END_NAME(poly)

    this->mac.clear();
    this->x.clear();
    this->B.clear();

    TIME_STATS_END
  }

  void lookup_calc_g(vector<FieldT> &g_mid, const vector<vector<FieldT>> &x,
                     const vector<FieldT> &alpha_dim, const FieldT &gamma,
                     size_t x_size, size_t dim) {
    FieldT x1_k = FieldT::zero();
    FieldT x2_k = FieldT::zero();
    for (size_t i = 0; i < dim; i++) {
      x1_k += alpha_dim[i] * x[0][i];
      x2_k += alpha_dim[i] * x[1][i];
    }
    g_mid[0] = (gamma + x1_k) * (gamma + x2_k);
    for (size_t i = 2; i < x_size; i++) {
      FieldT x_k = FieldT::zero();
      for (size_t j = 0; j < dim; j++) x_k += alpha_dim[j] * x[i][j];
      g_mid[i - 1] = g_mid[i - 2] * (gamma + x_k);
    }
  }

  void lookup_calc_h(vector<FieldT> &h_mid, const vector<vector<FieldT>> &s,
                     const vector<FieldT> &alpha_dim, const FieldT &beta,
                     const FieldT &gamma_beta, size_t s_size, size_t dim) {
    FieldT s1_k = FieldT::zero();
    FieldT s2_k = FieldT::zero();
    for (size_t i = 0; i < dim; i++) {
      s1_k += alpha_dim[i] * (s[0][i] + beta * s[1][i]);
      s2_k += alpha_dim[i] * (s[1][i] + beta * s[2][i]);
    }
    h_mid[0] = (gamma_beta + s1_k) * (gamma_beta + s2_k);
    for (size_t i = 2; i < s_size - 1; i++) {
      FieldT s_k = FieldT::zero();
      for (size_t j = 0; j < dim; j++)
        s_k += alpha_dim[j] * (s[i][j] + beta * s[i + 1][j]);
      h_mid[i - 1] = h_mid[i - 2] * (gamma_beta + s_k);
    }
  }

  void lookup_get_poly_value_g(
      vector<FieldT> &left_mac, vector<FieldT> &left_x,
      vector<FieldT> &right_mac, vector<FieldT> &right_x,
      const vector<vector<FieldT>> &mac_x, const vector<vector<FieldT>> &x,
      const vector<FieldT> &mac_g_mid, const vector<FieldT> &g_mid,
      const vector<FieldT> &alpha_dim, const FieldT &gamma, size_t i) {
    left_mac.resize(1);
    left_x.resize(1);
    right_mac.resize(2);
    right_x.resize(2);
    if (i != 0) {
      left_mac[0] = mac_g_mid[i - 1];
      left_x[0] = g_mid[i - 1];
    } else {
      auth_mac_linear(left_mac[0], left_x[0], mac_x[i], x[i], alpha_dim);
      auth_pub_add(left_mac[0], left_x[0], left_mac[0], left_x[0], gamma);
    }
    auth_mac_linear(right_mac[0], right_x[0], mac_x[i + 1], x[i + 1],
                    alpha_dim);
    auth_pub_add(right_mac[0], right_x[0], right_mac[0], right_x[0], gamma);
    right_mac[1] = mac_g_mid[i];
    right_x[1] = g_mid[i];
  }

  void lookup_get_poly_value_h(
      vector<FieldT> &left_mac, vector<FieldT> &left_x,
      vector<FieldT> &right_mac, vector<FieldT> &right_x,
      const vector<vector<FieldT>> &mac_s, const vector<vector<FieldT>> &s,
      const vector<FieldT> &mac_h_mid, const vector<FieldT> &h_mid,
      const vector<FieldT> &alpha_dim, const FieldT &gamma_beta,
      const FieldT &beta, size_t i) {
    size_t dim = alpha_dim.size();
    left_mac.resize(1);
    left_x.resize(1);
    right_mac.resize(2);
    right_x.resize(2);

    if (i != 0) {
      left_mac[0] = mac_h_mid[i - 1];
      left_x[0] = h_mid[i - 1];
    } else {
      vector<FieldT> mac_beta_s(dim);
      vector<FieldT> beta_s(dim);
      for (size_t j = 0; j < dim; j++) {
        auth_pub_mul(mac_beta_s[j], beta_s[j], mac_s[i + 1][j], s[i + 1][j],
                     beta);
        auth_mac_add(mac_beta_s[j], beta_s[j], mac_beta_s[j], beta_s[j],
                     mac_s[i][j], s[i][j]);
      }
      auth_mac_linear(left_mac[0], left_x[0], mac_beta_s, beta_s, alpha_dim);
      auth_pub_add(left_mac[0], left_x[0], left_mac[0], left_x[0], gamma_beta);
    }

    vector<FieldT> mac_beta_s(dim);
    vector<FieldT> beta_s(dim);
    for (size_t j = 0; j < dim; j++) {
      auth_pub_mul(mac_beta_s[j], beta_s[j], mac_s[i + 2][j], s[i + 2][j],
                   beta);
      auth_mac_add(mac_beta_s[j], beta_s[j], mac_beta_s[j], beta_s[j],
                   mac_s[i + 1][j], s[i + 1][j]);
    }
    auth_mac_linear(right_mac[0], right_x[0], mac_beta_s, beta_s, alpha_dim);
    auth_pub_add(right_mac[0], right_x[0], right_mac[0], right_x[0],
                 gamma_beta);
    right_mac[1] = mac_h_mid[i];
    right_x[1] = h_mid[i];
  }

  void zkp_lookup(const vector<vector<FieldT>> &mac_x,
                  const vector<vector<FieldT>> &x,
                  const vector<vector<FieldT>> &T) {
    map<vector<FieldT>, size_t> s_map;
    for (auto it = T.begin(); it != T.end(); it++) s_map[*it] = 1;
    for (auto it = x.begin(); it != x.end(); it++) s_map[*it] += 1;
    vector<vector<FieldT>> mac_s;
    vector<vector<FieldT>> s;
    for (auto it = s_map.begin(); it != s_map.end(); it++)
      for (size_t i = 0; i < it->second; i++) s.push_back(it->first);

    size_t t_size = T.size();
    size_t x_size = x.size();
    size_t s_size = s.size();
    size_t dim = mac_x[0].size();
    mac_s.resize(s_size);
    for (size_t i = 0; i < s_size; i++)
      authenticated_val_input(mac_s[i], s[i], dim);

    FieldT r[3];
    run_time += omp_get_wtime() - s_time;
    io->recv_data(&r[0], 3 * sizeof(FieldT));
    recv_size += 3 * sizeof(FieldT);
    s_time = omp_get_wtime();

    vector<FieldT> alpha_dim(dim);
    alpha_dim[0] = FieldT::one();
    for (size_t i = 1; i < dim; i++) alpha_dim[i] = alpha_dim[i - 1] * r[0];
    FieldT gamma_beta = r[2] * (FieldT::one() + r[1]);

    vector<FieldT> mac_g_mid(x_size - 1);
    vector<FieldT> g_mid(x_size - 1);
    lookup_calc_g(g_mid, x, alpha_dim, r[2], x_size, dim);
    authenticated_val_input(mac_g_mid, g_mid, x_size - 1);

    vector<FieldT> mac_h_mid(s_size - 2);
    vector<FieldT> h_mid(s_size - 2);
    lookup_calc_h(h_mid, s, alpha_dim, r[1], gamma_beta, s_size, dim);
    authenticated_val_input(mac_h_mid, h_mid, s_size - 2);

    vector<FieldT> co_gh(3);
    co_gh[0] = FieldT::one();
    co_gh[1] = -FieldT::one();
    co_gh[2] = FieldT::zero();
    for (size_t i = 0; i < x_size - 1; i++) {
      vector<FieldT> left_mac;
      vector<FieldT> left_x;
      vector<FieldT> right_mac;
      vector<FieldT> right_x;
      lookup_get_poly_value_g(left_mac, left_x, right_mac, right_x, mac_x, x,
                              mac_g_mid, g_mid, alpha_dim, r[2], i);
      zkp_poly_deg2(left_mac, left_x, right_mac, right_x, co_gh);
    }
    for (size_t i = 0; i < s_size - 2; i++) {
      vector<FieldT> left_mac;
      vector<FieldT> left_x;
      vector<FieldT> right_mac;
      vector<FieldT> right_x;
      lookup_get_poly_value_h(left_mac, left_x, right_mac, right_x, mac_s, s,
                              mac_h_mid, h_mid, alpha_dim, gamma_beta, r[1], i);
      zkp_poly_deg2(left_mac, left_x, right_mac, right_x, co_gh);
    }
  }

  void batch_check(bool zero_check, bool poly_check, bool range_check) {
    TIME_STATS_BEG
    // zero check.
    if (zero_check || range_check) {
      char dig[emp::Hash::DIGEST_SIZE];
      hash.digest(dig);
      io->send_data(dig, emp::Hash::DIGEST_SIZE);
      io->flush();
    }

    // poly check.
    if ((poly_check || range_check) && (vA0.size() != 0)) {
      FieldT seed;
      vector<FieldT> chi;
      FieldT U[2];
      run_time += omp_get_wtime() - s_time;
      io->recv_data(&seed, sizeof(FieldT));
      recv_size += sizeof(FieldT);
      s_time = omp_get_wtime();
      uni_hash_coeff_gen(chi, seed, vA0.size());

      U[0] = FieldT::zero();
      U[1] = FieldT::zero();
      for (size_t i = 0; i < vA0.size(); i++) {
        U[0] = U[0] + vA0[i] * chi[i];
        U[1] = U[1] + vA1[i] * chi[i];
      }
      FieldT Astar0;
      FieldT Astar1;
      vole->extend(Astar0, Astar1);

      U[0] = U[0] + Astar0;
      U[1] = U[1] + Astar1;
      io->send_data(U, 2 * sizeof(FieldT));
      vA0.clear();
      vA1.clear();
    }

    // // range check.
    // // do nothing.

    TIME_STATS_END
    return;
  }

  /* ======= primitive operations ======= */

#define SIDE_DECL() vector<FieldT> left, left_mac, right, right_mac
#define SIDE_PUSH_VAR(side, var, ident) \
  side.push_back(var.ident);            \
  side##_mac.push_back(var.mac_##ident)
#define SIDE_PUSH(side, ident) \
  side.push_back(ident);       \
  side##_mac.push_back(mac_##ident)
#define ZKP_POLY2() zkp_poly_deg2(left_mac, left, right_mac, right, co)

  template <typename Int>
  void zkp_max(const QuantizedValueProver<FieldT, Int> &input,
               const QuantizedValueProver<FieldT, Int> &output,
               size_t bits = 2 * sizeof(Int)) {
    TIME_STATS_BEG
    // 0 <= y - x <= B
    QuantizedValueProver<FieldT, Int> r = output - input;
    zkp_range(r.mac_value, r.value, B_max(bits), r.size());

    // prod(y - x) == 0
    // mid = prod(y - x)
    size_t sz = r.size();
    FieldT mid = r.value[0], mac_mid = r.mac_value[0];
    vector<FieldT> co = co_max<FieldT>();

    // prove mid == prod(y - x)
    for (size_t k = 1; k < sz; k++) {
      SIDE_DECL();
      SIDE_PUSH(left, mid);
      mid *= r.value[k];
      authenticated_val_input(mac_mid, mid);
      SIDE_PUSH_VAR(right, r, value[k]);
      SIDE_PUSH(right, mid);
      ZKP_POLY2();
    }
    // mid == 0
    check_zero(mac_mid);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_maximum(const QuantizedValueProver<FieldT, Int> &mn,
                   const QuantizedValueProver<FieldT, Int> &input,
                   const QuantizedValueProver<FieldT, Int> &output) {
    // y = max(x, mn)
    size_t sz = output.size();
    QuantizedValueProver<FieldT, Int> r1 = output - mn, r2 = output - input;
    // (y - mn)(y - x) == 0
    vector<FieldT> co = co_maximum<FieldT>(sz);
    zkp_poly_deg2(r1.mac_value, r1.value, r2.mac_value, r2.value, co);
    // 0 <= y - mn < B
    // 0 <= y - x < B
    r1.value.insert(r1.value.end(), r2.value.begin(), r2.value.end());
    r1.mac_value.insert(r1.mac_value.end(), r2.mac_value.begin(),
                        r2.mac_value.end());
    zkp_range(r1.mac_value, r1.value, B_maximum(2 * sizeof(Int)), 2 * sz);
  }

  template <typename Int>
  void zkp_minimum(const QuantizedValueProver<FieldT, Int> &mx,
                   const QuantizedValueProver<FieldT, Int> &input,
                   const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    // y = min(x, mx)
    size_t sz = output.size();
    QuantizedValueProver<FieldT, Int> r1 = mx - output, r2 = input - output;
    // (mx - x)(x - y) == 0
    vector<FieldT> co = co_minimum<FieldT>(sz);
    zkp_poly_deg2(r1.mac_value, r1.value, r2.mac_value, r2.value, co);
    // 0 <= mx - y < B
    // 0 <= x - y < B
    r1.value.insert(r1.value.end(), r2.value.begin(), r2.value.end());
    r1.mac_value.insert(r1.mac_value.end(), r2.mac_value.begin(),
                        r2.mac_value.end());
    zkp_range(r1.mac_value, r1.value, B_minimum(2 * sizeof(Int)), 2 * sz);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_sign(const QuantizedValueProver<FieldT, Int> &input,
                const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    size_t sz = output.size();
    // 0 <= y(x - y)
    // r1 = x - y
    QuantizedValueProver<FieldT, Int> r1 = input - output;
    // r2 = y * r1
    QuantizedValueProver<FieldT, Int> r2 = output * r1;
    r2.auth(this);
    // prove y * r1 - r2 == 0
    vector<FieldT> co = {1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, output, value[i]);
      SIDE_PUSH_VAR(right, r1, value[i]);
      SIDE_PUSH_VAR(right, r2, value[i]);
      ZKP_POLY2();
    }
    // prove 0 <= r2 < B
    zkp_range(r2.mac_value, r2.value, B_sign(2 * sizeof(Int)), sz);

    // (y - 1)(y + 1)(y^2 + x^2) == 0
    // r3 = (y - 1)(y + 1) = y ^ 2 - 1
    QuantizedValueProver<FieldT, Int> r3 = output * output - FieldT(1);
    r3.auth(this);
    // prove y^2 - r3 - 1 == 0
    co = {1, -1, -1};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, output, value[i]);
      SIDE_PUSH_VAR(right, output, value[i]);
      SIDE_PUSH_VAR(right, r3, value[i]);
      ZKP_POLY2();
    }
    // r4 = y^2 + x^2
    QuantizedValueProver<FieldT, Int> r4 = output * output + input * input;
    r4.auth(this);
    // prove y^2 + x^2 - r4 == 0
    co = {1, 1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, output, value[i]);
      SIDE_PUSH_VAR(left, input, value[i]);
      SIDE_PUSH_VAR(right, output, value[i]);
      SIDE_PUSH_VAR(right, input, value[i]);
      SIDE_PUSH_VAR(right, r4, value[i]);
      ZKP_POLY2();
    }
    // r5 = r3 * r4;
    QuantizedValueProver<FieldT, Int> r5 = r3 * r4;
    r5.auth(this);
    // prove r3 * r4 - r5 == 0
    co = {1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, r3, value[i]);
      SIDE_PUSH_VAR(right, r4, value[i]);
      SIDE_PUSH_VAR(right, r5, value[i]);
      ZKP_POLY2();
    }
    // r5 == 0
    check_zero(r5.mac_value);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_abs(const QuantizedValueProver<FieldT, Int> &input,
               const QuantizedValueProver<FieldT, Int> &sign,
               const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    // y = abs(x)
    // sign * x - y == 0
    size_t sz = output.size();
    // r = sign * x - y
    QuantizedValueProver<FieldT, long> r = sign * input - output;
    r.auth(this);
    // prove sign * x - y - r == 0
    vector<FieldT> co = {1, -1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, sign, value[i]);
      SIDE_PUSH_VAR(right, input, value[i]);
      SIDE_PUSH_VAR(right, output, value[i]);
      SIDE_PUSH_VAR(right, r, value[i]);
      ZKP_POLY2();
    }
    // prove r == 0
    check_zero(r.mac_value);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_right_shift(const QuantizedValueProver<FieldT, Int> &input,
                       const QuantizedValueProver<FieldT, Int> &bit,
                       const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    // y = x >> b
    size_t sz = output.size();
    vector<FieldT> t = {2}, mac_t = {0};
    QuantizedValueProver<FieldT, Int> two(t, mac_t, {1, 1});
    // r1 = 2^b
    QuantizedValueProver<FieldT, Int> r1 = two ^ bit;
    r1.auth(this);
    // TODO: prove r1 == 2^b

    // 0 <= x - y * r1 <= r1 - 1
    // r2 = y * r1
    QuantizedValueProver<FieldT, Int> r2 = output * r1;
    r2.auth(this);
    // prove y * r1 - r2 == 0
    vector<FieldT> co = {1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, output, value[i]);
      SIDE_PUSH_VAR(right, r1, value[i]);
      SIDE_PUSH_VAR(right, r2, value[i]);
      ZKP_POLY2();
    }
    // prove 0 <= r2 <= r1 - 1
    zkp_range(r2.mac_value, r2.value, B_shift(2 * sizeof(Int)), sz);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_round(const NormFPProver<FieldT> &fp,
                 const QuantizedValueProver<FieldT, Int> &input,
                 const QuantizedValueProver<FieldT, Int> &output) {
    // y = round(2^e/c * x)
    // 0 <= 2 * 2^e * x - 2cy + c <= 2c - 1
    size_t sz = output.size();
    // r = 2 * 2^e * x - 2cy + c
    QuantizedValueProver<FieldT, Int> r =
        (input * fp.power - output * fp.coeff) * FieldT(2) + fp.coeff;
    r.auth(this);
    // prove 2 * 2^e * x - 2cy + c - r == 0
    vector<FieldT> co = {2, -2, 1, -1, 0};
    vector<FieldT> left, left_mac;
    SIDE_PUSH_VAR(left, fp, power);
    SIDE_PUSH_VAR(left, fp, coeff);
    for (size_t i = 0; i < sz; i++) {
      vector<FieldT> right, right_mac;
      SIDE_PUSH_VAR(right, input, value[i]);
      SIDE_PUSH_VAR(right, output, value[i]);
      SIDE_PUSH_VAR(right, fp, coeff);
      SIDE_PUSH_VAR(right, r, value[i]);
      if (r.value[i] < 0 || r.value[i] >= B_round()) {
        input.value[i].print();
        output.value[i].print();
        (-r.value[i]).print();
        puts("");
      }
    }
    // prove 0 <= r <= 2c - 1
    zkp_range(r.mac_value, r.value, B_round(), sz);
  }

  template <typename Int>
  void zkp_round(const NormFPProver<FieldT> &fp,
                 const QuantizedValueProver<FieldT, Int> &output) {
    // y = round(2^e/c)
    // 0 <= 2 * 2^e - 2cy + c <= 2c - 1
    size_t sz = output.size();
    // r = 2 * 2^e - 2cy + c
    QuantizedValueProver<FieldT, Int> r =
        (output * FieldT(-1) * fp.coeff + fp.power) * FieldT(2) + fp.coeff;
    r.auth(this);
    // prove -2cy + 2 * 2^e + c - r == 0
    vector<FieldT> co = {-2, 2, 1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, fp, coeff);
      SIDE_PUSH_VAR(right, output, value[i]);
      SIDE_PUSH_VAR(right, fp, power);
      SIDE_PUSH_VAR(right, fp, coeff);
      SIDE_PUSH_VAR(right, r, value[i]);
      ZKP_POLY2();
    }
    zkp_range(r.mac_value, r.value, B_round(), sz);
  }

  template <typename Int>
  void zkp_round(const QuantizedValueProver<FieldT, Int> &input1,
                 const QuantizedValueProver<FieldT, Int> &input2,
                 const QuantizedValueProver<FieldT, Int> &output) {
    // y = round(x1 / x2)
    // 0 <= 2x1 - 2x2*y + x2 <= 2x2 - 1
    size_t sz = output.size();
    size_t f2 = sz / input2.size();
    // r = 2x1 - 2x2*y + x2
    QuantizedValueProver<FieldT, Int> r =
        (input1 - input2 * output) * FieldT(2) + input2;
    r.auth(this);
    // prove -2x2*y + 2x1 + x2 - r == 0
    vector<FieldT> co = {-2, 2, 1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, output, value[i]);
      SIDE_PUSH_VAR(right, input2, value[i / f2]);
      SIDE_PUSH_VAR(right, input1, value[i]);
      SIDE_PUSH_VAR(right, input2, value[i / f2]);
      SIDE_PUSH_VAR(right, r, value[i]);
      ZKP_POLY2();
    }
    // prove 0 <= r <= 2x2 - 1
    zkp_range(r.mac_value, r.value, B_round2(2 * sizeof(Int)), sz);
  }

  template <typename Int>
  void zkp_floor(const NormFPProver<FieldT> &fp,
                 const QuantizedValueProver<FieldT, Int> &output) {
    // y = floor(2^e/c)
    // 0 <= 2^e - cy <= c
    size_t sz = output.size();
    // r = 2^e - cy
    QuantizedValueProver<FieldT, Int> r =
        output * FieldT(-1) * fp.coeff + fp.power;
    r.auth(this);
    // prove -cy + 2^e - r == 0
    vector<FieldT> co = {-1, 1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, fp, coeff);
      SIDE_PUSH_VAR(right, output, value[i]);
      SIDE_PUSH_VAR(right, fp, power);
      SIDE_PUSH_VAR(right, r, value[i]);
      ZKP_POLY2();
    }
    // prove 0 <= r <= c
    zkp_range(r.mac_value, r.value, B_floor(), sz);
  }

  template <typename Int>
  void zkp_floor(const NormFPProver<FieldT> &fp,
                 const QuantizedValueProver<FieldT, Int> &input,
                 const QuantizedValueProver<FieldT, Int> &output) {
    // y = floor(2^e/c * x)
    // 0 <= 2^e * x - cy <= c
    size_t sz = output.size();
    // r = 2^e * x - cy
    QuantizedValueProver<FieldT, Int> r = input * fp.power - output * fp.coeff;
    r.auth(this);
    // prove 2^e * x - cy - r == 0
    vector<FieldT> co = {1, -1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, fp, power);
      SIDE_PUSH_VAR(left, fp, coeff);
      SIDE_PUSH_VAR(right, input, value[i]);
      SIDE_PUSH_VAR(right, output, value[i]);
      SIDE_PUSH_VAR(right, r, value[i]);
      ZKP_POLY2();
    }
    // prove 0 <= r <= c
    zkp_range(r.mac_value, r.value, B_floor(), sz);
  }

  template <typename Int>
  void zkp_sqrt(const QuantizedValueProver<FieldT, Int> &input,
                const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    // y = floor(sqrt(x))
    size_t sz = output.size();
    // r = x - y^2
    QuantizedValueProver<FieldT, Int> r = input - output * output;
    r.auth(this);
    // prove -y * y + x - r == 0
    vector<FieldT> co = {-1, 1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, output, value[i]);
      SIDE_PUSH_VAR(right, output, value[i]);
      SIDE_PUSH_VAR(right, input, value[i]);
      SIDE_PUSH_VAR(right, r, value[i]);
      ZKP_POLY2();
    }
    // prove 0 <= r <= B
    zkp_range(r.mac_value, r.value, B_std(2 * sizeof(Int)), sz);

    // r = (y + 1) ^ 2 - x
    r = (output + FieldT(1)) * (output + FieldT(1)) - input;
    r.auth(this);
    // prove y^2 + 2*y - x - r + 1 == 0
    co = {1, 2, -1, -1, 1};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, output, value[i]);
      SIDE_PUSH_VAR(right, output, value[i]);
      SIDE_PUSH_VAR(right, output, value[i]);
      SIDE_PUSH_VAR(right, input, value[i]);
      SIDE_PUSH_VAR(right, r, value[i]);
      ZKP_POLY2();
    }
    zkp_range(r.mac_value, r.value, B_std(2 * sizeof(Int)), sz);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_mul(const QuantizedValueProver<FieldT, Int> &input1,
               const QuantizedValueProver<FieldT, Int> &input2,
               const QuantizedValueProver<FieldT, Int> &output) {
    // y = x1 * x2
    // prove x1 * x2 - y1 == 0
    size_t sz = output.size();
    vector<FieldT> co = {1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, input1, value[i]);
      SIDE_PUSH_VAR(right, input2, value[i]);
      SIDE_PUSH_VAR(right, output, value[i]);
      ZKP_POLY2();
    }
  }

  template <typename Int>
  void zkp_mul(const FieldT &input1, const FieldT &mac_input1,
               const QuantizedValueProver<FieldT, Int> &input2,
               const QuantizedValueProver<FieldT, Int> &output) {
    // y = x1 * x2
    // prove x1 * x2 - y1 == 0
    size_t sz = output.size();
    vector<FieldT> co = {1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH(left, input1);
      SIDE_PUSH_VAR(right, input2, value[i]);
      SIDE_PUSH_VAR(right, output, value[i]);
      ZKP_POLY2();
    }
  }

  void zkp_mul(const FieldT &input1, const FieldT &mac_input1,
               const FieldT &input2, const FieldT &mac_input2,
               const FieldT &output, const FieldT &mac_output) {
    // y = x1 * x2
    // prove x1 * x2 - y1 == 0
    vector<FieldT> co = {1, -1, 0};
    SIDE_DECL();
    SIDE_PUSH(left, input1);
    SIDE_PUSH(right, input2);
    SIDE_PUSH(right, output);
    ZKP_POLY2();
  }

  template <typename Int>
  void zkp_relu(const QuantizedValueProver<FieldT, Int> &input,
                const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    vector<FieldT> x = {0}, mac_x = {0};
    QuantizedValueProver<FieldT, Int> mn(x, mac_x, {1, 1, 1, 1});  // 0
    zkp_maximum(mn, input, output);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_add(const NormFPProver<FieldT> &fp1, const NormFPProver<FieldT> &fp2,
               const QuantizedValueProver<FieldT, Int> &input1,
               const QuantizedValueProver<FieldT, Int> &input2,
               const QuantizedValueProver<FieldT, Int> &output1,
               const QuantizedValueProver<FieldT, Int> &output2,
               const QuantizedValueProver<FieldT, Int> &output) {
    // y1 = round(2^e1/c1 * x1)
    zkp_round(fp1, input1, output1);
    // y2 = round(2^e2/c2 * x2)
    zkp_round(fp2, input2, output2);
    // y = y1 + y2
  }

  template <typename Int>
  void zkp_skip_add(const SkipAddPrvModel<FieldT> &model,
                    const QuantizedValueProver<FieldT, Int> &input1,
                    const QuantizedValueProver<FieldT, Int> &input2,
                    const QuantizedValueProver<FieldT, Int> &output1,
                    const QuantizedValueProver<FieldT, Int> &output2,
                    const QuantizedValueProver<FieldT, Int> &output) {
    zkp_add(model.fp1, model.fp2, input1, input2, output1, output2, output);
  }

  template <typename Int>
  void zkp_res(const ResPrvModel<FieldT> &model,
               const QuantizedValueProver<FieldT, Int> &input1,
               const QuantizedValueProver<FieldT, Int> &input2,
               const QuantizedValueProver<FieldT, Int> &output1,
               const QuantizedValueProver<FieldT, Int> &output2,
               const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    zkp_add(model.fp1, model.fp2, input1, input2, output1, output2, output);
    TIME_STATS_END
  }

  template <typename Int>
  void maxpool_calc_R(vector<FieldT> &mac_R, vector<FieldT> &R,
                      vector<vector<FieldT>> &mid_res,
                      const QuantizedValueProver<FieldT, Int> &input,
                      const QuantizedValueProver<FieldT, Int> &output,
                      const size_t padding, const size_t stride,
                      const size_t window) {
    assert(input.shape[0] == 1 && output.shape[0] == 1);
    size_t input_d = input.shape[1];
    size_t input_l = input.shape[2];
    size_t input_w = input.shape[3];
    size_t output_d = output.shape[1];
    size_t output_l = output.shape[2];
    size_t output_w = output.shape[3];

    size_t output_size = output.size();
    assert(output_size == output_d * output_l * output_w);

    mid_res.resize(output_size);

    for (size_t k = 0; k < output_d; k++) {
      for (size_t i = 0; i < output_l; i++) {
        for (size_t j = 0; j < output_w; j++) {
          long output_idx = get_3D_idx(output_l, output_w, k, i, j);
          bool first_mid = true;
          FieldT mid = FieldT::one();
          for (size_t m = 0; m < window; m++) {
            for (size_t n = 0; n < window; n++) {
              long input_l_idx = (long)m - padding + stride * i;
              long input_w_idx = (long)n - padding + stride * j;
              long input_idx =
                  get_3D_idx(input_l, input_w, k, input_l_idx, input_w_idx);
              if (input_l_idx < 0 || input_l_idx >= input_l) continue;
              if (input_w_idx < 0 || input_w_idx >= input_w) continue;
              FieldT mac_o_minus_i;
              FieldT o_minus_i;
              auth_mac_sub(mac_o_minus_i, o_minus_i,
                           output.mac_value[output_idx],
                           output.value[output_idx], input.mac_value[input_idx],
                           input.value[input_idx]);
              mac_R.push_back(mac_o_minus_i);
              R.push_back(o_minus_i);
              mid *= o_minus_i;
              if (first_mid == false) {
                mid_res[output_idx].push_back(mid);
              }
              if (first_mid == true) {
                first_mid = false;
              }
            }
          }
        }
      }
    }
  }

  template <typename Int>
  void zkp_maxpool(const QuantizedValueProver<FieldT, Int> &input,
                   const QuantizedValueProver<FieldT, Int> &output,
                   const size_t padding = 1, const size_t stride = 2,
                   const size_t window = 3) {
    vector<FieldT> mac_R, R;
    vector<vector<FieldT>> mac_mid_res, mid_res;
    maxpool_calc_R(mac_R, R, mid_res, input, output, padding, stride, window);
    zkp_range(mac_R, R, B_maxpool(2 * sizeof(Int)), R.size());

    size_t output_size = output.size();
    mac_mid_res.resize(output_size);
    for (size_t i = 0; i < output_size; i++)
      authenticated_val_input(mac_mid_res[i], mid_res[i], mid_res[i].size());

    vector<FieldT> co = {1, -1, 0};
    size_t idx = 0;
    for (size_t i = 0; i < output_size; i++) {
      FieldT res_x = R[idx];
      FieldT mac_res_x = mac_R[idx];
      idx++;
      for (size_t j = 0; j < mid_res[i].size(); j++) {
        SIDE_DECL();
        SIDE_PUSH(left, res_x);
        SIDE_PUSH(right, R[idx]);
        SIDE_PUSH(right, mid_res[i][j]);
        mac_res_x = mac_mid_res[i][j];
        res_x = mid_res[i][j];
        idx++;
        ZKP_POLY2();
      }
    }
  }

  template <typename Int>
  void fc_calc_x(vector<FieldT> &mac_x, vector<FieldT> &x,
                 const vector<FieldT> &u,
                 const QuantizedValueProver<FieldT, Int> &A, size_t m) {
    size_t n = 1;
    assert(mac_x.size() == m + 1);
    assert(mac_x.size() == x.size());
    assert(u.size() == n);

    // x.T = u.T * A
    // #pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
      FieldT mid, mac_mid;
      for (size_t j = 0; j < n; j++) {
        size_t k = j * m + i;
        auth_pub_mul(mac_mid, mid, A.mac_value[k], A.value[k], u[j]);
        auth_mac_add(mac_x[i], x[i], mac_x[i], x[i], mac_mid, mid);
      }
    }
    // for bias, x[m] = sum(u)
    if (mac_x.size() == m + 1) {
      for (size_t j = 0; j < n; j++) {
        x[m] += u[j];
      }
    }
  }

  template <typename Int>
  void fc_calc_y(vector<FieldT> &mac_y, vector<FieldT> &y,
                 const vector<FieldT> &v,
                 const FCPrvModel<FieldT, Int> &model) {
    size_t m = model.weight.shape[0];
    size_t l = model.weight.shape[1];
    assert(mac_y.size() == m + 1 || mac_y.size() == y.size());
    assert(v.size() == l);

    FieldT neg_sum_v = FieldT(0);
    for (size_t i = 0; i < v.size(); i++) neg_sum_v -= v[i];

// y = B * v
#pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < l; j++) {
        size_t k = i * l + j;
        FieldT mid, mac_mid;
        auth_pub_mul(mac_mid, mid, model.weight.mac_value[k],
                     model.weight.value[k], v[j]);
        auth_mac_add(mac_y[i], y[i], mac_y[i], y[i], mac_mid, mid);
      }
    }
    // bias
    FieldT mid, mac_mid;
    for (size_t j = 0; j < l; j++) {
      auth_pub_mul(mac_mid, mid, model.bias.mac_value[j], model.bias.value[j],
                   v[j]);
      auth_mac_add(mac_y[m], y[m], mac_y[m], y[m], mac_mid, mid);
    }
  }

  void fc_calc_z(FieldT &mac_z, FieldT &z, const vector<FieldT> &u,
                 const vector<FieldT> &v, const vector<FieldT> &mac_C,
                 const vector<FieldT> &C, size_t n, size_t l) {
    assert(u.size() == n);
    assert(v.size() == l);

    // z = u.T * C * v
    for (size_t i = 0; i < l; i++) {
      // t = u.T * C
      FieldT t, mac_t;
      FieldT mid, mac_mid;
      for (size_t j = 0; j < n; j++) {
        size_t k = j * l + i;
        auth_pub_mul(mac_mid, mid, mac_C[k], C[k], u[j]);
        auth_mac_add(mac_t, t, mac_t, t, mac_mid, mid);
      }
      // z += t * v
      auth_pub_mul(mac_t, t, mac_t, t, v[i]);
      auth_mac_add(mac_z, z, mac_z, z, mac_t, t);
    }
  }

  template <typename Int>
  void zkp_fc(const FCPrvModel<FieldT, Int> &model,
              const QuantizedValueProver<FieldT, Int> &input,
              const QuantizedValueProver<FieldT, Int> &output) {
    size_t n = 1, l = model.weight.shape[1], m = model.weight.shape[0];
    size_t sz = n * l;
    vector<FieldT> mid(sz), mac_mid;

    const int min_size_multi_thread = 576;
    const int max_thread_num = 16;
    int grain_size;
    int thread_num;

    if (sz < min_size_multi_thread) {
      thread_num = 1;
      grain_size = sz;
    } else if (sz < min_size_multi_thread * max_thread_num) {
      thread_num = CEIL(sz, min_size_multi_thread);
      grain_size = min_size_multi_thread;
    } else {
      thread_num = max_thread_num;
      grain_size = CEIL(sz, max_thread_num);
    }
    vector<thread> threads(thread_num);
    for (int trd_idx = 0; trd_idx < thread_num; trd_idx++) {
      threads[trd_idx] = thread([=, &mid, &model, &input, &output]() {
        size_t start = min((size_t)trd_idx * grain_size, sz);
        size_t end = min((size_t)(trd_idx + 1) * grain_size, sz);
        for (size_t idx = start; idx < end; idx++) {
          // idx = i * l + j
          size_t i = idx / l;
          size_t j = idx % l;
          signed imid = 0;
          for (size_t k = 0; k < m; k++) {
            imid += input.ivalue[i * m + k] * model.weight.ivalue[k * l + j];
          }
          mid[idx] = FieldT(imid) + model.bias.value[j];
        }
      });
    }
    for (auto &t : threads) t.join();
    authenticated_val_input(mac_mid, mid, sz);

    vector<FieldT> u(n), v(l);
    run_time += omp_get_wtime() - s_time;
    io->recv_data(&u[0], n * sizeof(FieldT));
    io->recv_data(&v[0], l * sizeof(FieldT));
    recv_size += (n + l) * sizeof(FieldT);
    s_time = omp_get_wtime();

    size_t len = m + 1;
    vector<FieldT> mac_x(len), x(len), mac_y(len), y(len);
    FieldT mac_z, z;
    // x = u.T @ in1
    fc_calc_x(mac_x, x, u, input, m);
    // y = w @ v
    fc_calc_y(mac_y, y, v, model);
    // z = u.T @ mid @ v
    fc_calc_z(mac_z, z, u, v, mac_mid, mid, n, l);

    // x @ y - z == 0
    vector<FieldT> co(m, FieldT(1)), left, left_mac, right, right_mac;
    co.push_back(x[m]);
    co.push_back(FieldT(-1));
    co.push_back(FieldT(0));
    left = x;
    left_mac = mac_x;
    left.pop_back();
    left_mac.pop_back();
    right = y;
    right_mac = mac_y;
    right.push_back(z);
    right_mac.push_back(mac_z);
    zkp_poly_deg2(left_mac, left, right_mac, right, co);

    zkp_round(model.fp,
              QuantizedValueProver<FieldT, Int>(mid, mac_mid, output.shape),
              output);
  }

  template <typename Int>
  void conv_calc_R(vector<FieldT> &mid, const ConvPrvModel<FieldT, Int> &model,
                   const QuantizedValueProver<FieldT, Int> &input,
                   const QuantizedValueProver<FieldT, Int> &output,
                   size_t thread_num) {
    size_t weight_nd = model.weight.shape[0];
    size_t weight_d = model.weight.shape[1];
    size_t weight_l = model.weight.shape[2];
    size_t weight_w = model.weight.shape[3];
    size_t output_l = output.shape[2];
    size_t output_w = output.shape[3];
    size_t input_l = input.shape[2];
    size_t input_w = input.shape[3];
    size_t padding = model.padding;
    size_t stride = model.stride;

    size_t output_size = weight_nd * output_l * output_w;
    vector<signed> imid;
    mid.resize(output_size);

    vector<thread> threads(thread_num);
    const int grain_size = (output_size + thread_num - 1) / thread_num;

    for (size_t i = 0; i < thread_num; i++) {
      threads[i] = thread([=, &mid, &model, &input, &output]() {
        size_t start = i * grain_size;
        size_t end = (i + 1) * grain_size;
        if (start >= output_size) start = output_size;
        if (end >= output_size) end = output_size;
        FieldT e1 = FieldT(2) ^ (e - 1);
        FieldT e2 = e1 * FieldT(2);
        for (size_t idx = start; idx < end; idx++) {
          size_t k = (idx / (output_w * output_l));
          size_t i = (idx / output_w) % output_l;
          size_t j = idx % output_w;

          mid[idx] = model.bias.value[k];
          signed imid = 0;

          for (size_t s = 0; s < weight_d * weight_l * weight_w; s++) {
            size_t t = s / (weight_w * weight_l);
            size_t m = (s / weight_w) % weight_l;
            size_t n = s % weight_w;

            long weight_idx =
                get_4D_idx(weight_d, weight_l, weight_w, k, t, m, n);
            long input_l_idx = (long)m - padding + stride * i;
            long input_w_idx = (long)n - padding + stride * j;
            if (input_l_idx < 0 || input_l_idx >= input_l) continue;
            if (input_w_idx < 0 || input_w_idx >= input_w) continue;
            long input_idx =
                get_3D_idx(input_l, input_w, t, input_l_idx, input_w_idx);
            imid += model.weight.ivalue[weight_idx] * input.ivalue[input_idx];
          }

          mid[idx] += FieldT(imid);
        }
      });
    }
    for (size_t i = 0; i < thread_num; i++) threads[i].join();
  }

  void conv_get_poly_co_R(vector<FieldT> &co) {
    co.resize(4);
    co[0] = FieldT(-1);
    co[1] = FieldT(-1);
    co[2] = (FieldT(2) ^ (e));
    co[3] = (FieldT(2) ^ (e - 1));
  }

  template <typename Int>
  void conv_get_poly_value_R(vector<FieldT> &left_mac, vector<FieldT> &left_x,
                             vector<FieldT> &right_mac, vector<FieldT> &right_x,
                             const vector<FieldT> &mac_R,
                             const vector<FieldT> &R,
                             const vector<FieldT> &mac_mid,
                             const vector<FieldT> &mid, size_t idx,
                             const ConvPrvModel<FieldT, Int> &model,
                             const QuantizedValueProver<FieldT, Int> &output) {
    left_mac.resize(1);
    left_x.resize(1);
    right_mac.resize(3);
    right_x.resize(3);
    left_mac[0] = model.mac_E;
    left_x[0] = model.E;
    right_mac[0] = mac_mid[idx];
    right_x[0] = mid[idx];
    right_mac[1] = mac_R[idx];
    right_x[1] = R[idx];
    right_mac[2] = output.mac_value[idx];
    right_x[2] = output.value[idx];
  }

  template <typename Int>
  void conv_calc_x(vector<FieldT> &mac_x, vector<FieldT> &x,
                   const vector<FieldT> &u,
                   const ConvPrvModel<FieldT, Int> &model) {
    size_t weight_nd = model.weight.shape[0];
    size_t weight_d = model.weight.shape[1];
    size_t weight_l = model.weight.shape[2];
    size_t weight_w = model.weight.shape[3];

    size_t m = weight_d * weight_l * weight_w;
    FieldT sum_u = FieldT::zero();
    for (size_t i = 0; i < u.size(); i++) sum_u += u[i];

#pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < weight_nd; j++) {
        FieldT mac_mid;
        FieldT mid;
        auth_pub_mul(mac_mid, mid, model.weight.mac_value[j * m + i],
                     model.weight.value[j * m + i], u[j]);
        auth_mac_add(mac_x[i], x[i], mac_x[i], x[i], mac_mid, mid);
      }
    }
    auth_pub_mul(mac_x[m], x[m], model.bias.mac_value[0], model.bias.value[0],
                 u[0]);
    for (size_t j = 1; j < weight_nd; j++) {
      FieldT mac_mid;
      FieldT mid;
      auth_pub_mul(mac_mid, mid, model.bias.mac_value[j], model.bias.value[j],
                   u[j]);
      auth_mac_add(mac_x[m], x[m], mac_x[m], x[m], mac_mid, mid);
    }
  }

  template <typename Int>
  void conv_calc_y(vector<FieldT> &mac_y, vector<FieldT> &y,
                   const vector<FieldT> &v,
                   const ConvPrvModel<FieldT, Int> &model,
                   const QuantizedValueProver<FieldT, Int> &input,
                   const QuantizedValueProver<FieldT, Int> &output) {
    size_t weight_d = model.weight.shape[1];
    size_t weight_l = model.weight.shape[2];
    size_t weight_w = model.weight.shape[3];
    size_t output_l = output.shape[2];
    size_t output_w = output.shape[3];
    size_t input_l = input.shape[2];
    size_t input_w = input.shape[3];
    size_t padding = model.padding;
    size_t stride = model.stride;

    size_t m = weight_d * weight_l * weight_w;
    size_t l = output_l * output_w;

    for (size_t idx = 0; idx < l; idx++) {
      size_t i = idx / output_w;
      size_t j = idx % output_w;
#pragma omp parallel for
      for (size_t s = 0; s < m; s++) {
        size_t a = s / (weight_w * weight_l);
        size_t b = (s / weight_w) % weight_l;
        size_t c = s % weight_w;

        long weight_idx = get_3D_idx(weight_l, weight_w, a, b, c);
        long input_l_idx = (long)b - padding + stride * i;
        long input_w_idx = (long)c - padding + stride * j;
        FieldT mac_mid;
        FieldT mid;
        if (input_l_idx < 0 || input_l_idx >= input_l) {
          mac_mid = FieldT::zero();
          mid = FieldT::zero();
        } else if (input_w_idx < 0 || input_w_idx >= input_w) {
          mac_mid = FieldT::zero();
          mid = FieldT::zero();
        } else {
          long input_idx =
              get_3D_idx(input_l, input_w, a, input_l_idx, input_w_idx);
          auth_pub_mul(mac_mid, mid, input.mac_value[input_idx],
                       input.value[input_idx], v[idx]);
        }
        if (idx == 0) {
          mac_y[weight_idx] = mac_mid;
          y[weight_idx] = mid;
        } else
          auth_mac_add(mac_y[weight_idx], y[weight_idx], mac_y[weight_idx],
                       y[weight_idx], mac_mid, mid);
      }
      if (idx == 0)
        y[m] = v[idx];
      else
        y[m] += v[idx];
    }
  }

  template <typename Int>
  void conv_calc_z(FieldT &mac_z, FieldT &z, const vector<FieldT> &u,
                   const vector<FieldT> &v, const vector<FieldT> &mac_c,
                   const vector<FieldT> &c,
                   const QuantizedValueProver<FieldT, Int> &output) {
    size_t output_d = output.shape[1];
    size_t output_l = output.shape[2];
    size_t output_w = output.shape[3];

    size_t n = output_d;
    size_t l = output_l * output_w;
    vector<FieldT> mac_x(l);
    vector<FieldT> x(l);

#pragma omp parallel for
    for (size_t i = 0; i < l; i++) {
      auth_pub_mul(mac_x[i], x[i], mac_c[i], c[i], u[0]);
      for (size_t j = 1; j < n; j++) {
        FieldT mac_mid;
        FieldT mid;
        auth_pub_mul(mac_mid, mid, mac_c[j * l + i], c[j * l + i], u[j]);
        auth_mac_add(mac_x[i], x[i], mac_x[i], x[i], mac_mid, mid);
      }
    }

    auth_pub_mul(mac_z, z, mac_x[0], x[0], v[0]);
    for (size_t i = 1; i < l; i++) {
      FieldT mac_mid;
      FieldT mid;
      auth_pub_mul(mac_mid, mid, mac_x[i], x[i], v[i]);
      auth_mac_add(mac_z, z, mac_z, z, mac_mid, mid);
    }
  }

  void conv_get_poly_co_xy(vector<FieldT> &co_xy, size_t m, const FieldT &ym) {
    co_xy.resize(m + 3);
    for (size_t i = 0; i < m; i++) co_xy[i] = FieldT::one();
    co_xy[m] = ym;
    co_xy[m + 1] = -FieldT::one();
    co_xy[m + 2] = FieldT::zero();
  }

  void conv_get_poly_value_xy(vector<FieldT> &left_mac, vector<FieldT> &left_x,
                              vector<FieldT> &right_mac,
                              vector<FieldT> &right_x,
                              const vector<FieldT> &mac_x,
                              const vector<FieldT> &x,
                              const vector<FieldT> &mac_y,
                              const vector<FieldT> &y, const FieldT &mac_z,
                              const FieldT &z) {
    size_t m = mac_x.size() - 1;
    left_mac.resize(m);
    left_x.resize(m);
    right_mac.resize(m + 2);
    right_x.resize(m + 2);
    for (size_t i = 0; i < m; i++) {
      left_mac[i] = mac_y[i];
      left_x[i] = y[i];
      right_mac[i] = mac_x[i];
      right_x[i] = x[i];
    }
    right_mac[m] = mac_x[m];
    right_x[m] = x[m];
    right_mac[m + 1] = mac_z;
    right_x[m + 1] = z;
  }

  template <typename Int>
  void zkp_conv(const ConvPrvModel<FieldT, Int> &model,
                const QuantizedValueProver<FieldT, Int> &input,
                const QuantizedValueProver<FieldT, Int> &output,
                size_t thread_num = 16) {
    vector<FieldT> mac_mid;
    vector<FieldT> mid;

    double itime, ftime;
    itime = omp_get_wtime();
    conv_calc_R(mid, model, input, output, thread_num);

    authenticated_val_input(mac_mid, mid, mid.size());

    size_t n = model.weight.shape[0];
    size_t m =
        model.weight.shape[1] * model.weight.shape[2] * model.weight.shape[3];
    size_t l = output.shape[2] * output.shape[3];

    vector<FieldT> u(n);
    vector<FieldT> v(l);
    run_time += omp_get_wtime() - s_time;
    io->recv_data(&u[0], n * sizeof(FieldT));
    io->recv_data(&v[0], l * sizeof(FieldT));
    recv_size += (n + l) * sizeof(FieldT);
    s_time = omp_get_wtime();

    vector<FieldT> mac_x(m + 1);
    vector<FieldT> x(m + 1);
    vector<FieldT> mac_y(m + 1);
    vector<FieldT> y(m + 1);
    FieldT mac_z;
    FieldT z;

    conv_calc_x(mac_x, x, u, model);
    conv_calc_y(mac_y, y, v, model, input, output);
    conv_calc_z(mac_z, z, u, v, mac_mid, mid, output);

    vector<FieldT> co_xy;
    conv_get_poly_co_xy(co_xy, m, y[m]);
    vector<FieldT> left_mac;
    vector<FieldT> left_x;
    vector<FieldT> right_mac;
    vector<FieldT> right_x;
    conv_get_poly_value_xy(left_mac, left_x, right_mac, right_x, mac_x, x,
                           mac_y, y, mac_z, z);
    zkp_poly_deg2(left_mac, left_x, right_mac, right_x, co_xy);

    zkp_round(model.fp,
              QuantizedValueProver<FieldT, Int>(mid, mac_mid, output.shape),
              output);
  }

  template <typename Int>
  void zkp_mean(const QuantizedValueProver<FieldT, Int> &input,
                const QuantizedValueProver<FieldT, Int> &mean) {
    TIME_STATS_BEG
    size_t d1 = input.shape[0];
    size_t d2 = input.shape[1];
    vector<FieldT> R(d1), mac_R(d1);

    // R = 2 * sum(x) - 2d * mean + d
    for (size_t i = 0; i < d1; i++) {
      FieldT sum, mac_sum;
      for (size_t j = 0; j < d2; j++) {
        auth_mac_add(mac_sum, sum, input.mac_value[i * d2 + j],
                     input.value[i * d2 + j], mac_sum, sum);
      }
      // R = 2 * sum
      auth_pub_mul(mac_R[i], R[i], mac_sum, sum, FieldT(2));
      // R -= 2d * mean
      FieldT mid, mac_mid;
      auth_pub_mul(mac_mid, mid, mean.mac_value[i], mean.value[i],
                   FieldT(2 * d2));
      auth_mac_sub(mac_R[i], R[i], mac_R[i], R[i], mac_mid, mid);
      // R += d
      auth_pub_add(mac_R[i], R[i], mac_R[i], R[i], FieldT(d2));
    }

    // 0 <= R <= 2d
    FieldT B = B_mean(d2);
    zkp_range(mac_R, R, B, d1);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_var(const QuantizedValueProver<FieldT, Int> &input,
               const QuantizedValueProver<FieldT, Int> &mean,
               const QuantizedValueProver<FieldT, Int> &var) {
    TIME_STATS_BEG
    size_t d1 = input.shape[0];
    size_t d2 = input.shape[1];
    vector<FieldT> R(d1), mac_R;

    // 0 <= 2 * sum((x-mean)^2) - 2d * var + d < 2d
    for (size_t i = 0; i < d1; i++) {
      FieldT sum = FieldT::zero();
      for (size_t j = 0; j < d2; j++) {
        sum += (input.value[i * d2 + j] - mean.value[i]) ^ 2;
      }
      R[i] = FieldT(2) * sum - FieldT(2 * d2) * var.value[i] + FieldT(d2);
    }
    authenticated_val_input(mac_R, R, d1);

    vector<FieldT> co = co_variance<FieldT>(d2);
    vector<FieldT> left(d2), left_mac(d2), right(d2 + 2), right_mac(d2 + 2);
    // 2 * sum((x - mean) ^ 2) - 2d * var - R + d < 2d
    for (size_t i = 0; i < d1; i++) {
      for (size_t j = 0; j < d2; j++) {
        left[j] = right[j] = input.value[i * d2 + j] - mean.value[i];
        left_mac[j] = right_mac[j] =
            input.mac_value[i * d2 + j] - mean.mac_value[i];
      }
      right[d2] = var.value[i];
      right_mac[d2] = var.mac_value[i];
      right[d2 + 1] = R[i];
      right_mac[d2 + 1] = mac_R[i];
      zkp_poly_deg2(left_mac, left, right_mac, right, co);
    }

    FieldT B = B_var(d2);
    zkp_range(mac_R, R, B, d1);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_std(const QuantizedValueProver<FieldT, Int> &var,
               const QuantizedValueProver<FieldT, Int> &std) {
    TIME_STATS_BEG
    zkp_sqrt(var, std);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_std_max(const QuantizedValueProver<FieldT, Int> &std,
                   const QuantizedValueProver<FieldT, Int> &std_max) {
    TIME_STATS_BEG
    vector<FieldT> x = {1}, mac_x = {0};
    zkp_maximum(QuantizedValueProver<FieldT, Int>(x, mac_x, {1, 1}), std,
                std_max);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_norm(const NormFPProver<FieldT> &fp,
                const QuantizedValueProver<FieldT, Int> &input,
                const QuantizedValueProver<FieldT, Int> &mean,
                const QuantizedValueProver<FieldT, Int> &std_max,
                const QuantizedValueProver<FieldT, Int> &sub,
                const QuantizedValueProver<FieldT, Int> &norm) {
    TIME_STATS_BEG
    // sub = round(2^e/c * (input - mean))
    zkp_round(fp, input - mean, sub);
    // y = round(sub / std_max)
    zkp_round(sub, std_max, norm);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_layer_norm_out(const LayerNormPrvModel<FieldT, Int> &model,
                          const QuantizedValueProver<FieldT, Int> &input,
                          const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    // y = round(x * w + b)
    assert(input.shape.size() == 2);
    assert(output.shape.size() == 2);
    assert(input.shape == output.shape);
    size_t sz = output.size();
    size_t d1 = output.shape[0], d2 = output.shape[1];
    // r = x * w + b
    QuantizedValueProver<FieldT, Int> r(sz);
    r.shape = output.shape;
    for (size_t i = 0; i < d1; i++) {
      for (size_t j = 0; j < d2; j++) {
        size_t k = i * d2 + j;
        r.value[k] =
            input.value[k] * model.weight.value[j] + model.bias.value[j];
      }
    }
    r.auth(this);
    // prove x * w + b - r == 0
    vector<FieldT> co = {1, 1, -1, 0};
    for (size_t i = 0; i < d1; i++) {
      for (size_t j = 0; j < d2; j++) {
        size_t k = i * d2 + j;
        SIDE_DECL();
        SIDE_PUSH_VAR(left, input, value[k]);
        SIDE_PUSH_VAR(right, model.weight, value[j]);
        SIDE_PUSH_VAR(right, model.bias, value[j]);
        SIDE_PUSH_VAR(right, r, value[k]);
        ZKP_POLY2();
      }
    }
    // prove y == round(r)
    zkp_round(model.fp2, r, output);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_layer_norm(const LayerNormPrvModel<FieldT, Int> &model,
                      const QuantizedValueProver<FieldT, Int> &input,
                      const QuantizedValueProver<FieldT, Int> &mean,
                      const QuantizedValueProver<FieldT, Int> &var,
                      const QuantizedValueProver<FieldT, Int> &std,
                      const QuantizedValueProver<FieldT, Int> &std_max,
                      const QuantizedValueProver<FieldT, Int> &sub,
                      const QuantizedValueProver<FieldT, Int> &norm,
                      const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    // mean
    zkp_mean(input, mean);
    // variance
    zkp_var(input, mean, var);
    // standard
    zkp_std(var, std);
    // maximum
    zkp_std_max(std, std_max);
    // norm
    zkp_norm(model.fp1, input, mean, std_max, sub, norm);
    // x * g + b
    zkp_layer_norm_out(model, norm, output);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_layer_norm(const LayerNormPrvModel<FieldT, Int> &model,
                      const QuantizedValueProver<FieldT, Int> &input,
                      const QuantizedValueProver<FieldT, Int> &mean,
                      const QuantizedValueProver<FieldT, Int> &var,
                      const QuantizedValueProver<FieldT, Int> &std,
                      const QuantizedValueProver<FieldT, Int> &std_max,
                      const QuantizedValueProver<FieldT, Int> &sub,
                      const QuantizedValueProver<FieldT, Int> &norm) {
    TIME_STATS_BEG
    // mean
    zkp_mean(input, mean);
    // variance
    zkp_var(input, mean, var);
    // standard
    zkp_std(var, std);
    // maximum
    zkp_std_max(std, std_max);
    // norm
    zkp_norm(model.fp1, input, mean, std_max, sub, norm);
    TIME_STATS_END
  }

  template <typename Int>
  void calc_x(vector<FieldT> &mac_x, vector<FieldT> &x, const vector<FieldT> &u,
              const QuantizedValueProver<FieldT, Int> &A) {
    size_t n = A.shape[0];
    size_t m = A.shape[1];
    assert(mac_x.size() == m + 1 || mac_x.size() == m);
    assert(mac_x.size() == x.size());
    assert(u.size() == n);

    // x.T = u.T * A
    // #pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
      FieldT mid, mac_mid;
      for (size_t j = 0; j < n; j++) {
        size_t k = j * m + i;
        auth_pub_mul(mac_mid, mid, A.mac_value[k], A.value[k], u[j]);
        auth_mac_add(mac_x[i], x[i], mac_x[i], x[i], mac_mid, mid);
      }
    }
    // for bias, x[m] = sum(u)
    if (mac_x.size() == m + 1) {
      for (size_t j = 0; j < n; j++) {
        x[m] += u[j];
      }
    }
  }

  template <typename Int>
  void calc_y(vector<FieldT> &mac_y, vector<FieldT> &y, const vector<FieldT> &v,
              const QuantizedValueProver<FieldT, Int> &B,
              const QuantizedValueProver<FieldT, Int> *bias) {
    size_t m = B.shape[0];
    size_t l = B.shape[1];
    assert(mac_y.size() == m + 1 || mac_y.size() == y.size());
    assert(v.size() == l);
    assert(!bias || bias->shape[0] == l);

// y = B * v
#pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < l; j++) {
        size_t k = i * l + j;
        FieldT mid, mac_mid;
        auth_pub_mul(mac_mid, mid, B.mac_value[k], B.value[k], v[j]);
        auth_mac_add(mac_y[i], y[i], mac_y[i], y[i], mac_mid, mid);
      }
    }
    // bias
    if (bias) {
      assert(y.size() == m + 1);
      FieldT mid, mac_mid;
      for (size_t j = 0; j < l; j++) {
        auth_pub_mul(mac_mid, mid, bias->mac_value[j], bias->value[j], v[j]);
        auth_mac_add(mac_y[m], y[m], mac_y[m], y[m], mac_mid, mid);
      }
    }
  }

  void calc_z(FieldT &mac_z, FieldT &z, const vector<FieldT> &u,
              const vector<FieldT> &v, const vector<FieldT> &mac_C,
              const vector<FieldT> &C, size_t n, size_t l) {
    assert(u.size() == n);
    assert(v.size() == l);

    // z = u.T * C * v
    for (size_t i = 0; i < l; i++) {
      // t = u.T * C
      FieldT t, mac_t;
      FieldT mid, mac_mid;
      for (size_t j = 0; j < n; j++) {
        size_t k = j * l + i;
        auth_pub_mul(mac_mid, mid, mac_C[k], C[k], u[j]);
        auth_mac_add(mac_t, t, mac_t, t, mac_mid, mid);
      }
      // z += t * v
      auth_pub_mul(mac_t, t, mac_t, t, v[i]);
      auth_mac_add(mac_z, z, mac_z, z, mac_t, t);
    }
  }

  template <typename Int>
  void zkp_matrix_mul(const QuantizedValueProver<FieldT, Int> &in1,
                      const QuantizedValueProver<FieldT, Int> &in2,
                      const QuantizedValueProver<FieldT, Int> *in3,
                      const QuantizedValueProver<FieldT, Int> &out,
                      const NormFPProver<FieldT> &fp) {
    TIME_STATS_BEG
    assert(in1.shape[0] == out.shape[0]);
    assert(in1.shape[1] == in2.shape[0]);
    assert(in2.shape[1] == out.shape[1]);
    if (in3) {
      assert(in2.shape[1] == in3->shape[0]);
      assert(in3->shape.size() == 1);
    }

    size_t n = in1.shape[0], l = in2.shape[1], m = in1.shape[1];
    size_t sz = n * l;
    vector<FieldT> mid(sz), mac_mid;

    const int min_size_multi_thread = 576;
    const int max_thread_num = 16;
    int grain_size;
    int thread_num;

    if (sz < min_size_multi_thread) {
      thread_num = 1;
      grain_size = sz;
    } else if (sz < min_size_multi_thread * max_thread_num) {
      thread_num = CEIL(sz, min_size_multi_thread);
      grain_size = min_size_multi_thread;
    } else {
      thread_num = max_thread_num;
      grain_size = CEIL(sz, max_thread_num);
    }
    vector<thread> threads(thread_num);
    for (int trd_idx = 0; trd_idx < thread_num; trd_idx++) {
      threads[trd_idx] = thread([=, &mid, &in1, &in2, &in3, &out]() {
        size_t start = min((size_t)trd_idx * grain_size, sz);
        size_t end = min((size_t)(trd_idx + 1) * grain_size, sz);
        for (size_t idx = start; idx < end; idx++) {
          // idx = i * l + j
          size_t i = idx / l;
          size_t j = idx % l;
          long imid = 0;
          for (size_t k = 0; k < m; k++) {
            imid += in1.ivalue[i * m + k] * in2.ivalue[k * l + j];
          }
          mid[idx] = FieldT(imid);
          if (in3) {
            mid[idx] += in3->value[j];
          }
        }
      });
    }
    for (auto &t : threads) t.join();

    authenticated_val_input(mac_mid, mid, sz);

    // prove mid = in1 @ w + b
    vector<FieldT> u(n);
    vector<FieldT> v(l);
    run_time += omp_get_wtime() - s_time;
    io->recv_data(&u[0], n * sizeof(FieldT));
    io->recv_data(&v[0], l * sizeof(FieldT));
    recv_size += (n + l) * sizeof(FieldT);
    s_time = omp_get_wtime();

    size_t len = in3 ? m + 1 : m;
    vector<FieldT> mac_x(len), x(len), mac_y(len), y(len);
    FieldT mac_z, z;
    // x = u.T @ in1
    calc_x(mac_x, x, u, in1);
    // y = w @ v
    calc_y(mac_y, y, v, in2, in3);
    // z = u.T @ mid @ v
    calc_z(mac_z, z, u, v, mac_mid, mid, n, l);

    // x @ y - z == 0
    vector<FieldT> co(m, FieldT(1)), left, left_mac, right, right_mac;
    if (in3) co.push_back(x[m]);
    co.push_back(FieldT(-1));
    co.push_back(FieldT(0));
    left = x;
    left_mac = mac_x;
    if (in3) {  // pop x[m]
      left.pop_back();
      left_mac.pop_back();
    }
    right = y;
    right_mac = mac_y;
    right.push_back(z);
    right_mac.push_back(mac_z);
    zkp_poly_deg2(left_mac, left, right_mac, right, co);

    zkp_round(fp, QuantizedValueProver<FieldT, Int>(mid, mac_mid, out.shape),
              out);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_linear(const LinearPrvModel<FieldT, Int> &model,
                  const QuantizedValueProver<FieldT, Int> &input,
                  const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    zkp_matrix_mul(input, model.weight, &model.bias, output, model.fp);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_poly(const NormFPProver<FieldT> &fp1,
                const NormFPProver<FieldT> &fp2,
                const NormFPProver<FieldT> &fp3,
                const QuantizedValueProver<FieldT, Int> &input,
                const QuantizedValueProver<FieldT, Int> &output1,
                const QuantizedValueProver<FieldT, Int> &output2,
                const QuantizedValueProver<FieldT, Int> &output3,
                const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    // xx = x^2
    QuantizedValueProver<FieldT, Int> xx = input * input;
    xx.auth(this);
    zkp_mul(input, input, xx);
    // y1 = round(2^e1/c1 * x^2)
    zkp_round(fp1, xx, output1);
    // y2 = round(2^e2/c2 * x)
    zkp_round(fp2, input, output2);
    // y3 = round(2^e3/c3)
    zkp_round(fp3, output3);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_exp(const SftMaxPrvModel<FieldT> &model,
               const QuantizedValueProver<FieldT, Int> &input,
               const QuantizedValueProver<FieldT, Int> &z,
               const QuantizedValueProver<FieldT, Int> &p1,
               const QuantizedValueProver<FieldT, Int> &p2,
               const QuantizedValueProver<FieldT, Int> &p,
               const QuantizedValueProver<FieldT, Int> &l,
               const QuantizedValueProver<FieldT, Int> &poly_out1,
               const QuantizedValueProver<FieldT, Int> &poly_out2,
               const QuantizedValueProver<FieldT, Int> &poly_out3,
               const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    // z = floor(input * fp1)
    zkp_floor(model.fp1, input * FieldT(-1), z);
    // p = round(sx/sp * input + ln2/sp * z)
    zkp_add(model.fp2, model.fp3, input, z, p1, p2, p);
    // l = round(1/st * poly(p))
    zkp_poly(model.fp4, model.fp5, model.fp6, p, poly_out1, poly_out2,
             poly_out3, l);
    // t = l >> z
    zkp_right_shift(l, z, output);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_softmax_requant(const NormFPProver<FieldT> &fp,
                           const QuantizedValueProver<FieldT, Int> &input,
                           const QuantizedValueProver<FieldT, Int> &output1,
                           const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    // output1 = round(2^e/c * input)
    zkp_round(fp, input, output1);
    // output = round(output1 / sum(output1))
    size_t d1 = output.shape[0], d2 = output.shape[1];
    vector<FieldT> sum(d1), mac_sum(d1);
    for (size_t i = 0; i < d1; i++) {
      for (size_t j = 0; j < d2; j++) {
        size_t k = i * d2 + j;
        sum[i] += input.value[k];
        mac_sum[i] += input.mac_value[k];
      }
    }
    QuantizedValueProver<FieldT, Int> r(sum, mac_sum, {d1, 1});
    zkp_round(output1, r, output);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_qk(const AttnPrvModel<FieldT> &model,
              const QuantizedValueProver<FieldT, Int> &q,
              const QuantizedValueProver<FieldT, Int> &k,
              const QuantizedValueProver<FieldT, Int> &qk) {
    TIME_STATS_BEG
    zkp_matrix_mul(q, k, (QuantizedValueProver<FieldT, Int> *)nullptr, qk,
                   model.fp1);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_divqk(const AttnPrvModel<FieldT> &model,
                 const QuantizedValueProver<FieldT, Int> &input,
                 const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    zkp_round(model.fp2, input, output);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_softmax(const SftMaxPrvModel<FieldT> &model,
                   const GPT2PrvData<FieldT, Int> &data, int i) {
    TIME_STATS_BEG
    zkp_max(data.divqk[i], data.x_max[i]);
    zkp_exp(model, data.divqk[i] - data.x_max[i], data.z[i], data.p1[i],
            data.p2[i], data.p[i], data.l[i], data.exp_poly_out1[i],
            data.exp_poly_out2[i], data.exp_poly_out3[i], data.exp_out[i]);
    zkp_softmax_requant(model.fp7, data.exp_out[i], data.softmax_out1[i],
                        data.softmax_out[i]);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_softmax(const SftMaxPrvModel<FieldT> &model,
                   const QuantizedValueProver<FieldT, Int> &divqk,
                   const QuantizedValueProver<FieldT, Int> &x_max,
                   const QuantizedValueProver<FieldT, Int> &z,
                   const QuantizedValueProver<FieldT, Int> &p,
                   const QuantizedValueProver<FieldT, Int> &p1,
                   const QuantizedValueProver<FieldT, Int> &p2,
                   const QuantizedValueProver<FieldT, Int> &l,
                   const QuantizedValueProver<FieldT, Int> &exp_out,
                   const QuantizedValueProver<FieldT, Int> &exp_poly_out1,
                   const QuantizedValueProver<FieldT, Int> &exp_poly_out2,
                   const QuantizedValueProver<FieldT, Int> &exp_poly_out3,
                   const QuantizedValueProver<FieldT, Int> &softmax_out,
                   const QuantizedValueProver<FieldT, Int> &softmax_out1) {
    TIME_STATS_BEG
    zkp_max(divqk, x_max);
    zkp_exp(model, divqk - x_max, z, p1, p2, p, l, exp_poly_out1, exp_poly_out2,
            exp_poly_out3, exp_out);
    zkp_softmax_requant(model.fp7, exp_out, softmax_out1, softmax_out);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_qkv(const AttnPrvModel<FieldT> &model,
               const QuantizedValueProver<FieldT, Int> &softmax_out,
               const QuantizedValueProver<FieldT, Int> &v,
               const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    zkp_matrix_mul(softmax_out, v, (QuantizedValueProver<FieldT, Int> *)nullptr,
                   output, model.fp3);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_attn(const AttnPrvModel<FieldT> &model,
                const GPT2PrvData<FieldT, Int> &data, int i) {
    TIME_STATS_BEG
    zkp_qk(model, data.q[i], data.k[i], data.qk[i]);
    zkp_divqk(model, data.qk[i], data.divqk[i]);
    zkp_softmax(model.softmax, data, i);
    zkp_qkv(model, data.softmax_out[i], data.v[i], data.qkv[i]);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_attn_without_softmax(const AttnPrvModel<FieldT> &model,
                                const GPT2PrvData<FieldT, Int> &data, int i) {
    TIME_STATS_BEG
    zkp_qk(model, data.q[i], data.k[i], data.qk[i]);
    zkp_divqk(model, data.qk[i], data.divqk[i]);
    zkp_qkv(model, data.softmax_out[i], data.v[i], data.qkv[i]);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_mha(const MHAPrvModel<FieldT, Int> &model,
               const GPT2PrvData<FieldT, Int> &data, int i) {
    TIME_STATS_BEG
    zkp_linear(model.linear[0], data.layer_norm_out[2 * i],
               data.linear_out[4 * i]);
    zkp_linear(model.linear[1], data.mha_out[i], data.linear_out[4 * i + 1]);
    for (int j = 0; j < N_HEAD; j++) {
      zkp_attn(model.attn[j], data, i * N_HEAD + j);
    }
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_mha_without_softmax(const MHAPrvModel<FieldT, Int> &model,
                               const GPT2PrvData<FieldT, Int> &data, int i) {
    TIME_STATS_BEG
    zkp_linear(model.linear[0], data.layer_norm_out[2 * i],
               data.linear_out[4 * i]);
    zkp_linear(model.linear[1], data.mha_out[i], data.linear_out[4 * i + 1]);
    for (int j = 0; j < N_HEAD; j++) {
      zkp_attn_without_softmax(model.attn[j], data, i * N_HEAD + j);
    }
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_gelu_zb(const GeluPrvModel<FieldT> &model,
                   const QuantizedValueProver<FieldT, Int> &zb) {
    TIME_STATS_BEG
    zkp_floor(model.fp1, zb);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_gelu_erf_out(const QuantizedValueProver<FieldT, Int> &l,
                        const QuantizedValueProver<FieldT, Int> &sign,
                        const QuantizedValueProver<FieldT, Int> &t) {
    TIME_STATS_BEG
    zkp_mul(l, sign, t);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_gelu_out(const GeluPrvModel<FieldT> &model,
                    const QuantizedValueProver<FieldT, Int> &input,
                    const QuantizedValueProver<FieldT, Int> &t,
                    const QuantizedValueProver<FieldT, Int> &output1,
                    const QuantizedValueProver<FieldT, Int> &output2,
                    const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    // y = round((sx/(2*sy)) * x * (1 + st * t))
    //   = round((sx/(2*sy)) * x + (sx*st/(2*sy) * x * t))

    // r = x * t
    QuantizedValueProver<FieldT, Int> r = input * t;
    r.auth(this);
    zkp_mul(input, t, r);
    zkp_add(model.fp5, model.fp6, input, r, output1, output2, output);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_gelu(const GeluPrvModel<FieldT> &model,
                const QuantizedValueProver<FieldT, Int> &input,
                const QuantizedValueProver<FieldT, Int> &sign,
                const QuantizedValueProver<FieldT, Int> &q_abs,
                const QuantizedValueProver<FieldT, Int> &zb,
                const QuantizedValueProver<FieldT, Int> &q_min,
                const QuantizedValueProver<FieldT, Int> &poly_out1,
                const QuantizedValueProver<FieldT, Int> &poly_out2,
                const QuantizedValueProver<FieldT, Int> &poly_out3,
                const QuantizedValueProver<FieldT, Int> &l,
                const QuantizedValueProver<FieldT, Int> &t,
                const QuantizedValueProver<FieldT, Int> &output1,
                const QuantizedValueProver<FieldT, Int> &output2,
                const QuantizedValueProver<FieldT, Int> &output) {
    TIME_STATS_BEG
    zkp_sign(input, sign);
    zkp_abs(input, sign, q_abs);
    zkp_gelu_zb(model, zb);
    zkp_minimum(zb, q_abs, q_min);
    // l = round(1/st * poly(sx/sqrt(2) * mn))
    zkp_poly(model.fp2, model.fp3, model.fp4, q_min, poly_out1, poly_out2,
             poly_out3, l);
    zkp_gelu_erf_out(l, sign, t);
    zkp_gelu_out(model, input, t, output1, output2, output);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_ffn(const FFNPrvModel<FieldT, Int> &model,
               const GPT2PrvData<FieldT, Int> &data, int i) {
    TIME_STATS_BEG
    zkp_linear(model.linear[0], data.layer_norm_out[2 * i + 1],
               data.linear_out[4 * i + 2]);
    zkp_gelu(model.gelu, data.linear_out[4 * i + 2], data.sign[i],
             data.q_abs[i], data.zb[i], data.q_min[i], data.gelu_poly_out1[i],
             data.gelu_poly_out2[i], data.gelu_poly_out3[i], data.erf_l[i],
             data.erf_out[i], data.gelu_out1[i], data.gelu_out2[i],
             data.gelu_out[i]);
    zkp_linear(model.linear[1], data.gelu_out[i], data.linear_out[4 * i + 3]);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_ffn_without_gelu(const FFNPrvModel<FieldT, Int> &model,
                            const GPT2PrvData<FieldT, Int> &data, int i) {
    TIME_STATS_BEG
    zkp_linear(model.linear[0], data.layer_norm_out[2 * i + 1],
               data.linear_out[4 * i + 2]);
    zkp_linear(model.linear[1], data.gelu_out[i], data.linear_out[4 * i + 3]);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_trans(const TransPrvModel<FieldT, Int> *models,
                 const GPT2PrvData<FieldT, Int> &data) {
    TIME_STATS_BEG
    const QuantizedValueProver<FieldT, Int> *in = &data.embd_out;
    for (int i = 0; i < TRANS_USED; i++) {
      printf("%d\n", i);
      zkp_layer_norm(models[i].layer_norm[0], *in, data.mean[2 * i],
                     data.var[2 * i], data.std[2 * i], data.std_max[2 * i],
                     data.sub[2 * i], data.norm[2 * i],
                     data.layer_norm_out[2 * i]);
      zkp_mha(models[i].mha, data, i);
      zkp_res(models[i].res[0], *in, data.linear_out[4 * i + 1],
              data.res_out1[2 * i], data.res_out2[2 * i], data.res_out[2 * i]);
      zkp_layer_norm(models[i].layer_norm[1], data.res_out[2 * i],
                     data.mean[2 * i + 1], data.var[2 * i + 1],
                     data.std[2 * i + 1], data.std_max[2 * i + 1],
                     data.sub[2 * i + 1], data.norm[2 * i + 1],
                     data.layer_norm_out[2 * i + 1]);
      zkp_ffn(models[i].ffn, data, i);
      zkp_res(models[i].res[1], data.res_out[2 * i], data.linear_out[4 * i + 3],
              data.res_out1[2 * i + 1], data.res_out2[2 * i + 1],
              data.res_out[2 * i + 1]);
      in = &data.res_out[2 * i + 1];
    }
    TIME_STATS_END
  }

#undef SIDE_DECL
#undef SIDE_PUSH_VAR
#undef SIDE_PUSH
#undef ZKP_POLY2
};
#endif
