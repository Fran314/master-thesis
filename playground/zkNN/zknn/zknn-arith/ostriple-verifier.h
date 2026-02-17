#ifndef FP_OS_TRIPLE_VER_H__
#define FP_OS_TRIPLE_VER_H__

#include <omp.h>

#include <cstdio>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "zknn/utility.h"
#include "zknn/vole/lvole-verifier.h"
#include "zknn/zknn-arith/cnn-verifier.h"
#include "zknn/zknn-arith/ostriple-common.h"
#include "zknn/zknn-arith/quantized-value.h"
#include "zknn/zknn-arith/transformer-verifier.h"

using namespace emp;
using namespace std;

extern size_t send_size;
extern size_t recv_size;

template <typename IO, typename FieldT>
class FpOSTripleVer {
 private:
  IO *io;
  IO **ios;
  size_t threads;
  VoleTripleVer<IO, FieldT> *vole = nullptr;
  FieldT delta;
  Hash hash;
  vector<FieldT> vB;
  const static unsigned long Repeat = 42;
  const static unsigned long e = 48;
  const FieldT L = FieldT(1 << 20);

  vector<vector<FieldT>> vzeta;
  vector<size_t> vrange_num;
  vector<FieldT> vrange;

  vector<FieldT> key, B;

 public:
  FpOSTripleVer(IO **ios, size_t threads) {
    io = ios[0];
    this->ios = ios;
    this->threads = threads;
    vole = new VoleTripleVer<IO, FieldT>(ios, threads);
    vole->setup();
    this->delta = vole->delta;
  }

  ~FpOSTripleVer() { delete vole; }
  /* ---------------------inputs----------------------*/

  void authenticated_val_input(FieldT &key) {
    vole->extend(key);
    FieldT u;
    run_time += omp_get_wtime() - s_time;
    io->recv_data(&u, sizeof(FieldT));
    s_time = omp_get_wtime();
    recv_size += sizeof(FieldT);
    key = key - u * delta;
  }

  void authenticated_val_input(vector<FieldT> &key, size_t len) {
    vector<FieldT> u;

    TIME_STATS_BEG_NAME(extend)
    vole->extend(key, len);
    TIME_STATS_END_NAME(extend)
    
    u.resize(len);
    run_time += omp_get_wtime() - s_time;
    io->recv_data(&u[0], len * sizeof(FieldT));
    s_time = omp_get_wtime();
    recv_size += len * sizeof(FieldT);

    // #pragma omp parallel for
    for (size_t i = 0; i < len; ++i) {
      key[i] = key[i] - u[i] * delta;
    }
  }

  void inline auth_mac_add(FieldT &key, const FieldT &key1,
                           const FieldT &key2) {
    key = key1 + key2;
  }

  void inline auth_mac_sub(FieldT &key, const FieldT &key1,
                           const FieldT &key2) {
    key = key1 - key2;
  }

  void inline auth_pub_add(FieldT &key, const FieldT &key1, const FieldT &c) {
    key = key1 + c * delta;
  }

  void inline auth_pub_mul(FieldT &key, const FieldT &key1, const FieldT &c) {
    key = c * key1;
  }

  void inline auth_mac_linear(FieldT &key, const vector<FieldT> &vkey,
                              const vector<FieldT> &vc) {
    key = FieldT::zero();
    for (size_t i = 0; i < vkey.size(); i++) {
      key += vkey[i] * vc[i];
    }
  }

  void check_zero(const vector<FieldT> &key) {
    size_t len = key.size();
    hash.put(&key[0], len * sizeof(FieldT));
  }

  void check_zero(const FieldT &key) { hash.put(&key, sizeof(FieldT)); }

  void zkp_poly_deg2(const vector<FieldT> &left_key,
                     const vector<FieldT> &right_key,
                     const vector<FieldT> &coeff) {
    size_t left_num = left_key.size();
    size_t right_num = right_key.size();

    //    FieldT B_p = FieldT::zero();
    FieldT B = FieldT::zero();

    for (size_t i = 0; i < left_num; ++i) {
      B = B + coeff[i] * left_key[i] * right_key[i];
    }

    for (size_t i = left_num; i < right_num; i++) {
      B = B + coeff[i] * right_key[i] * delta;
    }
    B = B + coeff[right_num] * delta * delta;
    vB.push_back(B);
  }

  void zkp_poly_deg2(const vector<FieldT> &left_key,
                     const vector<FieldT> &right_key,
                     const vector<FieldT> &coeff, size_t B_idx) {
    size_t left_num = left_key.size();
    size_t right_num = right_key.size();

    //    FieldT B_p = FieldT::zero();
    FieldT B = FieldT::zero();

    for (size_t i = 0; i < left_num; ++i) {
      B = B + coeff[i] * left_key[i] * right_key[i];
    }

    for (size_t i = left_num; i < right_num; i++) {
      B = B + coeff[i] * right_key[i] * delta;
    }
    B = B + coeff[right_num] * delta * delta;

    vB[B_idx] = B;
    // vB.push_back(B);
  }

  void zkp_poly_deg2_p(const vector<FieldT> &left_key,
                       const vector<FieldT> &right_key,
                       const vector<FieldT> &coeff) {
    size_t left_num = left_key.size();
    size_t right_num = right_key.size();

    FieldT B_p = FieldT::zero();
    FieldT B = FieldT::zero();

#pragma omp parallel private(B_p) shared(B)
    {
#pragma omp for
      for (size_t i = 0; i < left_num; ++i) {
        B_p = B_p + coeff[i] * left_key[i] * right_key[i];
      }
#pragma omp critical
      B += B_p;
    }

    for (size_t i = left_num; i < right_num; i++) {
      B = B + coeff[i] * right_key[i] * delta;
    }
    B = B + coeff[right_num] * delta * delta;
    vB.push_back(B);
  }

  void zkp_range(const vector<FieldT> &key, const FieldT &B, size_t num) {
    this->key.insert(this->key.end(), key.begin(),key.end());
    while (num--) {
      this->B.push_back(B);
    }
  }


  void zkp_range_batch(const FieldT& B_bound) {
    TIME_STATS_BEG

    size_t num = key.size();

    vector<FieldT> key_y1, key_y2, key_y3;

    TIME_STATS_BEG_NAME(auth)
    authenticated_val_input(key_y1, num);
    authenticated_val_input(key_y2, num);
    authenticated_val_input(key_y3, num);
    TIME_STATS_END_NAME(auth)

    // shortness test begin
    TIME_STATS_BEG_NAME(shortness_test)

    size_t random_num =
        (num * Repeat * 4 + 8 * sizeof(unsigned) - 1) / (8 * sizeof(unsigned));
    vector<unsigned> gamma(random_num);
    randomize(gamma);
    io->send_data(&gamma[0], random_num * sizeof(unsigned));

    vector<FieldT> key_zeta, zeta;
    zeta.resize(Repeat);
    key_zeta.resize(Repeat);

#pragma omp parallel for
    for (size_t i = 0; i < Repeat; i++) {
      for (size_t j = 0; j < num; j++)
        if (test_bit(gamma, (j * 4 + 0) * i) == true)
          auth_mac_add(key_zeta[i], key_zeta[i], key[j]);
      for (size_t j = 0; j < num; j++)
        if (test_bit(gamma, (j * 4 + 1) * i) == true)
          auth_mac_add(key_zeta[i], key_zeta[i], key_y1[j]);
      for (size_t j = 0; j < num; j++)
        if (test_bit(gamma, (j * 4 + 2) * i) == true)
          auth_mac_add(key_zeta[i], key_zeta[i], key_y2[j]);
      for (size_t j = 0; j < num; j++)
        if (test_bit(gamma, (j * 4 + 3) * i) == true)
          auth_mac_add(key_zeta[i], key_zeta[i], key_y3[j]);
    }

    // shortness test end
    TIME_STATS_END_NAME(shortness_test)

    // shortness test begin
    TIME_STATS_BEG_NAME(digit_decomposition)

    size_t nbits = (B_bound * 4 * num).as_bigint().num_bits();
    vector<FieldT> key_zeta_bit(Repeat * nbits);
    authenticated_val_input(key_zeta_bit, Repeat * nbits);

    for (int i = 0; i < Repeat; i++) {
      FieldT key_rhs;
      for (int j = 0; j < nbits; j++) {
        int k = i * nbits + j;
        auth_pub_mul(key_zeta_bit[k], key_zeta_bit[k], FieldT(2) ^ j);
        auth_mac_add(key_rhs, key_rhs, key_zeta_bit[k]);
      }
      auth_mac_sub(key_zeta[i], key_zeta[i], key_rhs);
    }
    check_zero(key_zeta);

    TIME_STATS_END_NAME(digit_decomposition)


    TIME_STATS_BEG_NAME(poly)
    // vector<FieldT> co;
    // co.push_back(FieldT(4));
    // co.push_back(FieldT(1));
    // co.push_back(FieldT(1));
    // co.push_back(FieldT(1));
    // co.push_back(FieldT(-4) * B);
    // co.push_back(FieldT(-1));
    size_t vB_oSize = vB.size();
    vB.resize(vB.size() + num);
#pragma omp parallel for
    for (size_t i = 0; i < num; i++) {
	  vector<FieldT> co;
	  co.resize(6);
	  co[0] = FieldT(4);
	  co[1] = FieldT(1);
	  co[2] = FieldT(1);
	  co[3] = FieldT(1);
	  co[4] = FieldT(-4) * B[i];
	  co[5] = FieldT(-1);
      vector<FieldT> left_key;
      vector<FieldT> right_key;
      left_key.resize(4);
      right_key.resize(5);
      left_key[0] = key[i];
      left_key[1] = key_y1[i];
      left_key[2] = key_y2[i];
      left_key[3] = key_y3[i];
      right_key[0] = key[i];
      right_key[1] = key_y1[i];
      right_key[2] = key_y2[i];
      right_key[3] = key_y3[i];
      right_key[4] = key[i];
      zkp_poly_deg2(left_key, right_key, co, vB_oSize + i);
    }
    TIME_STATS_END_NAME(poly)

    this->key.clear();
    this->B.clear();

    TIME_STATS_END
  }


  void lookup_get_poly_value_g(vector<FieldT> &left_key,
                               vector<FieldT> &right_key,
                               const vector<vector<FieldT>> &key_x,
                               const vector<FieldT> &key_g_mid,
                               const vector<FieldT> &alpha_dim,
                               const FieldT &gamma, size_t i) {
    left_key.resize(1);
    right_key.resize(2);
    if (i != 0)
      left_key[0] = key_g_mid[i - 1];
    else {
      auth_mac_linear(left_key[0], key_x[i], alpha_dim);
      auth_pub_add(left_key[0], left_key[0], gamma);
    }
    auth_mac_linear(right_key[0], key_x[i + 1], alpha_dim);
    auth_pub_add(right_key[0], right_key[0], gamma);
    right_key[1] = key_g_mid[i];
  }

  void lookup_get_poly_value_h(vector<FieldT> &left_key,
                               vector<FieldT> &right_key,
                               const vector<vector<FieldT>> &key_s,
                               const vector<FieldT> &key_h_mid,
                               const vector<FieldT> &alpha_dim,
                               const FieldT &gamma_beta, const FieldT &beta,
                               size_t i) {
    size_t dim = alpha_dim.size();
    left_key.resize(1);
    right_key.resize(2);

    if (i != 0)
      left_key[0] = key_h_mid[i - 1];
    else {
      vector<FieldT> key_beta_s(dim);
      for (size_t j = 0; j < dim; j++) {
        auth_pub_mul(key_beta_s[j], key_s[i + 1][j], beta);
        auth_mac_add(key_beta_s[j], key_beta_s[j], key_s[i][j]);
      }
      auth_mac_linear(left_key[0], key_beta_s, alpha_dim);
      auth_pub_add(left_key[0], left_key[0], gamma_beta);
    }

    vector<FieldT> key_beta_s(dim);
    for (size_t j = 0; j < dim; j++) {
      auth_pub_mul(key_beta_s[j], key_s[i + 2][j], beta);
      auth_mac_add(key_beta_s[j], key_beta_s[j], key_s[i + 1][j]);
    }
    auth_mac_linear(right_key[0], key_beta_s, alpha_dim);
    auth_pub_add(right_key[0], right_key[0], gamma_beta);
    right_key[1] = key_h_mid[i];
  }

  void zkp_lookup(const vector<vector<FieldT>> &key_x,
                  const vector<vector<FieldT>> &T) {
    vector<vector<FieldT>> key_s;

    size_t t_size = T.size();
    size_t x_size = key_x.size();
    size_t s_size = key_x.size() + T.size();
    size_t dim = key_x[0].size();
    key_s.resize(s_size);
    for (size_t i = 0; i < s_size; i++) authenticated_val_input(key_s[i], dim);

    FieldT r[3] = {FieldT::random_element(), FieldT::random_element(),
                   FieldT::random_element()};
    io->send_data(&r[0], 3 * sizeof(FieldT));

    vector<FieldT> key_g_mid(x_size - 1);
    vector<FieldT> key_h_mid(s_size - 2);
    authenticated_val_input(key_g_mid, x_size - 1);
    authenticated_val_input(key_h_mid, s_size - 2);

    vector<FieldT> alpha_dim(dim);
    alpha_dim[0] = FieldT::one();
    for (size_t i = 1; i < dim; i++) alpha_dim[i] = alpha_dim[i - 1] * r[0];
    FieldT gamma_beta = r[2] * (FieldT::one() + r[1]);

    vector<FieldT> co_gh(3);
    co_gh[0] = FieldT::one();
    co_gh[1] = -FieldT::one();
    co_gh[2] = FieldT::zero();
    for (size_t i = 0; i < x_size - 1; i++) {
      vector<FieldT> left_key;
      vector<FieldT> right_key;
      lookup_get_poly_value_g(left_key, right_key, key_x, key_g_mid, alpha_dim,
                              r[2], i);
      zkp_poly_deg2(left_key, right_key, co_gh);
    }
    for (size_t i = 0; i < s_size - 2; i++) {
      vector<FieldT> left_key;
      vector<FieldT> right_key;
      lookup_get_poly_value_h(left_key, right_key, key_s, key_h_mid, alpha_dim,
                              gamma_beta, r[1], i);
      zkp_poly_deg2(left_key, right_key, co_gh);
    }
  }

  bool batch_check(bool zero_check, bool poly_check, bool range_check) {
    TIME_STATS_BEG
    // zero check.
    bool ret = true;
    if (zero_check || range_check) {
      char dig[emp::Hash::DIGEST_SIZE];
      hash.digest(dig);
      char dig_recv[emp::Hash::DIGEST_SIZE];

      run_time += omp_get_wtime() - s_time;
      io->recv_data(dig_recv, emp::Hash::DIGEST_SIZE);
      s_time = omp_get_wtime();
      recv_size += emp::Hash::DIGEST_SIZE;

      if (!cmpBlock((block *)dig, (block *)dig_recv,
                    emp::Hash::DIGEST_SIZE / 16)) {
        printf("zero check error\n");
        ret = false;
      }
    }

    // poly check.
    if ((poly_check || range_check) && (vB.size() != 0)) {
      FieldT seed = FieldT::random_element();
      io->send_data(&seed, sizeof(FieldT));
      vector<FieldT> chi;
      uni_hash_coeff_gen(chi, seed, vB.size());

      FieldT W = FieldT::zero();
      for (size_t i = 0; i < vB.size(); i++) {
        W = W + vB[i] * chi[i];
      }
      FieldT Bstar;
      vole->extend(Bstar);
      W = W + Bstar;

      FieldT U[2];

      run_time += omp_get_wtime() - s_time;
      io->recv_data(U, 2 * sizeof(FieldT));
      recv_size += 2 * sizeof(FieldT);
      s_time = omp_get_wtime();

      if (W != (U[0] + U[1] * delta)) {
        printf("poly check error\n");
        ret = false;
      }
      vB.clear();
    }

    // range check.
    if (range_check) {
      for (size_t i = 0; i < vzeta.size(); i++) {
        FieldT RL = (FieldT(4) * FieldT(vrange_num[i], true) * vrange[i] +
                     FieldT::one()) *
                    L;
        for (size_t j = 0; j < Repeat; j++) {
          if ((vzeta[i][j] > RL) || (vzeta[i][j] < FieldT::zero())) {
            printf("range check error\n");
            ret = false;
          }
        }
      }
      vzeta.clear();
      vrange_num.clear();
      vrange.clear();
    }
    TIME_STATS_END
    return ret;
  }

  /* ======= primitive operations ======= */

#define SIDE_DECL() vector<FieldT> left_key, right_key
#define SIDE_PUSH(side, ident) side##_key.push_back(key_##ident)
#define SIDE_PUSH_VAR(side, var, ident) side##_key.push_back(var.key_##ident)
#define ZKP_POLY2() zkp_poly_deg2(left_key, right_key, co)

  template <typename Int>
  void zkp_max(const QuantizedValueVerifier<FieldT> &input,
               const QuantizedValueVerifier<FieldT> &output,
               size_t bits = 2 * sizeof(Int)) {
    // 0 <= y - x <= B
    QuantizedValueVerifier<FieldT> r = output - input;
    zkp_range(r.key_value, B_max(bits), r.size());

    // prod(y - x) == 0
    // mid = prod(y - x)
    size_t sz = r.size();
    FieldT key_mid = r.key_value[0];
    vector<FieldT> co = co_max<FieldT>();
    for (size_t k = 1; k < sz; k++) {
      SIDE_DECL();
      SIDE_PUSH(left, mid);
      authenticated_val_input(key_mid);
      SIDE_PUSH_VAR(right, r, value[k]);
      SIDE_PUSH(right, mid);
      ZKP_POLY2();
    }
    // mid == 0
    check_zero(key_mid);
  }

  template <typename Int>
  void zkp_maximum(const QuantizedValueVerifier<FieldT> &mn,
                   const QuantizedValueVerifier<FieldT> &input,
                   const QuantizedValueVerifier<FieldT> &output) {
    // y = max(x, mn)
    size_t sz = output.size();
    QuantizedValueVerifier<FieldT> r1 = output - mn, r2 = output - input;
    // (y - mn)(y - x) == 0
    vector<FieldT> co = co_minimum<FieldT>(sz);
    zkp_poly_deg2(r1.key_value, r2.key_value, co);
    // 0 <= y - mn < B
    // 0 <= y - x < B
    r1.key_value.insert(r1.key_value.end(), r2.key_value.begin(),
                        r2.key_value.end());
    zkp_range(r1.key_value, B_maximum(2 * sizeof(Int)), 2 * sz);
  }

  template <typename Int>
  void zkp_minimum(const QuantizedValueVerifier<FieldT> &mx,
                   const QuantizedValueVerifier<FieldT> &input,
                   const QuantizedValueVerifier<FieldT> &output) {
    // y = min(x, zb)
    size_t sz = output.size();
    QuantizedValueVerifier<FieldT> r1 = mx - output, r2 = input - output;
    // (mx - x)(x - y) == 0
    vector<FieldT> co = co_minimum<FieldT>(sz);
    zkp_poly_deg2(r1.key_value, r2.key_value, co);
    // 0 <= mx - y < B
    // 0 <= x - y < B
    r1.key_value.insert(r1.key_value.end(), r2.key_value.begin(),
                        r2.key_value.end());
    zkp_range(r1.key_value, B_minimum(2 * sizeof(Int)), 2 * sz);
  }

  template <typename Int>
  void zkp_sign(const QuantizedValueVerifier<FieldT> &input,
                const QuantizedValueVerifier<FieldT> &output) {
    size_t sz = output.size();
    // 0 < y(x - y)
    // r1 = x - y
    QuantizedValueVerifier<FieldT> r1 = input - output;
    // r2 = y * r1
    QuantizedValueVerifier<FieldT> r2(sz);
    r2.auth(this);
    // verify y * r1 - r2 == 0
    vector<FieldT> co = {1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, output, value[i]);
      SIDE_PUSH_VAR(right, r1, value[i]);
      SIDE_PUSH_VAR(right, r2, value[i]);
      ZKP_POLY2();
    }
    // verify 0 <= r2 < B
    zkp_range(r2.key_value, B_sign(2 * sizeof(Int)), sz);

    // (y - 1)(y + 1)(y^2 + x^2) == 0
    // r3 = (y - 1)(y + 1) = y ^ 2 - 1
    QuantizedValueVerifier<FieldT> r3(sz);
    r3.auth(this);
    // verify y^2 - r3 - 1 == 0
    co = {1, -1, -1};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, output, value[i]);
      SIDE_PUSH_VAR(right, output, value[i]);
      SIDE_PUSH_VAR(right, r3, value[i]);
      ZKP_POLY2();
    }
    // r4 = y^2 + x^2
    QuantizedValueVerifier<FieldT> r4(sz);
    r4.auth(this);
    // verify y^2 + x^2 - r4 == 0
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
    QuantizedValueVerifier<FieldT> r5(sz);
    r5.auth(this);
    // verify r3 * r4 - r5 == 0
    co = {1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, r3, value[i]);
      SIDE_PUSH_VAR(right, r4, value[i]);
      SIDE_PUSH_VAR(right, r5, value[i]);
      ZKP_POLY2();
    }
    // r5 == 0
    check_zero(r5.key_value);
  }

  template <typename Int>
  void zkp_abs(const QuantizedValueVerifier<FieldT> &input,
               const QuantizedValueVerifier<FieldT> &sign,
               const QuantizedValueVerifier<FieldT> &output) {
    // y = abs(x)
    // sign * x - y == 0
    size_t sz = output.size();
    // r = sign * x - y
    QuantizedValueVerifier<FieldT> r(sz);
    r.auth(this);
    // verify sign * x - y - r == 0
    vector<FieldT> co = {1, -1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, sign, value[i]);
      SIDE_PUSH_VAR(right, input, value[i]);
      SIDE_PUSH_VAR(right, output, value[i]);
      SIDE_PUSH_VAR(right, r, value[i]);
      ZKP_POLY2();
    }
    // verify r == 0
    check_zero(r.key_value);
  }

  template <typename Int>
  void zkp_right_shift(const QuantizedValueVerifier<FieldT> &input,
                       const QuantizedValueVerifier<FieldT> &bit,
                       const QuantizedValueVerifier<FieldT> &output) {
    // y = x >> b
    size_t sz = output.size();
    // r1 = 2^b
    QuantizedValueVerifier<FieldT> r1(sz);
    r1.auth(this);
    // TODO: prove r1 == 2^b

    // 0 <= x - y * r1 <= r1 - 1
    // r2 = y * r1
    QuantizedValueVerifier<FieldT> r2(sz);
    r2.auth(this);
    // verify y * r1 - r2 == 0
    vector<FieldT> co = {1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, output, value[i]);
      SIDE_PUSH_VAR(right, r1, value[i]);
      SIDE_PUSH_VAR(right, r2, value[i]);
      ZKP_POLY2();
    }
    // verify 0 <= r2 <= r1 - 1
    zkp_range(r2.key_value, B_shift(2 * sizeof(Int)), sz);
  }

  template <typename Int>
  void zkp_round(const NormFPVerifier<FieldT> &fp,
                 const QuantizedValueVerifier<FieldT> &input,
                 const QuantizedValueVerifier<FieldT> &output) {
    // y = round(2^e/c * x)
    // 0 <= 2 * 2^e * x - 2cy + c <= 2c - 1
    size_t sz = output.size();
    // r = 2 * 2^e * x - 2cy + c
    QuantizedValueVerifier<FieldT> r(sz);
    r.auth(this);
    // verify 2 * 2^e * x - 2cy + c - r == 0
    vector<FieldT> co = {2, -2, 1, -1, 0};
    vector<FieldT> left_key;
    SIDE_PUSH_VAR(left, fp, power);
    SIDE_PUSH_VAR(left, fp, coeff);
    for (size_t i = 0; i < sz; i++) {
      vector<FieldT> right_key;
      SIDE_PUSH_VAR(right, input, value[i]);
      SIDE_PUSH_VAR(right, output, value[i]);
      SIDE_PUSH_VAR(right, fp, coeff);
      SIDE_PUSH_VAR(right, r, value[i]);
    }
    // prove 0 <= r <= 2c - 1
    zkp_range(r.key_value, B_round(), sz);
  }

  template <typename Int>
  void zkp_round(const NormFPVerifier<FieldT> &fp,
                 const QuantizedValueVerifier<FieldT> &output) {
    // y = round(2^e/c)
    // 0 <= 2 * 2^e - 2cy + c <= 2c - 1
    size_t sz = output.size();
    // r = 2 * 2^e - 2cy
    QuantizedValueVerifier<FieldT> r(sz);
    r.auth(this);
    // verify -2cy + 2 * 2^e + c - r == 0
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
    zkp_range(r.key_value, B_round(), sz);
  }

  template <typename Int>
  void zkp_round(const QuantizedValueVerifier<FieldT> &input1,
                 const QuantizedValueVerifier<FieldT> &input2,
                 const QuantizedValueVerifier<FieldT> &output) {
    // y = round(x1 / x2)
    // 0 <= 2x1 - 2x2*y + x2 <= 2x2 - 1
    size_t sz = output.size();
    size_t f2 = sz / input2.size();
    // r = 2x1 - 2x2*y + x2
    QuantizedValueVerifier<FieldT> r(sz);
    r.auth(this);
    // verify -2x2*y + 2x1 + x2 - r == 0
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
    // verify 0 <= r <= 2x2 - 1
    zkp_range(r.key_value, B_round2(2 * sizeof(Int)), sz);
  }

  template <typename Int>
  void zkp_floor(const NormFPVerifier<FieldT> &fp,
                 const QuantizedValueVerifier<FieldT> &output) {
    // y = floor(2^e/c)
    // 0 <= 2^e - cy <= c
    size_t sz = output.size();
    // r = 2^e - cy
    QuantizedValueVerifier<FieldT> r(sz);
    r.auth(this);
    // verify -cy + 2^e - r == 0
    vector<FieldT> co = {-1, 1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, fp, coeff);
      SIDE_PUSH_VAR(right, output, value[i]);
      SIDE_PUSH_VAR(right, fp, power);
      SIDE_PUSH_VAR(right, r, value[i]);
      ZKP_POLY2();
    }
    // verify 0 <= r <= c
    zkp_range(r.key_value, B_floor(), sz);
  }

  template <typename Int>
  void zkp_floor(const NormFPVerifier<FieldT> &fp,
                 const QuantizedValueVerifier<FieldT> &input,
                 const QuantizedValueVerifier<FieldT> &output) {
    // y = floor(2^e/c * x)
    // 0 <= 2^e * x - cy <= c
    size_t sz = output.size();
    // r = 2^e * x - cy
    QuantizedValueVerifier<FieldT> r(sz);
    r.auth(this);
    // verify 2^e * x - cy - r == 0
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
    // verify 0 <= r <= c
    zkp_range(r.key_value, B_floor(), sz);
  }

  template <typename Int>
  void zkp_sqrt(const QuantizedValueVerifier<FieldT> &input,
                const QuantizedValueVerifier<FieldT> &output) {
    size_t sz = output.size();
    QuantizedValueVerifier<FieldT> r(sz);
    r.auth(this);
    // verify -y * y + x - r == 0
    vector<FieldT> co = {-1, 1, -1, 0};
    for (size_t i = 0; i < sz; i++) {
      SIDE_DECL();
      SIDE_PUSH_VAR(left, output, value[i]);
      SIDE_PUSH_VAR(right, output, value[i]);
      SIDE_PUSH_VAR(right, input, value[i]);
      SIDE_PUSH_VAR(right, r, value[i]);
      ZKP_POLY2();
    }
    // verify 0 <= r <= B
    zkp_range(r.key_value, B_std(2 * sizeof(Int)), sz);

    // r = (y + 1) ^ 2 - x
    r.auth(this);
    // verify y^2 + 2*y - x - r + 1 == 0
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
    zkp_range(r.key_value, B_std(2 * sizeof(Int)), sz);
  }

  void zkp_mul(const QuantizedValueVerifier<FieldT> &input1,
               const QuantizedValueVerifier<FieldT> &input2,
               const QuantizedValueVerifier<FieldT> &output) {
    // y = x1 * x2
    // verify x1 * x2 - y1 == 0
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

  void zkp_mul(const FieldT &key_input1,
               const QuantizedValueVerifier<FieldT> &input2,
               const QuantizedValueVerifier<FieldT> &output) {
    // y = x1 * x2
    // verify x1 * x2 - y1 == 0
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

  void zkp_mul(const FieldT &key_input1, const FieldT &key_input2,
               const FieldT &key_output) {
    // y = x1 * x2
    // verify x1 * x2 - y1 == 0
    vector<FieldT> co = {1, -1, 0};
    SIDE_DECL();
    SIDE_PUSH(left, input1);
    SIDE_PUSH(right, input2);
    SIDE_PUSH(right, output);
    ZKP_POLY2();
  }

  template <typename Int>
  void zkp_relu(const QuantizedValueVerifier<FieldT> &input,
                const QuantizedValueVerifier<FieldT> &output) {
    TIME_STATS_BEG
    vector<FieldT> key_x = {FieldT(0) * delta};
    QuantizedValueVerifier<FieldT> mn(key_x, {1, 1, 1, 1});  // 0
    zkp_maximum<Int>(mn, input, output);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_add(const NormFPVerifier<FieldT> &fp1,
               const NormFPVerifier<FieldT> &fp2,
               const QuantizedValueVerifier<FieldT> &input1,
               const QuantizedValueVerifier<FieldT> &input2,
               const QuantizedValueVerifier<FieldT> &output1,
               const QuantizedValueVerifier<FieldT> &output2,
               const QuantizedValueVerifier<FieldT> &output) {
    // y1 = round(2^e1/c1 * x1)
    zkp_round<Int>(fp1, input1, output1);
    // y2 = round(2^e2/c2 * x2)
    zkp_round<Int>(fp2, input2, output2);
    // y = y1 + y2
  }

  template <typename Int>
  void zkp_skip_add(const SkipAddVerModel<FieldT> &model,
                    const QuantizedValueVerifier<FieldT> &input1,
                    const QuantizedValueVerifier<FieldT> &input2,
                    const QuantizedValueVerifier<FieldT> &output1,
                    const QuantizedValueVerifier<FieldT> &output2,
                    const QuantizedValueVerifier<FieldT> &output) {
    zkp_add<Int>(model.fp1, model.fp2, input1, input2, output1, output2,
                 output);
  }

  template <typename Int>
  void zkp_res(const ResVerModel<FieldT> &model,
               const QuantizedValueVerifier<FieldT> &input1,
               const QuantizedValueVerifier<FieldT> &input2,
               const QuantizedValueVerifier<FieldT> &output1,
               const QuantizedValueVerifier<FieldT> &output2,
               const QuantizedValueVerifier<FieldT> &output) {
    zkp_add<Int>(model.fp1, model.fp2, input1, input2, output1, output2,
                 output);
  }

  void maxpool_calc_R(vector<FieldT> &key_R, vector<size_t> &mid_res_size,
                      const QuantizedValueVerifier<FieldT> &input,
                      const QuantizedValueVerifier<FieldT> &output,
                      const size_t padding, const size_t stride,
                      const size_t window) {
    size_t input_d = input.shape[1];
    size_t input_l = input.shape[2];
    size_t input_w = input.shape[3];
    size_t output_d = output.shape[1];
    size_t output_l = output.shape[2];
    size_t output_w = output.shape[3];

    size_t output_size = output_d * output_l * output_w;
    mid_res_size.resize(output_size);

    for (size_t k = 0; k < output_d; k++) {
      for (size_t i = 0; i < output_l; i++) {
        for (size_t j = 0; j < output_w; j++) {
          long output_idx = get_3D_idx(output_l, output_w, k, i, j);
          mid_res_size[output_idx] = -1;
          for (size_t m = 0; m < window; m++) {
            for (size_t n = 0; n < window; n++) {
              long input_l_idx = (long)m - padding + stride * i;
              long input_w_idx = (long)n - padding + stride * j;
              long input_idx =
                  get_3D_idx(input_l, input_w, k, input_l_idx, input_w_idx);
              if (input_l_idx < 0 || input_l_idx >= input_l) continue;
              if (input_w_idx < 0 || input_w_idx >= input_w) continue;
              FieldT key_o_minus_i;
              auth_mac_sub(key_o_minus_i, output.key_value[output_idx],
                           input.key_value[input_idx]);
              key_R.push_back(key_o_minus_i);
              mid_res_size[output_idx] += 1;
            }
          }
        }
      }
    }
  }

  template <typename Int>
  void zkp_maxpool(const QuantizedValueVerifier<FieldT> &input,
                   const QuantizedValueVerifier<FieldT> &output,
                   const size_t padding = 1, const size_t stride = 2,
                   const size_t window = 3) {
    vector<FieldT> key_R;
    vector<size_t> mid_res_size;
    maxpool_calc_R(key_R, mid_res_size, input, output, padding, stride, window);
    zkp_range(key_R, B_maxpool(2 * sizeof(Int)), key_R.size());

    vector<vector<FieldT>> key_mid_res;
    size_t output_size = output.size();
    key_mid_res.resize(output_size);
    for (size_t i = 0; i < output_size; i++)
      authenticated_val_input(key_mid_res[i], mid_res_size[i]);

    vector<FieldT> co = {1, -1, 0};
    size_t idx = 0;
    for (size_t i = 0; i < output_size; i++) {
      FieldT key_res_x = key_R[idx];
      idx++;
      for (size_t j = 0; j < mid_res_size[i]; j++) {
        SIDE_DECL();
        SIDE_PUSH(left, res_x);
        SIDE_PUSH(right, R[idx]);
        SIDE_PUSH(right, mid_res[i][j]);
        key_res_x = key_mid_res[i][j];
        idx++;
        ZKP_POLY2();
      }
    }
  }

  void fc_calc_x(vector<FieldT> &key_x, const vector<FieldT> &u,
                 const QuantizedValueVerifier<FieldT> &A, size_t m) {
    size_t n = 1;
    assert(key_x.size() == m + 1);
    assert(u.size() == n);

    FieldT key_mid;
    // x.T = u.T * A
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        size_t k = j * m + i;
        auth_pub_mul(key_mid, A.key_value[k], u[j]);
        auth_mac_add(key_x[i], key_x[i], key_mid);
      }
    }
    // for bias, x[m] = sum(u)
    if (key_x.size() == m + 1) {
      for (size_t j = 0; j < n; j++) {
        key_x[m] += u[j];
      }
    }
  }

  void fc_calc_y(vector<FieldT> &key_y, const vector<FieldT> &v,
                 const FCVerModel<FieldT> &model) {
    size_t m = model.weight.shape[0];
    size_t l = model.weight.shape[1];
    assert(key_y.size() == m + 1);
    assert(v.size() == l);

    FieldT neg_sum_v = FieldT(0);
    for (size_t i = 0; i < v.size(); i++) neg_sum_v -= v[i];

// y = B * v
#pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < l; j++) {
        size_t k = i * l + j;
        FieldT key_mid;
        auth_pub_mul(key_mid, model.weight.key_value[k], v[j]);
        auth_mac_add(key_y[i], key_y[i], key_mid);
      }
    }
    // bias
    FieldT key_mid;
    for (size_t j = 0; j < l; j++) {
      auth_pub_mul(key_mid, model.bias.key_value[j], v[j]);
      auth_mac_add(key_y[m], key_y[m], key_mid);
    }
  }

  void fc_calc_z(FieldT &key_z, const vector<FieldT> &u,
                 const vector<FieldT> &v, const vector<FieldT> &key_C, size_t n,
                 size_t l) {
    assert(u.size() == n);
    assert(v.size() == l);

    // z = u.T * C * v
    for (size_t i = 0; i < l; i++) {
      // t = u.T * C
      FieldT key_t;
      FieldT key_mid;
      for (size_t j = 0; j < n; j++) {
        size_t k = j * l + i;
        auth_pub_mul(key_mid, key_C[k], u[j]);
        auth_mac_add(key_t, key_t, key_mid);
      }
      // z += t * v
      auth_pub_mul(key_t, key_t, v[i]);
      auth_mac_add(key_z, key_z, key_t);
    }
  }

  template <typename Int>
  void zkp_fc(const FCVerModel<FieldT> &model,
              const QuantizedValueVerifier<FieldT> &input,
              const QuantizedValueVerifier<FieldT> &output) {
    size_t n = 1, l = model.weight.shape[1], m = model.weight.shape[0];
    size_t sz = n * l;
    vector<FieldT> key_mid;

    authenticated_val_input(key_mid, sz);

    vector<FieldT> u(n);
    vector<FieldT> v(l);
    for (size_t i = 0; i < n; i++) u[i] = FieldT::random_element();
    for (size_t i = 0; i < l; i++) v[i] = FieldT::random_element();
    io->send_data(&u[0], n * sizeof(FieldT));
    io->send_data(&v[0], l * sizeof(FieldT));

    size_t len = m + 1;
    vector<FieldT> key_x(len), key_y(len);
    FieldT key_z;
    // x
    fc_calc_x(key_x, u, input, m);
    // y
    fc_calc_y(key_y, v, model);
    // z
    fc_calc_z(key_z, u, v, key_mid, n, l);

    vector<FieldT> co(m, FieldT(1)), left_key, right_key;
    co.push_back(key_x[m]);
    co.push_back(FieldT(-1));
    co.push_back(FieldT(0));
    left_key = key_x;
    left_key.pop_back();
    right_key = key_y;
    right_key.push_back(key_z);
    zkp_poly_deg2(left_key, right_key, co);

    zkp_round<Int>(model.fp,
                   QuantizedValueVerifier<FieldT>(key_mid, output.shape),
                   output);
  }
  void conv_get_poly_co_R(vector<FieldT> &co) {
    co.resize(4);
    co[0] = FieldT(-1);
    co[1] = FieldT(-1);
    co[2] = (FieldT(2) ^ (e));
    co[3] = (FieldT(2) ^ (e - 1));
  }

  void conv_get_poly_value_R(vector<FieldT> &left_key,
                             vector<FieldT> &right_key,
                             const vector<FieldT> &key_R,
                             const vector<FieldT> &key_mid, size_t idx,
                             const ConvVerModel<FieldT> &model,
                             const QuantizedValueVerifier<FieldT> &output) {
    left_key.resize(1);
    right_key.resize(3);
    left_key[0] = model.key_E;
    right_key[0] = key_mid[idx];
    right_key[1] = key_R[idx];
    right_key[2] = output.key_value[idx];
  }

  void conv_calc_x(vector<FieldT> &key_x, const vector<FieldT> &u,
                   const ConvVerModel<FieldT> &model) {
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
        FieldT key_mid;
        auth_pub_mul(key_mid, model.weight.key_value[j * m + i], u[j]);
        auth_mac_add(key_x[i], key_x[i], key_mid);
      }
    }

    auth_pub_mul(key_x[m], model.bias.key_value[0], u[0]);
    for (size_t j = 1; j < weight_nd; j++) {
      FieldT key_mid;
      auth_pub_mul(key_mid, model.bias.key_value[j], u[j]);
      auth_mac_add(key_x[m], key_x[m], key_mid);
    }
  }

  void conv_calc_y(vector<FieldT> &key_y, const vector<FieldT> &v,
                   const ConvVerModel<FieldT> &model,
                   const QuantizedValueVerifier<FieldT> &input,
                   const QuantizedValueVerifier<FieldT> &output) {
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
        FieldT key_mid;
        if (input_l_idx < 0 || input_l_idx >= input_l)
          key_mid = FieldT::zero();
        else if (input_w_idx < 0 || input_w_idx >= input_w)
          key_mid = FieldT::zero();
        else {
          long input_idx =
              get_3D_idx(input_l, input_w, a, input_l_idx, input_w_idx);
          auth_pub_mul(key_mid, input.key_value[input_idx], v[idx]);
        }

        if (idx == 0)
          key_y[weight_idx] = key_mid;
        else
          auth_mac_add(key_y[weight_idx], key_y[weight_idx], key_mid);
      }
      if (idx == 0)
        key_y[m] = v[idx];
      else
        key_y[m] += v[idx];
    }
  }

  void conv_calc_z(FieldT &key_z, const vector<FieldT> &u,
                   const vector<FieldT> &v, const vector<FieldT> &key_c,
                   const QuantizedValueVerifier<FieldT> &output) {
    size_t output_d = output.shape[1];
    size_t output_l = output.shape[2];
    size_t output_w = output.shape[3];

    size_t n = output_d;
    size_t l = output_l * output_w;
    vector<FieldT> key_x(l);

#pragma omp parallel for
    for (size_t i = 0; i < l; i++) {
      auth_pub_mul(key_x[i], key_c[i], u[0]);
      for (size_t j = 1; j < n; j++) {
        FieldT key_mid;
        auth_pub_mul(key_mid, key_c[j * l + i], u[j]);
        auth_mac_add(key_x[i], key_x[i], key_mid);
      }
    }

    auth_pub_mul(key_z, key_x[0], v[0]);
    for (size_t i = 1; i < l; i++) {
      FieldT key_mid;
      auth_pub_mul(key_mid, key_x[i], v[i]);
      auth_mac_add(key_z, key_z, key_mid);
    }
  }

  void conv_get_poly_co_xy(vector<FieldT> &co_xy, size_t m, const FieldT &ym) {
    co_xy.resize(m + 3);
    for (size_t i = 0; i < m; i++) co_xy[i] = FieldT::one();
    co_xy[m] = ym;
    co_xy[m + 1] = -FieldT::one();
    co_xy[m + 2] = FieldT::zero();
  }

  void conv_get_poly_value_xy(vector<FieldT> &left_key,
                              vector<FieldT> &right_key,
                              const vector<FieldT> &key_x,
                              const vector<FieldT> &key_y,
                              const FieldT &key_z) {
    size_t m = key_x.size() - 1;
    left_key.resize(m);
    right_key.resize(m + 2);
    for (size_t i = 0; i < m; i++) {
      left_key[i] = key_y[i];
      right_key[i] = key_x[i];
    }
    right_key[m] = key_x[m];
    right_key[m + 1] = key_z;
  }

  template <typename Int>
  void zkp_conv(const ConvVerModel<FieldT> &model,
                const QuantizedValueVerifier<FieldT> &input,
                const QuantizedValueVerifier<FieldT> &output) {
    size_t R_size = output.size();

    vector<FieldT> key_mid;
    authenticated_val_input(key_mid, R_size);

    size_t n = model.weight.shape[0];
    size_t m =
        model.weight.shape[1] * model.weight.shape[2] * model.weight.shape[3];
    size_t l = output.shape[2] * output.shape[3];

    vector<FieldT> u(n);
    vector<FieldT> v(l);
    for (size_t i = 0; i < n; i++) u[i] = FieldT::random_element();
    for (size_t i = 0; i < l; i++) v[i] = FieldT::random_element();
    io->send_data(&u[0], n * sizeof(FieldT));
    io->send_data(&v[0], l * sizeof(FieldT));

    vector<FieldT> key_x(m + 1);
    vector<FieldT> key_y(m + 1);
    FieldT key_z;

    conv_calc_x(key_x, u, model);
    conv_calc_y(key_y, v, model, input, output);
    conv_calc_z(key_z, u, v, key_mid, output);

    vector<FieldT> co_xy;
    conv_get_poly_co_xy(co_xy, m, key_y[m]);
    vector<FieldT> left_key;
    vector<FieldT> right_key;
    conv_get_poly_value_xy(left_key, right_key, key_x, key_y, key_z);
    zkp_poly_deg2(left_key, right_key, co_xy);

    zkp_round<Int>(model.fp,
                   QuantizedValueVerifier<FieldT>(key_mid, output.shape),
                   output);
  }

  void zkp_mean(const QuantizedValueVerifier<FieldT> &input,
                const QuantizedValueVerifier<FieldT> &mean) {
    size_t d1 = input.shape[0];
    size_t d2 = input.shape[1];
    vector<FieldT> key_R(d1);

    // R = 2 * sum(x) - 2d * mean + d
    for (size_t i = 0; i < d1; i++) {
      FieldT key_sum = FieldT::zero();
      for (size_t j = 0; j < d2; j++) {
        auth_mac_add(key_sum, input.key_value[i * d2 + j], key_sum);
      }
      // R = 2 * sum
      auth_pub_mul(key_R[i], key_sum, FieldT(2));
      // R -= 2d * mean
      FieldT key_mid;
      auth_pub_mul(key_mid, mean.key_value[i], FieldT(2 * d2));
      auth_mac_sub(key_R[i], key_R[i], key_mid);
      // R += d
      auth_pub_add(key_R[i], key_R[i], FieldT(d2));
    }

    FieldT B = B_mean(d2);
    zkp_range(key_R, B, d1);
  }

  void zkp_var(const QuantizedValueVerifier<FieldT> &input,
               const QuantizedValueVerifier<FieldT> &mean,
               const QuantizedValueVerifier<FieldT> &var) {
    size_t d1 = input.shape[0];
    size_t d2 = input.shape[1];
    vector<FieldT> key_R;
    authenticated_val_input(key_R, d1);

    vector<FieldT> co = co_variance<FieldT>(d2);
    vector<FieldT> left_key(d2), right_key(d2 + 2);
    // 2 * sum((x - mean) ^ 2) - 2d * var - R + d < 2d
    for (size_t i = 0; i < d1; i++) {
      for (size_t j = 0; j < d2; j++) {
        left_key[j] = right_key[j] =
            input.key_value[i * d2 + j] - mean.key_value[i];
      }
      right_key[d2] = var.key_value[i];
      right_key[d2 + 1] = key_R[i];
      zkp_poly_deg2(left_key, right_key, co);
    }

    FieldT B = B_var(d2);
    zkp_range(key_R, B, d1);
  }

  template <typename Int>
  void zkp_std(const QuantizedValueVerifier<FieldT> &var,
               const QuantizedValueVerifier<FieldT> &std) {
    zkp_sqrt<Int>(var, std);
  }

  template <typename Int>
  void zkp_std_max(const QuantizedValueVerifier<FieldT> &std,
                   const QuantizedValueVerifier<FieldT> &std_max) {
    vector<FieldT> key_x = {FieldT(1) * delta};
    zkp_maximum<Int>(QuantizedValueVerifier<FieldT>(key_x, {1, 1}), std,
                     std_max);
  }

  template <typename Int>
  void zkp_norm(const NormFPVerifier<FieldT> fp,
                const QuantizedValueVerifier<FieldT> &input,
                const QuantizedValueVerifier<FieldT> &mean,
                const QuantizedValueVerifier<FieldT> &std_max,
                const QuantizedValueVerifier<FieldT> &sub,
                const QuantizedValueVerifier<FieldT> &norm) {
    // sub = round(2^e/c * (input - mean))
    zkp_round<Int>(fp, input - mean, sub);
    // y = round(sub / std_max)
    zkp_round<Int>(sub, std_max, norm);
  }

  template <typename Int>
  void zkp_layer_norm_out(const LayerNormVerModel<FieldT> &model,
                          const QuantizedValueVerifier<FieldT> &input,
                          const QuantizedValueVerifier<FieldT> &output) {
    // y = round(x * w + b)
    size_t sz = output.size();
    size_t d1 = output.shape[0], d2 = output.shape[1];
    // r = x * w + b
    QuantizedValueVerifier<FieldT> r(sz);
    r.shape = output.shape;
    r.auth(this);
    // verify x * w + b - r == 0
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
    // verify y == round(r)
    zkp_round<Int>(model.fp2, r, output);
  }

  template <typename Int>
  void zkp_layer_norm(const LayerNormVerModel<FieldT> &model,
                      const QuantizedValueVerifier<FieldT> &input,
                      const QuantizedValueVerifier<FieldT> &mean,
                      const QuantizedValueVerifier<FieldT> &var,
                      const QuantizedValueVerifier<FieldT> &std,
                      const QuantizedValueVerifier<FieldT> &std_max,
                      const QuantizedValueVerifier<FieldT> &sub,
                      const QuantizedValueVerifier<FieldT> &norm,
                      const QuantizedValueVerifier<FieldT> &output) {
    TIME_STATS_BEG
    // mean
    zkp_mean(input, mean);
    // variance
    zkp_var(input, mean, var);
    // std
    zkp_std<Int>(var, std);
    // maximum
    zkp_std_max<Int>(std, std_max);
    // norm
    zkp_norm<Int>(model.fp1, input, mean, std_max, sub, norm);
    // x * g + b
    zkp_layer_norm_out<Int>(model, norm, output);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_layer_norm(const LayerNormVerModel<FieldT> &model,
                      const QuantizedValueVerifier<FieldT> &input,
                      const QuantizedValueVerifier<FieldT> &mean,
                      const QuantizedValueVerifier<FieldT> &var,
                      const QuantizedValueVerifier<FieldT> &std,
                      const QuantizedValueVerifier<FieldT> &std_max,
                      const QuantizedValueVerifier<FieldT> &sub,
                      const QuantizedValueVerifier<FieldT> &norm) {
    TIME_STATS_BEG
    // mean
    zkp_mean(input, mean);
    // variance
    zkp_var(input, mean, var);
    // std
    zkp_std<Int>(var, std);
    // maximum
    zkp_std_max<Int>(std, std_max);
    // norm
    zkp_norm<Int>(model.fp1, input, mean, std_max, sub, norm);
    TIME_STATS_END
  }

  void calc_x(vector<FieldT> &key_x, const vector<FieldT> &u,
              const QuantizedValueVerifier<FieldT> &A) {
    size_t n = A.shape[0];
    size_t m = A.shape[1];
    assert(key_x.size() == m + 1 || key_x.size() == m);
    assert(u.size() == n);

    FieldT key_mid;
    // x.T = u.T * A
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        size_t k = j * m + i;
        auth_pub_mul(key_mid, A.key_value[k], u[j]);
        auth_mac_add(key_x[i], key_x[i], key_mid);
      }
    }
    // for bias, x[m] = sum(u)
    if (key_x.size() == m + 1) {
      for (size_t j = 0; j < n; j++) {
        key_x[m] += u[j];
      }
    }
  }

  void calc_y(vector<FieldT> &key_y, const vector<FieldT> &v,
              const QuantizedValueVerifier<FieldT> &B,
              const QuantizedValueVerifier<FieldT> *bias) {
    size_t m = B.shape[0];
    size_t l = B.shape[1];
    assert(key_y.size() == m + 1 || key_y.size() == m);
    assert(v.size() == l);
    assert(!bias || bias->shape[0] == l);

// y = B * v
#pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < l; j++) {
        size_t k = i * l + j;
        FieldT key_mid;
        auth_pub_mul(key_mid, B.key_value[k], v[j]);
        auth_mac_add(key_y[i], key_y[i], key_mid);
      }
    }
    // bias
    if (bias) {
      assert(key_y.size() == m + 1);
      FieldT key_mid;
      for (size_t j = 0; j < l; j++) {
        auth_pub_mul(key_mid, bias->key_value[j], v[j]);
        auth_mac_add(key_y[m], key_y[m], key_mid);
      }
    }
  }

  void calc_z(FieldT &key_z, const vector<FieldT> &u, const vector<FieldT> &v,
              const vector<FieldT> &key_C, size_t n, size_t l) {
    assert(u.size() == n);
    assert(v.size() == l);

    // z = u.T * C * v
    for (size_t i = 0; i < l; i++) {
      // t = u.T * C
      FieldT key_t;
      FieldT key_mid;
      for (size_t j = 0; j < n; j++) {
        size_t k = j * l + i;
        auth_pub_mul(key_mid, key_C[k], u[j]);
        auth_mac_add(key_t, key_t, key_mid);
      }
      // z += t * v
      auth_pub_mul(key_t, key_t, v[i]);
      auth_mac_add(key_z, key_z, key_t);
    }
  }

  template <typename Int>
  void zkp_matrix_mul(const QuantizedValueVerifier<FieldT> &in1,
                      const QuantizedValueVerifier<FieldT> &in2,
                      const QuantizedValueVerifier<FieldT> *in3,
                      const QuantizedValueVerifier<FieldT> &out,
                      const NormFPVerifier<FieldT> &fp) {
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
    vector<FieldT> key_mid;
    authenticated_val_input(key_mid, sz);

    // verify mid = in1 @ w + b
    vector<FieldT> u(n);
    vector<FieldT> v(l);
    for (size_t i = 0; i < n; i++) u[i] = FieldT::random_element();
    for (size_t i = 0; i < l; i++) v[i] = FieldT::random_element();
    io->send_data(&u[0], n * sizeof(FieldT));
    io->send_data(&v[0], l * sizeof(FieldT));

    size_t len = in3 ? m + 1 : m;
    vector<FieldT> key_x(len), key_y(len);
    FieldT key_z;
    // x
    calc_x(key_x, u, in1);
    // y
    calc_y(key_y, v, in2, in3);
    // z
    calc_z(key_z, u, v, key_mid, n, l);

    vector<FieldT> co(m, FieldT(1)), left_key, right_key;
    if (in3) co.push_back(key_x[m]);
    co.push_back(FieldT(-1));
    co.push_back(FieldT(0));
    left_key = key_x;
    if (in3) left_key.pop_back();
    right_key = key_y;
    right_key.push_back(key_z);
    zkp_poly_deg2(left_key, right_key, co);

    zkp_round<Int>(fp, QuantizedValueVerifier<FieldT>(key_mid, out.shape), out);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_linear(const LinearVerModel<FieldT> &model,
                  const QuantizedValueVerifier<FieldT> &input,
                  const QuantizedValueVerifier<FieldT> &output) {
    TIME_STATS_BEG
    zkp_matrix_mul<Int>(input, model.weight, &model.bias, output, model.fp);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_poly(const NormFPVerifier<FieldT> &fp1,
                const NormFPVerifier<FieldT> &fp2,
                const NormFPVerifier<FieldT> &fp3,
                const QuantizedValueVerifier<FieldT> &input,
                const QuantizedValueVerifier<FieldT> &output1,
                const QuantizedValueVerifier<FieldT> &output2,
                const QuantizedValueVerifier<FieldT> &output3,
                const QuantizedValueVerifier<FieldT> &output) {
    // xx = x^2
    QuantizedValueVerifier<FieldT> xx(input.size());
    xx.auth(this);
    zkp_mul(input, input, xx);
    // y1 = round(2^e1/c1 * x^2)
    zkp_round<Int>(fp1, xx, output1);
    // y2 = round(2^e2/c2 * x)
    zkp_round<Int>(fp2, input, output2);
    // y3 = round(2^e3/c3)
    zkp_round<Int>(fp3, output3);
  }

  template <typename Int>
  void zkp_exp(const SftMaxVerModel<FieldT> &model,
               const QuantizedValueVerifier<FieldT> &input,
               const QuantizedValueVerifier<FieldT> &z,
               const QuantizedValueVerifier<FieldT> &p1,
               const QuantizedValueVerifier<FieldT> &p2,
               const QuantizedValueVerifier<FieldT> &p,
               const QuantizedValueVerifier<FieldT> &l,
               const QuantizedValueVerifier<FieldT> &poly_out1,
               const QuantizedValueVerifier<FieldT> &poly_out2,
               const QuantizedValueVerifier<FieldT> &poly_out3,
               const QuantizedValueVerifier<FieldT> &output) {
    // z = floor(input * fp1)
    zkp_floor<Int>(model.fp1, input * FieldT(-1), z);
    // p = round(sx/sp * input + ln2/sp * z)
    zkp_add<Int>(model.fp2, model.fp3, input, z, p1, p2, p);
    // l = round(1/st * poly(p))
    zkp_poly<Int>(model.fp4, model.fp5, model.fp6, p, poly_out1, poly_out2,
                  poly_out3, l);
    // t = l >> z
    zkp_right_shift<Int>(l, z, output);
  }

  template <typename Int>
  void zkp_softmax_requant(const NormFPVerifier<FieldT> &fp,
                           const QuantizedValueVerifier<FieldT> &input,
                           const QuantizedValueVerifier<FieldT> &output1,
                           const QuantizedValueVerifier<FieldT> &output) {
    // output1 = round(2^e/c * input)
    zkp_round<Int>(fp, input, output1);
    // output = round(output1 / sum(output1))
    size_t d1 = output.shape[0], d2 = output.shape[1];
    vector<FieldT> key_sum(d1);
    for (size_t i = 0; i < d1; i++) {
      for (size_t j = 0; j < d2; j++) {
        size_t k = i * d2 + j;
        key_sum[i] += input.key_value[k];
      }
    }
    QuantizedValueVerifier<FieldT> r(key_sum, {d1, 1});
    zkp_round<Int>(output1, r, output);
  }

  template <typename Int>
  void zkp_qk(const AttnVerModel<FieldT> &model,
              const QuantizedValueVerifier<FieldT> &q,
              const QuantizedValueVerifier<FieldT> &k,
              const QuantizedValueVerifier<FieldT> &qk) {
    zkp_matrix_mul<Int>(q, k, NULL, qk, model.fp1);
  }

  template <typename Int>
  void zkp_divqk(const AttnVerModel<FieldT> &model,
                 const QuantizedValueVerifier<FieldT> &input,
                 const QuantizedValueVerifier<FieldT> &output) {
    zkp_round<Int>(model.fp2, input, output);
  }

  template <typename Int>
  void zkp_softmax(const SftMaxVerModel<FieldT> &model,
                   const GPT2VerData<FieldT> &data, int i) {
    TIME_STATS_BEG
    zkp_max<Int>(data.divqk[i], data.x_max[i]);
    zkp_exp<Int>(model, data.divqk[i] - data.x_max[i], data.z[i], data.p1[i],
                 data.p2[i], data.p[i], data.l[i], data.exp_poly_out1[i],
                 data.exp_poly_out2[i], data.exp_poly_out3[i], data.exp_out[i]);
    zkp_softmax_requant<Int>(model.fp7, data.exp_out[i], data.softmax_out1[i],
                             data.softmax_out[i]);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_softmax(const SftMaxVerModel<FieldT> &model,
                   const QuantizedValueVerifier<FieldT> &divqk,
                   const QuantizedValueVerifier<FieldT> &x_max,
                   const QuantizedValueVerifier<FieldT> &z,
                   const QuantizedValueVerifier<FieldT> &p,
                   const QuantizedValueVerifier<FieldT> &p1,
                   const QuantizedValueVerifier<FieldT> &p2,
                   const QuantizedValueVerifier<FieldT> &l,
                   const QuantizedValueVerifier<FieldT> &exp_out,
                   const QuantizedValueVerifier<FieldT> &exp_poly_out1,
                   const QuantizedValueVerifier<FieldT> &exp_poly_out2,
                   const QuantizedValueVerifier<FieldT> &exp_poly_out3,
                   const QuantizedValueVerifier<FieldT> &softmax_out,
                   const QuantizedValueVerifier<FieldT> &softmax_out1) {
    TIME_STATS_BEG
    zkp_max<Int>(divqk, x_max);
    zkp_exp<Int>(model, divqk - x_max, z, p1, p2, p, l, exp_poly_out1,
                 exp_poly_out2, exp_poly_out3, exp_out);
    zkp_softmax_requant<Int>(model.fp7, exp_out, softmax_out1, softmax_out);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_qkv(const AttnVerModel<FieldT> &model,
               const QuantizedValueVerifier<FieldT> &softmax_out,
               const QuantizedValueVerifier<FieldT> &v,
               const QuantizedValueVerifier<FieldT> &output) {
    zkp_matrix_mul<Int>(softmax_out, v, NULL, output, model.fp3);
  }

  template <typename Int>
  void zkp_attn(const AttnVerModel<FieldT> &model,
                const GPT2VerData<FieldT> &data, int i) {
    zkp_qk<Int>(model, data.q[i], data.k[i], data.qk[i]);
    zkp_divqk<Int>(model, data.qk[i], data.divqk[i]);
    zkp_softmax<Int>(model.softmax, data, i);
    zkp_qkv<Int>(model, data.softmax_out[i], data.v[i], data.qkv[i]);
  }

  template <typename Int>
  void zkp_attn_without_softmax(const AttnVerModel<FieldT> &model,
                const GPT2VerData<FieldT> &data, int i) {
    zkp_qk<Int>(model, data.q[i], data.k[i], data.qk[i]);
    zkp_divqk<Int>(model, data.qk[i], data.divqk[i]);
    zkp_qkv<Int>(model, data.softmax_out[i], data.v[i], data.qkv[i]);
  }

  template <typename Int>
  void zkp_mha(const MHAVerModel<FieldT> &model,
               const GPT2VerData<FieldT> &data, int i) {
    zkp_linear<Int>(model.linear[0], data.layer_norm_out[2 * i],
                    data.linear_out[4 * i]);
    zkp_linear<Int>(model.linear[1], data.mha_out[i],
                    data.linear_out[4 * i + 1]);
    for (int j = 0; j < N_HEAD; j++) {
      zkp_attn<Int>(model.attn[j], data, i * N_HEAD + j);
    }
  }

  template <typename Int>
  void zkp_mha_without_softmax(const MHAVerModel<FieldT> &model,
               const GPT2VerData<FieldT> &data, int i) {
    zkp_linear<Int>(model.linear[0], data.layer_norm_out[2 * i],
                    data.linear_out[4 * i]);
    zkp_linear<Int>(model.linear[1], data.mha_out[i],
                    data.linear_out[4 * i + 1]);
    for (int j = 0; j < N_HEAD; j++) {
      zkp_attn_without_softmax<Int>(model.attn[j], data, i * N_HEAD + j);
    }
  }

  template <typename Int>
  void zkp_gelu_zb(const GeluVerModel<FieldT> &model,
                   const QuantizedValueVerifier<FieldT> &zb) {
    zkp_floor<Int>(model.fp1, zb);
  }

  void zkp_gelu_erf_out(const QuantizedValueVerifier<FieldT> &l,
                        const QuantizedValueVerifier<FieldT> &sign,
                        const QuantizedValueVerifier<FieldT> &t) {
    zkp_mul(l, sign, t);
  }

  template <typename Int>
  void zkp_gelu_out(const GeluVerModel<FieldT> &model,
                    const QuantizedValueVerifier<FieldT> &input,
                    const QuantizedValueVerifier<FieldT> &t,
                    const QuantizedValueVerifier<FieldT> &output1,
                    const QuantizedValueVerifier<FieldT> &output2,
                    const QuantizedValueVerifier<FieldT> &output) {
    // y = round((sx/(2*sy)) * x * (1 + st * t))
    //   = round((sx/(2*sy)) * x + (sx*st/(2*sy) * x * t))

    // r = x * t
    size_t sz = input.size();
    QuantizedValueVerifier<FieldT> r(sz);
    r.auth(this);
    zkp_mul(input, t, r);
    zkp_add<Int>(model.fp5, model.fp6, input, r, output1, output2, output);
  }

  template <typename Int>
  void zkp_gelu(const GeluVerModel<FieldT> &model,
                const QuantizedValueVerifier<FieldT> &input,
                const QuantizedValueVerifier<FieldT> &sign,
                const QuantizedValueVerifier<FieldT> &q_abs,
                const QuantizedValueVerifier<FieldT> &zb,
                const QuantizedValueVerifier<FieldT> &q_min,
                const QuantizedValueVerifier<FieldT> &poly_out1,
                const QuantizedValueVerifier<FieldT> &poly_out2,
                const QuantizedValueVerifier<FieldT> &poly_out3,
                const QuantizedValueVerifier<FieldT> &l,
                const QuantizedValueVerifier<FieldT> &t,
                const QuantizedValueVerifier<FieldT> &output1,
                const QuantizedValueVerifier<FieldT> &output2,
                const QuantizedValueVerifier<FieldT> &output) {
    TIME_STATS_BEG
    zkp_sign<Int>(input, sign);
    zkp_abs<Int>(input, sign, q_abs);
    zkp_gelu_zb<Int>(model, zb);
    zkp_minimum<Int>(zb, q_abs, q_min);
    // l = round(1/st * poly(sx/sqrt(2) * mn))
    zkp_poly<Int>(model.fp2, model.fp3, model.fp4, q_min, poly_out1, poly_out2,
                  poly_out3, l);
    zkp_gelu_erf_out(l, sign, t);
    zkp_gelu_out<Int>(model, input, t, output1, output2, output);
    TIME_STATS_END
  }

  template <typename Int>
  void zkp_ffn(const FFNVerModel<FieldT> &model,
               const GPT2VerData<FieldT> &data, int i) {
    zkp_linear<Int>(model.linear[0], data.layer_norm_out[2 * i + 1],
                    data.linear_out[4 * i + 2]);
    zkp_gelu<Int>(model.gelu, data.linear_out[4 * i + 2], data.sign[i],
                  data.q_abs[i], data.zb[i], data.q_min[i],
                  data.gelu_poly_out1[i], data.gelu_poly_out2[i],
                  data.gelu_poly_out3[i], data.erf_l[i], data.erf_out[i],
                  data.gelu_out1[i], data.gelu_out2[i], data.gelu_out[i]);
    zkp_linear<Int>(model.linear[1], data.gelu_out[i],
                    data.linear_out[4 * i + 3]);
  }

  template <typename Int>
  void zkp_ffn_without_gelu(const FFNVerModel<FieldT> &model,
               const GPT2VerData<FieldT> &data, int i) {
    zkp_linear<Int>(model.linear[0], data.layer_norm_out[2 * i + 1],
                    data.linear_out[4 * i + 2]);
    zkp_linear<Int>(model.linear[1], data.gelu_out[i],
                    data.linear_out[4 * i + 3]);
  }

  template <typename Int>
  void zkp_trans(const TransVerModel<FieldT> *models,
                 const GPT2VerData<FieldT> &data) {
    const QuantizedValueVerifier<FieldT> *in = &data.embd_out;
    for (int i = 0; i < TRANS_USED; i++) {
      printf("%d\n", i);
      zkp_layer_norm<Int>(models[i].layer_norm[0], *in, data.mean[2 * i],
                          data.var[2 * i], data.std[2 * i], data.std_max[2 * i],
                          data.sub[2 * i], data.norm[2 * i],
                          data.layer_norm_out[2 * i]);
      zkp_mha<Int>(models[i].mha, data, i);
      zkp_res<Int>(models[i].res[0], *in, data.linear_out[4 * i + 1],
                   data.res_out1[2 * i], data.res_out2[2 * i],
                   data.res_out[2 * i]);
      zkp_layer_norm<Int>(models[i].layer_norm[1], data.res_out[2 * i],
                          data.mean[2 * i + 1], data.var[2 * i + 1],
                          data.std[2 * i + 1], data.std_max[2 * i + 1],
                          data.sub[2 * i + 1], data.norm[2 * i + 1],
                          data.layer_norm_out[2 * i + 1]);
      zkp_ffn<Int>(models[i].ffn, data, i);
      zkp_res<Int>(models[i].res[1], data.res_out[2 * i],
                   data.linear_out[4 * i + 3], data.res_out1[2 * i + 1],
                   data.res_out2[2 * i + 1], data.res_out[2 * i + 1]);
      in = &data.res_out[2 * i + 1];
    }
  }
};
#endif
