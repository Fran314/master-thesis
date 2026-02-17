#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#define NORMALIZE_PRECISION 48
#define NORMALIZE_WEIGHT ((__int128)1 << NORMALIZE_PRECISION)

#define PATH_BUF_LEN 128
#define FULL_PATH(dir, layer, variable, cnt)                                  \
  ({                                                                          \
    char __ret[PATH_BUF_LEN];                                                 \
    int n =                                                                   \
        sprintf(__ret, "%s/%s_%s_%u", dir, #layer, #variable, (unsigned)cnt); \
    assert(n < PATH_BUF_LEN);                                                 \
    __ret;                                                                    \
  })

#define ARRLEN(arr) (sizeof(arr) / sizeof(arr[0]))
#define FOR_EACH(arr, ...) \
  for (size_t i = 0; i < ARRLEN(arr); i++) __VA_ARGS__
#define FOR_EACH_ASSIGN(arr, f) \
  for (size_t i = 0; i < ARRLEN(arr); i++) arr[i] = f;
#define FOR_EACH_DO(arr, f) \
  for (size_t i = 0; i < ARRLEN(arr); i++) arr[i].f

template <typename IO, typename FieldT>
class FpOSTriplePrv;
template <typename IO, typename FieldT>
class FpOSTripleVer;

template <typename FieldT>
class NormFPProver {
 public:
  // coefficient
  __int128 icoeff;
  FieldT coeff;
  FieldT mac_coeff;
  // exponent
  int iexp;
  FieldT power;
  FieldT mac_power;

  NormFPProver() {}
  NormFPProver(float f, bool inv = true) {
    // 1/f = coeff * 2^exp
    // f = 2^-exp / coeff
    if (inv) f = 1 / f;
    icoeff = (__int128)(f * NORMALIZE_WEIGHT);
    int cnt = 0;
    while (!(icoeff & 1) && cnt < NORMALIZE_PRECISION) {
      icoeff >>= 1;
      cnt++;
    }
    iexp = NORMALIZE_PRECISION - cnt;
//    printf("%f = 2^%ld / %ld\n", 1/f, iexp, icoeff);

    coeff = FieldT(icoeff);
    power = FieldT(1ll << iexp);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    ostriple->authenticated_val_input(mac_coeff, coeff);
    ostriple->authenticated_val_input(mac_power, power);
  }
};

template <typename FieldT>
class NormFPVerifier {
 public:
  FieldT key_coeff;
  FieldT key_power;

  NormFPVerifier() {}

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    ostriple->authenticated_val_input(key_coeff);
    ostriple->authenticated_val_input(key_power);
  }
};

template <typename FieldT, typename Int>
class QuantizedValueProver {
 public:
  // shape of value
  std::vector<size_t> shape;
  // quantized numbers
  std::vector<Int> ivalue;
  std::vector<FieldT> value;
  std::vector<FieldT> mac_value;
  // scaling factor
  float fscale;
  NormFPProver<FieldT> scale;
  // zero point
  Int iz;
  //  FieldT z;
  //  FieldT mac_z;

  QuantizedValueProver() {}
  QuantizedValueProver(size_t sz) : value(sz), mac_value(sz) {}
  QuantizedValueProver(vector<FieldT> &value, vector<FieldT> &mac_value,
                       vector<size_t> shape)
      : value(value), mac_value(mac_value), shape(shape) {}
  /**
   * `read` function should read the following fields from `filename`:
   *
   * shape/ivalue/fscale/iz
   */
  QuantizedValueProver(
      char *filename,
      std::function<size_t(char *, QuantizedValueProver<FieldT, Int> *)> read) {
    size_t sz = read(filename, this);

    scale = NormFPProver<FieldT>(fscale);
    //    z = FieldT(iz);
    value.resize(sz);
    for (size_t i = 0; i < sz; i++) {
      value[i] = FieldT(ivalue[i]);
    }
  }

  size_t size() const { return value.size(); }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    ostriple->authenticated_val_input(mac_value, value, size());
    //    ostriple->authenticated_val_input(mac_z, z);
  }

  QuantizedValueProver<FieldT, Int> operator+(
      const QuantizedValueProver<FieldT, Int> &o) {
    //    assert(shape.size() == o.shape.size());

    size_t sz = max(size(), o.size());
    size_t f1 = sz / size(), f2 = sz / o.size();
    vector<FieldT> v(sz), mac_v(sz);
    for (size_t i = 0; i < sz; i++) {
      size_t idx1 = i / f1;
      size_t idx2 = i / f2;
      v[i] = value[idx1] + o.value[idx2];
      mac_v[i] = mac_value[idx1] + o.mac_value[idx2];
    }
    vector<size_t> v_shape = sz == size() ? shape : o.shape;
    return QuantizedValueProver<FieldT, Int>(v, mac_v, v_shape);
  }

  // public addition
  QuantizedValueProver<FieldT, Int> operator+(const FieldT &o) const {
    size_t sz = size();
    vector<FieldT> v(sz), mac_v(sz);
    for (size_t i = 0; i < sz; i++) {
      v[i] = value[i] + o;
    }
    return QuantizedValueProver<FieldT, Int>(v, mac_v, shape);
  }

  QuantizedValueProver<FieldT, Int> operator-(
      const QuantizedValueProver<FieldT, Int> &o) const {
    //    assert(shape.size() == o.shape.size());

    size_t sz = max(size(), o.size());
    size_t f1 = sz / size(), f2 = sz / o.size();
    vector<FieldT> v(sz), mac_v(sz);
    for (size_t i = 0; i < sz; i++) {
      size_t idx1 = i / f1;
      size_t idx2 = i / f2;
      // additive homomorphic property
      v[i] = value[idx1] - o.value[idx2];
      mac_v[i] = mac_value[idx1] - o.mac_value[idx2];
    }
    vector<size_t> v_shape = sz == size() ? shape : o.shape;
    return QuantizedValueProver<FieldT, Int>(v, mac_v, v_shape);
  }

  // public subtraction
  QuantizedValueProver<FieldT, Int> operator-(const FieldT &o) const {
    size_t sz = size();
    vector<FieldT> v(sz), mac_v(sz);
    for (size_t i = 0; i < sz; i++) {
      v[i] = value[i] - o;
      mac_v[i] = mac_value[i] - o;
    }
    return QuantizedValueProver<FieldT, Int>(v, mac_v, shape);
  }

  QuantizedValueProver<FieldT, Int> operator*(
      const QuantizedValueProver<FieldT, Int> &o) const {
    //    assert(shape.size() == o.shape.size());

    size_t sz = max(size(), o.size());
    size_t f1 = sz / size(), f2 = sz / o.size();
    vector<FieldT> v(sz), mac_v(sz);
    for (size_t i = 0; i < sz; i++) {
      size_t idx1 = i / f1;
      size_t idx2 = i / f2;
      // there is no multiplicative homomorphic property
      v[i] = value[idx1] * o.value[idx2];
    }
    vector<size_t> v_shape = sz == size() ? shape : o.shape;
    return QuantizedValueProver<FieldT, Int>(v, mac_v, v_shape);
  }

  // public multiplication
  QuantizedValueProver<FieldT, Int> operator*(const FieldT &o) const {
    size_t sz = size();
    vector<FieldT> v(sz), mac_v(sz);
    for (size_t i = 0; i < sz; i++) {
      v[i] = value[i] * o;
      mac_v[i] = mac_value[i] * o;
    }
    return QuantizedValueProver<FieldT, Int>(v, mac_v, shape);
  }

  QuantizedValueProver<FieldT, Int> operator^(
      const QuantizedValueProver<FieldT, Int> &o) const {
    //    assert(shape.size() == o.shape.size());

    size_t sz = max(size(), o.size());
    size_t f1 = sz / size(), f2 = sz / o.size();
    vector<FieldT> v(sz), mac_v(sz);
    for (size_t i = 0; i < sz; i++) {
      size_t idx1 = i / f1;
      size_t idx2 = i / f2;
      v[i] = value[idx1] ^ o.value[idx2].as_bigint();
    }
    vector<size_t> v_shape = sz == size() ? shape : o.shape;
    return QuantizedValueProver<FieldT, Int>(v, mac_v, v_shape);
  }
};

template <typename FieldT>
class QuantizedValueVerifier {
 public:
  // shape of value
  std::vector<size_t> shape;
  // quantized numbers
  std::vector<FieldT> key_value;
  // zero point
  //  FieldT key_z;

  QuantizedValueVerifier() {}
  QuantizedValueVerifier(size_t sz) : key_value(sz) {}
  QuantizedValueVerifier(vector<FieldT> &key_value, vector<size_t> shape)
      : key_value(key_value), shape(shape) {}
  /**
   * `read` function should read the following fields from `filename`:
   *
   * shape
   */
  QuantizedValueVerifier(
      char *filename,
      std::function<size_t(char *, QuantizedValueVerifier<FieldT> *)> read) {
    size_t sz = read(filename, this);
    key_value.resize(sz);
  }

  size_t size() const { return key_value.size(); }

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    ostriple->authenticated_val_input(key_value, size());
    //    ostriple->authenticated_val_input(key_z);
  }

  QuantizedValueVerifier<FieldT> operator+(
      const QuantizedValueVerifier<FieldT> &o) {
    //    assert(shape.size() == o.shape.size());

    size_t sz = max(size(), o.size());
    size_t f1 = sz / size(), f2 = sz / o.size();
    vector<FieldT> key_v(sz);
    for (size_t i = 0; i < sz; i++) {
      size_t idx1 = i / f1;
      size_t idx2 = i / f2;
      key_v[i] = key_value[idx1] + o.key_value[idx2];
    }
    vector<size_t> v_shape = sz == size() ? shape : o.shape;
    return QuantizedValueVerifier<FieldT>(key_v, v_shape);
  }

  QuantizedValueVerifier<FieldT> operator-(
      const QuantizedValueVerifier<FieldT> &o) const {
    //    assert(shape.size() == o.shape.size());

    size_t sz = max(size(), o.size());
    size_t f1 = sz / size(), f2 = sz / o.size();
    vector<FieldT> key_v(sz);
    for (size_t i = 0; i < sz; i++) {
      size_t idx1 = i / f1;
      size_t idx2 = i / f2;
      key_v[i] = key_value[idx1] - o.key_value[idx2];
    }
    vector<size_t> v_shape = sz == size() ? shape : o.shape;
    return QuantizedValueVerifier<FieldT>(key_v, v_shape);
  }

  QuantizedValueVerifier<FieldT> operator*(const FieldT& o) const {
    size_t sz = size();
    vector<FieldT> key_v(sz);
    for (size_t i = 0; i < sz; i++) {
      key_v[i] = key_value[i] * o;
    }
    return QuantizedValueVerifier<FieldT>(key_v, shape);
  }
};

#endif  // __NETWORK_H__
