#ifndef OSTRIPLE_COMMON_H
#define OSTRIPLE_COMMON_H

#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "utility"

#define CEIL(x, y) ((x) + (y) - 1) / (y)

#define CONCATE(arg1, arg2) CONCATE1(arg1, arg2)
#define CONCATE1(arg1, arg2) CONCATE2(arg1, arg2)
#define CONCATE2(arg1, arg2) arg1##arg2

#define CONVERT_1(type, ident, ...) type(ident)
#define CONVERT_2(type, ident, ...) type(ident), CONVERT_1(type, __VA_ARGS__)
#define CONVERT_3(type, ident, ...) type(ident), CONVERT_2(type, __VA_ARGS__)
#define CONVERT_4(type, ident, ...) type(ident), CONVERT_3(type, __VA_ARGS__)
#define CONVERT_5(type, ident, ...) type(ident), CONVERT_4(type, __VA_ARGS__)
#define CONVERT_6(type, ident, ...) type(ident), CONVERT_5(type, __VA_ARGS__)
#define CONVERT_7(type, ident, ...) type(ident), CONVERT_6(type, __VA_ARGS__)
#define CONVERT_8(type, ident, ...) type(ident), CONVERT_7(type, __VA_ARGS__)

#define NARGS(...) NARGS_(__VA_ARGS__, SEQ_N())
#define NARGS_(...) ARG_N(__VA_ARGS__)
#define ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define SEQ_N() 8, 7, 6, 5, 4, 3, 2, 1, 0

#define CONVERT_(N, type, ...) CONCATE(CONVERT_, N)(type, __VA_ARGS__)
#define CONVERT(type, ...) CONVERT_(NARGS(__VA_ARGS__), type, __VA_ARGS__)

#define QUANT_UMAX(n) (1ll << (n))
#define QUANT_IMAX(n) (1ll << ((n) - 1))

#define B_max(n) FieldT(QUANT_UMAX(n))
#define B_maximum(n) FieldT(QUANT_UMAX(n))
#define B_minimum(n) FieldT(QUANT_UMAX(n))
#define B_sign(n) FieldT(QUANT_IMAX(n))
#define B_maxpool(n) FieldT(QUANT_UMAX(n))
#define B_round() FieldT(1ll << (25))
#define B_round2(n) (FieldT(QUANT_IMAX(n + 10)))
#define B_floor() FieldT(QUANT_UMAX(24))
#define B_mean(n) FieldT(2 * (n))
#define B_var(n) FieldT(2 * (n))
#define B_std(n) FieldT(1ll << n)
#define B_shift(n) FieldT(QUANT_UMAX(n))

template <typename FieldT>
std::vector<FieldT> co_max() {
  return {CONVERT(FieldT, 1, -1, 0)};
}

template <typename FieldT>
std::vector<FieldT> co_maximum(size_t n) {
  std::vector<FieldT> co(n, FieldT(1));
  co.push_back(FieldT(0));
  return co;
}

template <typename FieldT>
std::vector<FieldT> co_minimum(size_t n) {
  std::vector<FieldT> co(n, FieldT(1));
  co.push_back(FieldT(0));
  return co;
}

template <typename FieldT>
std::vector<FieldT> co_variance(size_t n) {
  std::vector<FieldT> co(n, FieldT(2));
  co.insert(co.end(), {CONVERT(FieldT, -2 * n, -1, n)});
  return co;
}

#endif  // OSTRIPLE_COMMON_H
