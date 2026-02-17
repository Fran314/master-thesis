#ifndef __TRANSFORMER_PROVER_H__
#define __TRANSFORMER_PROVER_H__

#include <fstream>

#include "quantized-value.h"
#include "transformer-common.h"

#define READ_FUNC transformer::read_vec<FieldT, Int>
#define LOAD_DATA_PATH(path, out) \
  out = QuantizedValueProver<FieldT, Int>(path, READ_FUNC)
#define LOAD_DATA(dir, layer, variable, cnt, out) \
  LOAD_DATA_PATH(FULL_PATH(dir, layer, variable, cnt), out)
#define LOAD_DATA_ARR(dir, layer, variable, arr) \
  FOR_EACH(arr, LOAD_DATA(dir, layer, variable, i, arr[i]))
#define LOAD_N(dir, layer, variable, cnt, out) \
  read_number(FULL_PATH(dir, layer, variable, cnt), out)

namespace transformer {

template <typename FieldT, typename Int>
size_t read_vec(char *filename, QuantizedValueProver<FieldT, Int> *x) {
  // open file
  std::fstream file(filename, std::ios_base::in);
  if (!file.is_open()) {
    std::perror(filename);
    exit(1);
  }

  // iz
  x->iz = 0;
  // fscale
  size_t ndim;
  file >> x->fscale >> ndim;
  // shape
  assert(x->shape.empty());
  x->shape.resize(ndim);
  size_t sz = 1;
  for (size_t i = 0; i < ndim; i++) {
    int d;
    file >> d;
    x->shape[i] = d;
    sz *= x->shape[i];
  }
  x->ivalue.resize(sz);
  for (size_t i = 0; i < sz; i++) {
    Int v;
    file >> v;
    x->ivalue[i] = v;
  }

  file.close();
  return sz;
}

}  // namespace transformer

template <typename Number>
static void read_number(char *filename, Number &x) {
  static_assert(std::is_same<Number, float>() || std::is_same<Number, long>(),
                "number is not float or long");
  std::fstream file(filename, std::ios_base::in);
  if (!file.is_open()) {
    std::perror(filename);
    exit(1);
  }
  file >> x;

  file.close();
}

template <typename FieldT, typename Int>
class GPT2PrvData {
 public:
  // embedding output
  QuantizedValueProver<FieldT, Int> embd_out;
  // exp
  QuantizedValueProver<FieldT, Int> z[TRANS_NUM * N_HEAD];
  QuantizedValueProver<FieldT, Int> p[TRANS_NUM * N_HEAD];
  QuantizedValueProver<FieldT, Int> p1[TRANS_NUM * N_HEAD];
  QuantizedValueProver<FieldT, Int> p2[TRANS_NUM * N_HEAD];
  QuantizedValueProver<FieldT, Int> l[TRANS_NUM * N_HEAD];
  QuantizedValueProver<FieldT, Int> exp_poly_out1[TRANS_NUM * N_HEAD];
  QuantizedValueProver<FieldT, Int> exp_poly_out2[TRANS_NUM * N_HEAD];
  QuantizedValueProver<FieldT, Int> exp_poly_out3[TRANS_NUM * N_HEAD];
  QuantizedValueProver<FieldT, Int> exp_out[TRANS_NUM * N_HEAD];
  // softmax
  QuantizedValueProver<FieldT, Int> x_max[TRANS_NUM * N_HEAD];
  QuantizedValueProver<FieldT, Int> softmax_out[TRANS_NUM * N_HEAD];
  QuantizedValueProver<FieldT, Int> softmax_out1[TRANS_NUM * N_HEAD];
  // gelu
  QuantizedValueProver<FieldT, Int> sign[TRANS_NUM];
  QuantizedValueProver<FieldT, Int> q_abs[TRANS_NUM];
  QuantizedValueProver<FieldT, Int> zb[TRANS_NUM];
  QuantizedValueProver<FieldT, Int> q_min[TRANS_NUM];
  QuantizedValueProver<FieldT, Int> erf_l[TRANS_NUM];
  QuantizedValueProver<FieldT, Int> gelu_poly_out1[TRANS_NUM];
  QuantizedValueProver<FieldT, Int> gelu_poly_out2[TRANS_NUM];
  QuantizedValueProver<FieldT, Int> gelu_poly_out3[TRANS_NUM];
  QuantizedValueProver<FieldT, Int> erf_out[TRANS_NUM];
  QuantizedValueProver<FieldT, Int> gelu_out[TRANS_NUM];
  QuantizedValueProver<FieldT, Int> gelu_out1[TRANS_NUM];
  QuantizedValueProver<FieldT, Int> gelu_out2[TRANS_NUM];
  // layer normalization
  QuantizedValueProver<FieldT, Int> mean[2 * TRANS_NUM + 1];
  QuantizedValueProver<FieldT, Int> var[2 * TRANS_NUM + 1];
  QuantizedValueProver<FieldT, Int> std[2 * TRANS_NUM + 1];
  QuantizedValueProver<FieldT, Int> std_max[2 * TRANS_NUM + 1];
  QuantizedValueProver<FieldT, Int> sub[2 * TRANS_NUM + 1];
  QuantizedValueProver<FieldT, Int> norm[2 * TRANS_NUM + 1];
  QuantizedValueProver<FieldT, Int> layer_norm_out[2 * TRANS_NUM + 1];
  // linear
  QuantizedValueProver<FieldT, Int> linear_out[TRANS_NUM * 4];
  // attention
  QuantizedValueProver<FieldT, Int> q[TRANS_NUM * N_HEAD];
  QuantizedValueProver<FieldT, Int> k[TRANS_NUM * N_HEAD];
  QuantizedValueProver<FieldT, Int> v[TRANS_NUM * N_HEAD];
  QuantizedValueProver<FieldT, Int> qk[TRANS_NUM * N_HEAD];
  QuantizedValueProver<FieldT, Int> divqk[TRANS_NUM * N_HEAD];
  QuantizedValueProver<FieldT, Int> qkv[TRANS_NUM * N_HEAD];
  // mha
  QuantizedValueProver<FieldT, Int> mha_out[TRANS_NUM];  // stack
  // res
  QuantizedValueProver<FieldT, Int> res_out[TRANS_NUM * 2];
  QuantizedValueProver<FieldT, Int> res_out1[TRANS_NUM * 2];
  QuantizedValueProver<FieldT, Int> res_out2[TRANS_NUM * 2];

  GPT2PrvData(const char *dir) {
    char path[PATH_BUF_LEN];
    // embd_out
    sprintf(path, "%s/embd_out", dir);
    LOAD_DATA_PATH(path, embd_out);
    // exp
    LOAD_DATA_ARR(dir, i_exp, z, z);
    LOAD_DATA_ARR(dir, i_exp, p, p);
    LOAD_DATA_ARR(dir, i_exp, p1, p1);
    LOAD_DATA_ARR(dir, i_exp, p2, p2);
    LOAD_DATA_ARR(dir, i_exp, l, l);
    LOAD_DATA_ARR(dir, i_exp, x1, exp_poly_out1);
    LOAD_DATA_ARR(dir, i_exp, x2, exp_poly_out2);
    FOR_EACH(exp_poly_out3, {
      long t;
      read_number(FULL_PATH(dir, i_exp, x3, i), t);
      exp_poly_out3[i].shape.push_back(1);
      exp_poly_out3[i].shape.push_back(1);
      exp_poly_out3[i].value.push_back(FieldT(t));
    })
    LOAD_DATA_ARR(dir, i_exp, out, exp_out);
    // softmax
    LOAD_DATA_ARR(dir, i_softmax, x_max, x_max);
    LOAD_DATA_ARR(dir, i_softmax, out, softmax_out);
    LOAD_DATA_ARR(dir, i_softmax, out1, softmax_out1);
    // gelu
    LOAD_DATA_ARR(dir, i_erf, sign, sign);
    LOAD_DATA_ARR(dir, i_erf, q_abs, q_abs);
    FOR_EACH(zb, {
      long t;
      read_number(FULL_PATH(dir, i_erf, zb, i), t);
      zb[i].shape.push_back(1);
      zb[i].shape.push_back(1);
      zb[i].value.push_back(FieldT(t));
    })
    LOAD_DATA_ARR(dir, i_erf, q_min, q_min);
    LOAD_DATA_ARR(dir, i_erf, l, erf_l);
    LOAD_DATA_ARR(dir, i_erf, x1, gelu_poly_out1);
    LOAD_DATA_ARR(dir, i_erf, x2, gelu_poly_out2);
    FOR_EACH(gelu_poly_out3, {
      long t;
      read_number(FULL_PATH(dir, i_erf, x3, i), t);
      gelu_poly_out3[i].shape.push_back(1);
      gelu_poly_out3[i].shape.push_back(1);
      gelu_poly_out3[i].value.push_back(FieldT(t));
    })
    LOAD_DATA_ARR(dir, i_erf, out, erf_out);
    LOAD_DATA_ARR(dir, gelu, out, gelu_out);
    LOAD_DATA_ARR(dir, gelu, out1, gelu_out1);
    LOAD_DATA_ARR(dir, gelu, out2, gelu_out2);
    // layer normalization
    LOAD_DATA_ARR(dir, layer_norm, mean, mean);
    LOAD_DATA_ARR(dir, layer_norm, var, var);
    LOAD_DATA_ARR(dir, layer_norm, std, std);
    LOAD_DATA_ARR(dir, layer_norm, std_max, std_max);
    LOAD_DATA_ARR(dir, layer_norm, sub, sub);
    LOAD_DATA_ARR(dir, layer_norm, norm, norm);
    LOAD_DATA_ARR(dir, layer_norm, out, layer_norm_out);
    // linear
    LOAD_DATA_ARR(dir, linear, out, linear_out);
    // attention
    LOAD_DATA_ARR(dir, attention, q, q);
    LOAD_DATA_ARR(dir, attention, k, k);
    LOAD_DATA_ARR(dir, attention, v, v);
    LOAD_DATA_ARR(dir, attention, qk, qk);
    LOAD_DATA_ARR(dir, attention, divqk, divqk);
    LOAD_DATA_ARR(dir, attention, qkv, qkv);
    // mha
    LOAD_DATA_ARR(dir, mha, mha, mha_out);
    // res
    LOAD_DATA_ARR(dir, res, out, res_out);
    LOAD_DATA_ARR(dir, res, out1, res_out1);
    LOAD_DATA_ARR(dir, res, out2, res_out2);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    embd_out.auth(ostriple);
    // exp
    FOR_EACH_DO(z, auth(ostriple));
    FOR_EACH_DO(p, auth(ostriple));
    FOR_EACH_DO(p1, auth(ostriple));
    FOR_EACH_DO(p2, auth(ostriple));
    FOR_EACH_DO(l, auth(ostriple));
    FOR_EACH_DO(exp_poly_out1, auth(ostriple));
    FOR_EACH_DO(exp_poly_out2, auth(ostriple));
    FOR_EACH_DO(exp_poly_out3, auth(ostriple));
    FOR_EACH_DO(exp_out, auth(ostriple));
    // softmax
    FOR_EACH_DO(x_max, auth(ostriple));
    FOR_EACH_DO(softmax_out, auth(ostriple));
    FOR_EACH_DO(softmax_out1, auth(ostriple));
    // gelu
    FOR_EACH_DO(sign, auth(ostriple));
    FOR_EACH_DO(q_abs, auth(ostriple));
    FOR_EACH_DO(zb, auth(ostriple));
    FOR_EACH_DO(q_min, auth(ostriple));
    FOR_EACH_DO(erf_l, auth(ostriple));
    FOR_EACH_DO(gelu_poly_out1, auth(ostriple));
    FOR_EACH_DO(gelu_poly_out2, auth(ostriple));
    FOR_EACH_DO(gelu_poly_out3, auth(ostriple));
    FOR_EACH_DO(erf_out, auth(ostriple));
    FOR_EACH_DO(gelu_out, auth(ostriple));
    FOR_EACH_DO(gelu_out1, auth(ostriple));
    FOR_EACH_DO(gelu_out2, auth(ostriple));
    // layer normalization
    FOR_EACH_DO(mean, auth(ostriple));
    FOR_EACH_DO(var, auth(ostriple));
    FOR_EACH_DO(std, auth(ostriple));
    FOR_EACH_DO(std_max, auth(ostriple));
    FOR_EACH_DO(sub, auth(ostriple));
    FOR_EACH_DO(norm, auth(ostriple));
    FOR_EACH_DO(layer_norm_out, auth(ostriple));
    // linear
    FOR_EACH_DO(linear_out, auth(ostriple));
    // attention
    FOR_EACH_DO(q, auth(ostriple));
    FOR_EACH_DO(k, auth(ostriple));
    FOR_EACH_DO(v, auth(ostriple));
    FOR_EACH_DO(qk, auth(ostriple));
    FOR_EACH_DO(divqk, auth(ostriple));
    FOR_EACH_DO(qkv, auth(ostriple));
    // mha
    FOR_EACH_DO(mha_out, auth(ostriple));
    // res
    FOR_EACH_DO(res_out, auth(ostriple));
    FOR_EACH_DO(res_out1, auth(ostriple));
    FOR_EACH_DO(res_out2, auth(ostriple));
  }
};

template <typename FieldT>
class ResPrvModel {
 public:
  NormFPProver<FieldT> fp1, fp2;

  ResPrvModel() {}
  ResPrvModel(const char *dir, int i) {
    float f1, f2;
    LOAD_N(dir, res, f1, i, f1);
    LOAD_N(dir, res, f2, i, f2);
    fp1 = NormFPProver<FieldT>(f1, false);
    fp2 = NormFPProver<FieldT>(f2, false);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    fp1.auth(ostriple);
    fp2.auth(ostriple);
  }
};

template <typename FieldT>
class SftMaxPrvModel {
 public:
  NormFPProver<FieldT> fp1, fp2, fp3, fp4, fp5, fp6, fp7;

  SftMaxPrvModel() {}
  SftMaxPrvModel(const char *dir, unsigned i) {
    float f1, f2, f3, f4, f5, f6, f7;
    LOAD_N(dir, i_exp, z_f, i, f1);
    LOAD_N(dir, i_exp, p1_f, i, f2);
    LOAD_N(dir, i_exp, p2_f, i, f3);
    LOAD_N(dir, i_exp, f1, i, f4);
    LOAD_N(dir, i_exp, f2, i, f5);
    LOAD_N(dir, i_exp, f3, i, f6);
    LOAD_N(dir, i_softmax, out1_f, i, f7);
    fp1 = NormFPProver<FieldT>(f1, false);
    fp2 = NormFPProver<FieldT>(f2, false);
    fp3 = NormFPProver<FieldT>(f3, false);
    fp4 = NormFPProver<FieldT>(f4, false);
    fp5 = NormFPProver<FieldT>(f5, false);
    fp6 = NormFPProver<FieldT>(f6, false);
    fp7 = NormFPProver<FieldT>(f7, false);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    fp1.auth(ostriple);
    fp2.auth(ostriple);
    fp3.auth(ostriple);
    fp4.auth(ostriple);
    fp5.auth(ostriple);
    fp6.auth(ostriple);
    fp7.auth(ostriple);
  }
};

template <typename FieldT>
class GeluPrvModel {
 public:
  NormFPProver<FieldT> fp1, fp2, fp3, fp4, fp5, fp6;

  GeluPrvModel() {}
  GeluPrvModel(const char *dir, unsigned i) {
    float f1, f2, f3, f4, f5, f6;
    LOAD_N(dir, i_erf, zb_f, i, f1);
    LOAD_N(dir, i_erf, f1, i, f2);
    LOAD_N(dir, i_erf, f2, i, f3);
    LOAD_N(dir, i_erf, f3, i, f4);
    LOAD_N(dir, gelu, f1, i, f5);
    LOAD_N(dir, gelu, f2, i, f6);
    fp1 = NormFPProver<FieldT>(f1, false);
    fp2 = NormFPProver<FieldT>(f2, false);
    fp3 = NormFPProver<FieldT>(f3, false);
    fp4 = NormFPProver<FieldT>(f4, false);
    fp5 = NormFPProver<FieldT>(f5, false);
    fp6 = NormFPProver<FieldT>(f6, false);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    fp1.auth(ostriple);
    fp2.auth(ostriple);
    fp3.auth(ostriple);
    fp4.auth(ostriple);
    fp5.auth(ostriple);
    fp6.auth(ostriple);
  }
};

template <typename FieldT, typename Int>
class LayerNormPrvModel {
 public:
  QuantizedValueProver<FieldT, Int> weight, bias;
  NormFPProver<FieldT> fp1, fp2;

  LayerNormPrvModel() {}
  LayerNormPrvModel(const char *dir, int i)
      : weight(FULL_PATH(dir, layer_norm, g, i), READ_FUNC),
        bias(FULL_PATH(dir, layer_norm, b, i), READ_FUNC) {
    float f1, f2;
    LOAD_N(dir, layer_norm, norm, i, f1);
    LOAD_N(dir, layer_norm, out_f, i, f2);
    fp1 = NormFPProver<FieldT>(f1, false);  // requant
    fp2 = NormFPProver<FieldT>(f2, false);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    weight.auth(ostriple);
    bias.auth(ostriple);
    fp1.auth(ostriple);
    fp2.auth(ostriple);
  }
};

template <typename FieldT, typename Int>
class LinearPrvModel {
 public:
  QuantizedValueProver<FieldT, Int> weight;
  QuantizedValueProver<FieldT, Int> bias;
  NormFPProver<FieldT> fp;

  LinearPrvModel() {}
  LinearPrvModel(const char *dir, unsigned i)
      : weight(FULL_PATH(dir, linear, w, i), READ_FUNC),
        bias(FULL_PATH(dir, linear, b, i), READ_FUNC) {
    float f;
    LOAD_N(dir, linear, f, i, f);
    fp = NormFPProver<FieldT>(f, false);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    weight.auth(ostriple);
    bias.auth(ostriple);
    fp.auth(ostriple);
  }
};

template <typename FieldT, typename Int>
class FFNPrvModel {
 public:
  LinearPrvModel<FieldT, Int> linear[2];
  GeluPrvModel<FieldT> gelu;

  FFNPrvModel() {}
  FFNPrvModel(const char *dir, int i)
      : linear{LinearPrvModel<FieldT, Int>(dir, 4 * i + 2),
               LinearPrvModel<FieldT, Int>(dir, 4 * i + 3)},
        gelu(dir, i) {}

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    FOR_EACH_DO(linear, auth(ostriple));
    gelu.auth(ostriple);
  }
};

template <typename FieldT>
class AttnPrvModel {
 public:
  // q @ k
  // requantization for division
  // softmax_out @ v
  NormFPProver<FieldT> fp1, fp2, fp3;
  SftMaxPrvModel<FieldT> softmax;

  AttnPrvModel() {}
  AttnPrvModel(const char *dir, int i) : softmax(dir, i) {
    float f1, f2, f3;
    LOAD_N(dir, attention, qk_f, i, f1);
    LOAD_N(dir, attention, divqk_f, i, f2);
    LOAD_N(dir, attention, qkv_f, i, f3);
    fp1 = NormFPProver<FieldT>(f1, false);
    fp2 = NormFPProver<FieldT>(f2, false);
    fp3 = NormFPProver<FieldT>(f3, false);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    fp1.auth(ostriple);
    fp2.auth(ostriple);
    fp3.auth(ostriple);
    softmax.auth(ostriple);
  }
};

template <typename FieldT, typename Int>
class MHAPrvModel {
 public:
  LinearPrvModel<FieldT, Int> linear[2];
  AttnPrvModel<FieldT> attn[N_HEAD];

  MHAPrvModel() {}
  MHAPrvModel(const char *dir, int j)
      : linear{LinearPrvModel<FieldT, Int>(dir, 4 * j),
               LinearPrvModel<FieldT, Int>(dir, 4 * j + 1)} {
    FOR_EACH_ASSIGN(attn, (AttnPrvModel<FieldT>(dir, N_HEAD * j + i)));
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    FOR_EACH_DO(linear, auth(ostriple));
    FOR_EACH_DO(attn, auth(ostriple));
  }
};

template <typename FieldT, typename Int>
class TransPrvModel {
 public:
  MHAPrvModel<FieldT, Int> mha;
  FFNPrvModel<FieldT, Int> ffn;
  LayerNormPrvModel<FieldT, Int> layer_norm[2];
  ResPrvModel<FieldT> res[2];

  TransPrvModel() {}
  TransPrvModel(const char *dir, unsigned i)
      : mha(dir, i),
        ffn(dir, i),
        layer_norm{LayerNormPrvModel<FieldT, Int>(dir, 2 * i),
                   LayerNormPrvModel<FieldT, Int>(dir, 2 * i + 1)},
        res{ResPrvModel<FieldT>(dir, 2 * i),
            ResPrvModel<FieldT>(dir, 2 * i + 1)} {}

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    mha.auth(ostriple);
    ffn.auth(ostriple);
    FOR_EACH_DO(layer_norm, auth(ostriple));
    FOR_EACH_DO(res, auth(ostriple));
  }
};

template <typename FieldT, typename Int>
class GPT2PrvModel {
 public:
  TransPrvModel<FieldT, Int> trans[TRANS_NUM];
  LayerNormPrvModel<FieldT, Int> layer_norm;

  GPT2PrvModel(const char *dir) : layer_norm(dir, 2 * TRANS_NUM) {
    FOR_EACH_ASSIGN(trans, (TransPrvModel<FieldT, Int>(dir, i)));
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    FOR_EACH_DO(trans, auth(ostriple));
    layer_norm.auth(ostriple);
  }
};

#undef READ_FUNC
#undef LOAD_DATA_PATH
#undef LOAD_DATA
#undef LOAD_DATA_ARR

#endif  // __TRANSFORMER_PROVER_H__