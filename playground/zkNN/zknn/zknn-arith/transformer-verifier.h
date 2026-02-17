#ifndef ZK_TRANSFORMER_VERIFIER_H
#define ZK_TRANSFORMER_VERIFIER_H

#include <fstream>

#include "quantized-value.h"
#include "transformer-common.h"

#define READ_FUNC transformer::read_shape<FieldT>
#define LOAD_DATA_PATH(path, out) \
  out = QuantizedValueVerifier<FieldT>(path, READ_FUNC)
#define LOAD_DATA(dir, layer, variable, cnt, out) \
  LOAD_DATA_PATH(FULL_PATH(dir, layer, variable, cnt), out)
#define LOAD_DATA_ARR(dir, layer, variable, arr) \
  FOR_EACH(arr, LOAD_DATA(dir, layer, variable, i, arr[i]))

namespace transformer {
template <typename FieldT>
static size_t read_shape(char *filename, QuantizedValueVerifier<FieldT> *x) {
  std::fstream file(filename, std::ios_base::in);
  if (!file.is_open()) {
    std::perror(filename);
    exit(1);
  }

  // unneeded content
  float scale;
  size_t ndim;
  file >> scale >> ndim;
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

  file.close();
  return sz;
}
}  // namespace transformer

template <typename FieldT>
class GPT2VerData {
 public:
  // input
  QuantizedValueVerifier<FieldT> input;
  // embedding output
  QuantizedValueVerifier<FieldT> embd_out;
  // exp
  QuantizedValueVerifier<FieldT> z[TRANS_NUM * N_HEAD];
  QuantizedValueVerifier<FieldT> p[TRANS_NUM * N_HEAD];
  QuantizedValueVerifier<FieldT> p1[TRANS_NUM * N_HEAD];
  QuantizedValueVerifier<FieldT> p2[TRANS_NUM * N_HEAD];
  QuantizedValueVerifier<FieldT> l[TRANS_NUM * N_HEAD];
  QuantizedValueVerifier<FieldT> exp_poly_out1[TRANS_NUM * N_HEAD];
  QuantizedValueVerifier<FieldT> exp_poly_out2[TRANS_NUM * N_HEAD];
  QuantizedValueVerifier<FieldT> exp_poly_out3[TRANS_NUM * N_HEAD];
  QuantizedValueVerifier<FieldT> exp_out[TRANS_NUM * N_HEAD];
  // softmax
  QuantizedValueVerifier<FieldT> x_max[TRANS_NUM * N_HEAD];
  QuantizedValueVerifier<FieldT> softmax_out[TRANS_NUM * N_HEAD];
  QuantizedValueVerifier<FieldT> softmax_out1[TRANS_NUM * N_HEAD];
  // gelu
  QuantizedValueVerifier<FieldT> sign[TRANS_NUM];
  QuantizedValueVerifier<FieldT> q_abs[TRANS_NUM];
  QuantizedValueVerifier<FieldT> zb[TRANS_NUM];
  QuantizedValueVerifier<FieldT> q_min[TRANS_NUM];
  QuantizedValueVerifier<FieldT> erf_l[TRANS_NUM];
  QuantizedValueVerifier<FieldT> gelu_poly_out1[TRANS_NUM];
  QuantizedValueVerifier<FieldT> gelu_poly_out2[TRANS_NUM];
  QuantizedValueVerifier<FieldT> gelu_poly_out3[TRANS_NUM];
  QuantizedValueVerifier<FieldT> erf_out[TRANS_NUM];
  QuantizedValueVerifier<FieldT> gelu_out[TRANS_NUM];
  QuantizedValueVerifier<FieldT> gelu_out1[TRANS_NUM];
  QuantizedValueVerifier<FieldT> gelu_out2[TRANS_NUM];
  // layer normalization
  QuantizedValueVerifier<FieldT> mean[2 * TRANS_NUM + 1];
  QuantizedValueVerifier<FieldT> var[2 * TRANS_NUM + 1];
  QuantizedValueVerifier<FieldT> std[2 * TRANS_NUM + 1];
  QuantizedValueVerifier<FieldT> std_max[2 * TRANS_NUM + 1];
  QuantizedValueVerifier<FieldT> sub[2 * TRANS_NUM + 1];
  QuantizedValueVerifier<FieldT> norm[2 * TRANS_NUM + 1];
  QuantizedValueVerifier<FieldT> layer_norm_out[2 * TRANS_NUM + 1];
  // linear
  QuantizedValueVerifier<FieldT> linear_out[TRANS_NUM * 4];
  // attention
  QuantizedValueVerifier<FieldT> q[TRANS_NUM * N_HEAD];
  QuantizedValueVerifier<FieldT> k[TRANS_NUM * N_HEAD];
  QuantizedValueVerifier<FieldT> v[TRANS_NUM * N_HEAD];
  QuantizedValueVerifier<FieldT> qk[TRANS_NUM * N_HEAD];
  QuantizedValueVerifier<FieldT> divqk[TRANS_NUM * N_HEAD];
  QuantizedValueVerifier<FieldT> qkv[TRANS_NUM * N_HEAD];
  // mha
  QuantizedValueVerifier<FieldT> mha_out[TRANS_NUM];  // stack
  // res
  QuantizedValueVerifier<FieldT> res_out[TRANS_NUM * 2];
  QuantizedValueVerifier<FieldT> res_out1[TRANS_NUM * 2];
  QuantizedValueVerifier<FieldT> res_out2[TRANS_NUM * 2];
  // final output
  QuantizedValueVerifier<FieldT> final_out;

  GPT2VerData(const char *dir) {
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
      exp_poly_out3[i].key_value.resize(1);
      exp_poly_out3[i].shape.push_back(1);
      exp_poly_out3[i].shape.push_back(1);
    });
    LOAD_DATA_ARR(dir, i_exp, out, exp_out);
    // softmax
    LOAD_DATA_ARR(dir, i_softmax, x_max, x_max);
    LOAD_DATA_ARR(dir, i_softmax, out, softmax_out);
    LOAD_DATA_ARR(dir, i_softmax, out1, softmax_out1);
    // gelu
    LOAD_DATA_ARR(dir, i_erf, sign, sign);
    LOAD_DATA_ARR(dir, i_erf, q_abs, q_abs);
    FOR_EACH(zb, {
      zb[i].key_value.resize(1);
      zb[i].shape.push_back(1);
      zb[i].shape.push_back(1);
    });
    LOAD_DATA_ARR(dir, i_erf, q_min, q_min);
    LOAD_DATA_ARR(dir, i_erf, l, erf_l);
    LOAD_DATA_ARR(dir, i_erf, x1, gelu_poly_out1);
    LOAD_DATA_ARR(dir, i_erf, x2, gelu_poly_out2);
    FOR_EACH(gelu_poly_out3, {
      gelu_poly_out3[i].key_value.resize(1);
      gelu_poly_out3[i].shape.push_back(1);
      gelu_poly_out3[i].shape.push_back(1);
    });
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
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
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
class ResVerModel {
 public:
  NormFPVerifier<FieldT> fp1, fp2;

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    fp1.auth(ostriple);
    fp2.auth(ostriple);
  }
};

template <typename FieldT>
class SftMaxVerModel {
 public:
  NormFPVerifier<FieldT> fp1, fp2, fp3, fp4, fp5, fp6, fp7;

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
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
class GeluVerModel {
 public:
  NormFPVerifier<FieldT> fp1, fp2, fp3, fp4, fp5, fp6;

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    fp1.auth(ostriple);
    fp2.auth(ostriple);
    fp3.auth(ostriple);
    fp4.auth(ostriple);
    fp5.auth(ostriple);
    fp6.auth(ostriple);
  }
};

template <typename FieldT>
class LayerNormVerModel {
 public:
  QuantizedValueVerifier<FieldT> weight, bias;
  NormFPVerifier<FieldT> fp1, fp2;

  LayerNormVerModel() {}
  LayerNormVerModel(const char *dir, int i)
      : weight(FULL_PATH(dir, layer_norm, g, i), READ_FUNC),
        bias(FULL_PATH(dir, layer_norm, b, i), READ_FUNC) {}

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    weight.auth(ostriple);
    bias.auth(ostriple);
    fp1.auth(ostriple);
    fp2.auth(ostriple);
  }
};

template <typename FieldT>
class LinearVerModel {
 public:
  QuantizedValueVerifier<FieldT> weight, bias;
  NormFPVerifier<FieldT> fp;

  LinearVerModel() {}
  LinearVerModel(const char *dir, int i)
      : weight(FULL_PATH(dir, linear, w, i), READ_FUNC),
        bias(FULL_PATH(dir, linear, b, i), READ_FUNC) {}

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    weight.auth(ostriple);
    bias.auth(ostriple);
    fp.auth(ostriple);
  }
};

template <typename FieldT>
class FFNVerModel {
 public:
  LinearVerModel<FieldT> linear[2];
  GeluVerModel<FieldT> gelu;

  FFNVerModel() {}
  FFNVerModel(const char *dir, int i)
      : linear{LinearVerModel<FieldT>(dir, 4 * i + 2),
               LinearVerModel<FieldT>(dir, 4 * i + 3)} {}

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    FOR_EACH_DO(linear, auth(ostriple));
    gelu.auth(ostriple);
  }
};

template <typename FieldT>
class AttnVerModel {
 public:
  NormFPVerifier<FieldT> fp1, fp2, fp3;
  SftMaxVerModel<FieldT> softmax;

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    fp1.auth(ostriple);
    fp2.auth(ostriple);
    fp3.auth(ostriple);
    softmax.auth(ostriple);
  }
};

template <typename FieldT>
class MHAVerModel {
 public:
  LinearVerModel<FieldT> linear[2];
  AttnVerModel<FieldT> attn[N_HEAD];

  MHAVerModel() {}
  MHAVerModel(const char *dir, int i)
      : linear{LinearVerModel<FieldT>(dir, 4 * i),
               LinearVerModel<FieldT>(dir, 4 * i + 1)} {}

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    FOR_EACH_DO(linear, auth(ostriple));
    FOR_EACH_DO(attn, auth(ostriple));
  }
};

template <typename FieldT>
class TransVerModel {
 public:
  MHAVerModel<FieldT> mha;
  FFNVerModel<FieldT> ffn;
  LayerNormVerModel<FieldT> layer_norm[2];
  ResVerModel<FieldT> res[2];

  TransVerModel() {}
  TransVerModel(const char *dir, unsigned i)
      : mha(dir, i),
        ffn(dir, i),
        layer_norm{LayerNormVerModel<FieldT>(dir, 2 * i),
                   LayerNormVerModel<FieldT>(dir, 2 * i + 1)} {}

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    mha.auth(ostriple);
    ffn.auth(ostriple);
    FOR_EACH_DO(layer_norm, auth(ostriple));
    FOR_EACH_DO(res, auth(ostriple));
  }
};

template <typename FieldT>
class GPT2VerModel {
 public:
  TransVerModel<FieldT> trans[TRANS_NUM];
  LayerNormVerModel<FieldT> layer_norm;

  GPT2VerModel(const char *dir) : layer_norm(dir, 2 * TRANS_NUM) {
    FOR_EACH_ASSIGN(trans, (TransVerModel<FieldT>(dir, i)));
  }

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    FOR_EACH_DO(trans, auth(ostriple));
    layer_norm.auth(ostriple);
  }
};

#undef READ_FUNC
#undef LOAD_DATA_PATH
#undef LOAD_DATA
#undef LOAD_DATA_ARR

#endif  // ZK_TRANSFORMER_VERIFIER_H