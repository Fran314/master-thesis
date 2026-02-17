#ifndef __CNN_PPOVER_H__
#define __CNN_PPOVER_H__

#include "cnn-common.h"
#include "quantized-value.h"

#define READ_FUNC cnn::read_vec<FieldT, Int>
#define LOAD_DATA_PATH(path, out) \
  out = QuantizedValueProver<FieldT, Int>(path, READ_FUNC)
#define LOAD_DATA(dir, layer, variable, cnt, out) \
  LOAD_DATA_PATH(FULL_PATH(dir, layer, variable, cnt), out)
#define LOAD_DATA_ARR(dir, layer, variable, arr) \
  FOR_EACH(arr, LOAD_DATA(dir, layer, variable, i, arr[i]))

namespace cnn {

template <typename FieldT, typename Int>
size_t read_vec(char *filename, QuantizedValueProver<FieldT, Int> *x) {
  // open file
  std::fstream file(filename, std::ios_base::in);
  if (!file.is_open()) {
    std::perror(filename);
    exit(1);
  }

  // fscale, iz
  float in1, in2, out;
  size_t ndim;
  file >> in1 >> in2 >> out >> x->fscale >> x->iz >> ndim;
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
  // ivalue
  x->ivalue.resize(sz);
  for (size_t i = 0; i < sz; i++) {
    Int v;
    file >> v;
    v -= x->iz;
    x->ivalue[i] = v;
  }

  file.close();
  return sz;
}

}  // namespace cnn

inline void read_sf(char *filename, float &in1, float &in2, float &out) {
  std::fstream file(filename, std::ios_base::in);
  if (!file.is_open()) {
    std::perror(filename);
    exit(1);
  }
  file >> in1 >> in2 >> out;
  file.close();
}

template <typename FieldT, typename Int>
class ConvPrvModel {
 public:
  QuantizedValueProver<FieldT, Int> weight, bias;
  NormFPProver<FieldT> fp;
  size_t padding, stride;

  ConvPrvModel(){};
  ConvPrvModel(const char *dir, int i)
      : weight(FULL_PATH(dir, conv, weight, i), READ_FUNC),
        bias(FULL_PATH(dir, conv, bias, i), READ_FUNC) {
    read_param(FULL_PATH(dir, conv, param, i), padding, stride);

    float in1, in2, out;
    read_sf(FULL_PATH(dir, conv, weight, i), in1, in2, out);
    fp = NormFPProver<FieldT>((in1 * weight.fscale) / out);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    weight.auth(ostriple);
    bias.auth(ostriple);
    fp.auth(ostriple);
  }
};

template <typename FieldT, typename Int>
class FCPrvModel {
 public:
  QuantizedValueProver<FieldT, Int> weight, bias;
  NormFPProver<FieldT> fp;

  FCPrvModel() {}
  FCPrvModel(const char *dir, int i)
      : weight(FULL_PATH(dir, conv, weight, i), READ_FUNC),
        bias(FULL_PATH(dir, conv, bias, i), READ_FUNC) {
    float in1, in2, out;
    read_sf(FULL_PATH(dir, conv, weight, i), in1, in2, out);
    fp = NormFPProver<FieldT>((in1 * weight.fscale) / out);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    weight.auth(ostriple);
    bias.auth(ostriple);
    fp.auth(ostriple);
  }
};

template <typename FieldT>
class SkipAddPrvModel {
 public:
  NormFPProver<FieldT> fp1, fp2;

  SkipAddPrvModel() {}
  SkipAddPrvModel(const char *dir, int i) {
    float in1, in2, out;
    read_sf(FULL_PATH(dir, skip_add, output, i), in1, in2, out);
    fp1 = NormFPProver<FieldT>(in1 / out);
    fp2 = NormFPProver<FieldT>(in2 / out);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    fp1.auth(ostriple);
    fp2.auth(ostriple);
  }
};

template <typename FieldT, typename Int>
class ResNet18PrvModel {
 public:
  ConvPrvModel<FieldT, Int> conv_model[resnet18_convs];
  SkipAddPrvModel<FieldT> skip_add_model[resnet18_skips];
  FCPrvModel<FieldT, Int> fc_model;

  ResNet18PrvModel(const char *dir) {
    FOR_EACH_ASSIGN(conv_model, (ConvPrvModel<FieldT, Int>(dir, i)));
    FOR_EACH_ASSIGN(skip_add_model, (SkipAddPrvModel<FieldT>(dir, i)));
    fc_model = FCPrvModel<FieldT, Int>(dir, resnet18_fc_idx);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    FOR_EACH_DO(conv_model, auth(ostriple));
    FOR_EACH_DO(skip_add_model, auth(ostriple));
    fc_model.auth(ostriple);
  }
};

template <typename FieldT, typename Int>
class ResNet18PrvData {
 public:
  QuantizedValueProver<FieldT, Int> input;
  QuantizedValueProver<FieldT, Int> conv_output[resnet18_convs];
  QuantizedValueProver<FieldT, Int> relu_output[resnet18_relus];
  QuantizedValueProver<FieldT, Int> add_output[resnet18_skips];
  QuantizedValueProver<FieldT, Int> add_output1[resnet18_skips];
  QuantizedValueProver<FieldT, Int> add_output2[resnet18_skips];
  QuantizedValueProver<FieldT, Int> max_output;
  QuantizedValueProver<FieldT, Int> avg_output;
  QuantizedValueProver<FieldT, Int> fc_output;

  ResNet18PrvData(const char *dir) {
    LOAD_DATA_ARR(dir, conv, output, conv_output);
    LOAD_DATA_ARR(dir, relu, output, relu_output);
    LOAD_DATA_ARR(dir, skip_add, output, add_output);
    LOAD_DATA_ARR(dir, skip_add, output1, add_output1);
    LOAD_DATA_ARR(dir, skip_add, output2, add_output2);

    char path[PATH_BUF_LEN];
    sprintf(path, "%s/input", dir);
    LOAD_DATA_PATH(path, input);

    sprintf(path, "%s/maxpool_output_0", dir);
    LOAD_DATA_PATH(path, max_output);

    sprintf(path, "%s/avg_output", dir);
    LOAD_DATA_PATH(path, avg_output);

    LOAD_DATA(dir, conv, output, resnet18_fc_idx, fc_output);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    input.auth(ostriple);
    FOR_EACH_DO(conv_output, auth(ostriple));
    FOR_EACH_DO(relu_output, auth(ostriple));
    FOR_EACH_DO(add_output, auth(ostriple));
    FOR_EACH_DO(add_output1, auth(ostriple));
    FOR_EACH_DO(add_output2, auth(ostriple));
    max_output.auth(ostriple);
    avg_output.auth(ostriple);
    fc_output.auth(ostriple);
  }
};

template <typename FieldT, typename Int>
class ResNet50PrvModel {
 public:
  ConvPrvModel<FieldT, Int> conv_model[resnet50_convs];
  SkipAddPrvModel<FieldT> skip_add_model[resnet50_skips];
  FCPrvModel<FieldT, Int> fc_model;

  ResNet50PrvModel(const char *dir) {
    FOR_EACH_ASSIGN(conv_model, (ConvPrvModel<FieldT, Int>(dir, i)));
    FOR_EACH_ASSIGN(skip_add_model, (SkipAddPrvModel<FieldT>(dir, i)));
    fc_model = FCPrvModel<FieldT, Int>(dir, resnet50_fc_idx);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    FOR_EACH_DO(conv_model, auth(ostriple));
    FOR_EACH_DO(skip_add_model, auth(ostriple));
    fc_model.auth(ostriple);
  }
};

template <typename FieldT, typename Int>
class ResNet50PrvData {
 public:
  QuantizedValueProver<FieldT, Int> input;
  QuantizedValueProver<FieldT, Int> conv_output[resnet50_convs];
  QuantizedValueProver<FieldT, Int> relu_output[resnet50_relus];
  QuantizedValueProver<FieldT, Int> add_output[resnet50_skips];
  QuantizedValueProver<FieldT, Int> add_output1[resnet50_skips];
  QuantizedValueProver<FieldT, Int> add_output2[resnet50_skips];
  QuantizedValueProver<FieldT, Int> max_output;
  QuantizedValueProver<FieldT, Int> avg_output;
  QuantizedValueProver<FieldT, Int> fc_output;

  ResNet50PrvData(const char *dir) {
    LOAD_DATA_ARR(dir, conv, output, conv_output);
    LOAD_DATA_ARR(dir, relu, output, relu_output);
    LOAD_DATA_ARR(dir, skip_add, output, add_output);
    LOAD_DATA_ARR(dir, skip_add, output1, add_output1);
    LOAD_DATA_ARR(dir, skip_add, output2, add_output2);

    char path[PATH_BUF_LEN];
    sprintf(path, "%s/input", dir);
    LOAD_DATA_PATH(path, input);

    sprintf(path, "%s/maxpool_output_0", dir);
    LOAD_DATA_PATH(path, max_output);

    sprintf(path, "%s/avg_output", dir);
    LOAD_DATA_PATH(path, avg_output);

    LOAD_DATA(dir, conv, output, resnet50_fc_idx, fc_output);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    input.auth(ostriple);
    FOR_EACH_DO(conv_output, auth(ostriple));
    FOR_EACH_DO(relu_output, auth(ostriple));
    FOR_EACH_DO(add_output, auth(ostriple));
    FOR_EACH_DO(add_output1, auth(ostriple));
    FOR_EACH_DO(add_output2, auth(ostriple));
    max_output.auth(ostriple);
    avg_output.auth(ostriple);
    fc_output.auth(ostriple);
  }
};

template <typename FieldT, typename Int>
class ResNet101PrvModel {
 public:
  ConvPrvModel<FieldT, Int> conv_model[resnet101_convs];
  SkipAddPrvModel<FieldT> skip_add_model[resnet101_skips];
  FCPrvModel<FieldT, Int> fc_model;

  ResNet101PrvModel(const char *dir) {
    FOR_EACH_ASSIGN(conv_model, (ConvPrvModel<FieldT, Int>(dir, i)));
    FOR_EACH_ASSIGN(skip_add_model, (SkipAddPrvModel<FieldT>(dir, i)));
    fc_model = FCPrvModel<FieldT, Int>(dir, resnet101_fc_idx);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    FOR_EACH_DO(conv_model, auth(ostriple));
    FOR_EACH_DO(skip_add_model, auth(ostriple));
    fc_model.auth(ostriple);
  }
};

template <typename FieldT, typename Int>
class ResNet101PrvData {
 public:
  QuantizedValueProver<FieldT, Int> input;
  QuantizedValueProver<FieldT, Int> conv_output[resnet101_convs];
  QuantizedValueProver<FieldT, Int> relu_output[resnet101_relus];
  QuantizedValueProver<FieldT, Int> add_output[resnet101_skips];
  QuantizedValueProver<FieldT, Int> add_output1[resnet101_skips];
  QuantizedValueProver<FieldT, Int> add_output2[resnet101_skips];
  QuantizedValueProver<FieldT, Int> max_output;
  QuantizedValueProver<FieldT, Int> avg_output;
  QuantizedValueProver<FieldT, Int> fc_output;

  ResNet101PrvData(const char *dir) {
    LOAD_DATA_ARR(dir, conv, output, conv_output);
    LOAD_DATA_ARR(dir, relu, output, relu_output);
    LOAD_DATA_ARR(dir, skip_add, output, add_output);
    LOAD_DATA_ARR(dir, skip_add, output1, add_output1);
    LOAD_DATA_ARR(dir, skip_add, output2, add_output2);

    char path[PATH_BUF_LEN];
    sprintf(path, "%s/input", dir);
    LOAD_DATA_PATH(path, input);

    sprintf(path, "%s/maxpool_output_0", dir);
    LOAD_DATA_PATH(path, max_output);

    sprintf(path, "%s/avg_output", dir);
    LOAD_DATA_PATH(path, avg_output);

    LOAD_DATA(dir, conv, output, resnet101_fc_idx, fc_output);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    input.auth(ostriple);
    FOR_EACH_DO(conv_output, auth(ostriple));
    FOR_EACH_DO(relu_output, auth(ostriple));
    FOR_EACH_DO(add_output, auth(ostriple));
    FOR_EACH_DO(add_output1, auth(ostriple));
    FOR_EACH_DO(add_output2, auth(ostriple));
    max_output.auth(ostriple);
    avg_output.auth(ostriple);
    fc_output.auth(ostriple);
  }
};

template <typename FieldT, typename Int>
class VGG11PrvModel {
 public:
  ConvPrvModel<FieldT, Int> conv_model[vgg11_convs];
  FCPrvModel<FieldT, Int> fc_model[vgg11_fcs];

  VGG11PrvModel(const char *dir) {
    FOR_EACH_ASSIGN(conv_model, (ConvPrvModel<FieldT, Int>(dir, i)));
    FOR_EACH_ASSIGN(fc_model, (FCPrvModel<FieldT, Int>(dir, vgg11_convs + i)));
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    FOR_EACH_DO(conv_model, auth(ostriple));
    FOR_EACH_DO(fc_model, auth(ostriple));
  }
};

template <typename FieldT, typename Int>
class VGG11PrvData {
 public:
  QuantizedValueProver<FieldT, Int> input;
  QuantizedValueProver<FieldT, Int> conv_output[vgg11_convs];
  QuantizedValueProver<FieldT, Int> relu_output[vgg11_relus];
  QuantizedValueProver<FieldT, Int> max_output[vgg11_maxs];
  QuantizedValueProver<FieldT, Int> avg_output;
  QuantizedValueProver<FieldT, Int> fc_output[vgg11_fcs];

  VGG11PrvData(const char *dir) {
    LOAD_DATA_ARR(dir, conv, output, conv_output);
    LOAD_DATA_ARR(dir, relu, output, relu_output);
    LOAD_DATA_ARR(dir, maxpool, output, max_output);
    FOR_EACH(fc_output,
             LOAD_DATA(dir, conv, output, (vgg11_convs + i), fc_output[i]));

    char path[PATH_BUF_LEN];
    sprintf(path, "%s/input", dir);
    LOAD_DATA_PATH(path, input);

    sprintf(path, "%s/avg_output", dir);
    LOAD_DATA_PATH(path, avg_output);
  }

  template <typename IO>
  void auth(FpOSTriplePrv<IO, FieldT> *ostriple) {
    input.auth(ostriple);
    FOR_EACH_DO(conv_output, auth(ostriple));
    FOR_EACH_DO(relu_output, auth(ostriple));
    FOR_EACH_DO(max_output, auth(ostriple));
    FOR_EACH_DO(fc_output, auth(ostriple));
    avg_output.auth(ostriple);
  }
};

#undef READ_FUNC
#undef LOAD_DATA_PATH
#undef LOAD_DATA
#undef LOAD_DATA_ARR

#endif  // __CNN_PPOVER_H__