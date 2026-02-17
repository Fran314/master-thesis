#ifndef __CNN_VERIFIER_H__
#define __CNN_VERIFIER_H__

#include "cnn-common.h"
#include "quantized-value.h"

#define READ_FUNC cnn::read_shape<FieldT>
#define LOAD_DATA_PATH(path, out) \
  out = QuantizedValueVerifier<FieldT>(path, READ_FUNC)
#define LOAD_DATA(dir, layer, variable, cnt, out) \
  LOAD_DATA_PATH(FULL_PATH(dir, layer, variable, cnt), out)
#define LOAD_DATA_ARR(dir, layer, variable, cnt, arr) \
  FOR_EACH(arr, LOAD_DATA(dir, layer, variable, i, arr[i]))

namespace cnn {
template <typename FieldT>
static size_t read_shape(char *filename, QuantizedValueVerifier<FieldT> *x) {
  // open file
  std::fstream file(filename, std::ios_base::in);
  if (!file.is_open()) {
    std::perror(filename);
    exit(1);
  }
  // unneeded content
  float in1, in2, out, sf;
  signed zero;
  size_t ndim;
  file >> in1 >> in2 >> out >> sf >> zero >> ndim;
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
}

template <typename FieldT>
class ConvVerModel {
 public:
  QuantizedValueVerifier<FieldT> weight, bias;
  NormFPVerifier<FieldT> fp;
  size_t padding, stride;

  ConvVerModel(){};
  ConvVerModel(const char *dir, int i)
      : weight(FULL_PATH(dir, conv, weight, i), READ_FUNC),
        bias(FULL_PATH(dir, conv, bias, i), READ_FUNC) {
    read_param(FULL_PATH(dir, conv, param, i), padding, stride);
  }

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    weight.auth(ostriple);
    bias.auth(ostriple);
    fp.auth(ostriple);
  }
};

template <typename FieldT>
class FCVerModel {
 public:
  QuantizedValueVerifier<FieldT> weight, bias;
  NormFPVerifier<FieldT> fp;

  FCVerModel(){};
  FCVerModel(const char *dir, int i)
      : weight(FULL_PATH(dir, conv, weight, i), READ_FUNC),
        bias(FULL_PATH(dir, conv, bias, i), READ_FUNC) {}

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    weight.auth(ostriple);
    bias.auth(ostriple);
    fp.auth(ostriple);
  }
};

template <typename FieldT>
class SkipAddVerModel {
 public:
  NormFPVerifier<FieldT> fp1, fp2;

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    fp1.auth(ostriple);
    fp2.auth(ostriple);
  }
};

template <typename FieldT>
class ResNet18VerModel {
 public:
  ConvVerModel<FieldT> conv_model[resnet18_convs];
  SkipAddVerModel<FieldT> skip_add_model[resnet18_skips];
  FCVerModel<FieldT> fc_model;

  ResNet18VerModel(const char *dir) {
    FOR_EACH_ASSIGN(conv_model, (ConvVerModel<FieldT>(dir, i)));
    fc_model = FCVerModel<FieldT>(dir, resnet18_fc_idx);
  }

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    FOR_EACH_DO(conv_model, auth(ostriple));
    FOR_EACH_DO(skip_add_model, auth(ostriple));
    fc_model.auth(ostriple);
  }
};

template <typename FieldT>
class ResNet18VerData {
 public:
  QuantizedValueVerifier<FieldT> input;
  QuantizedValueVerifier<FieldT> conv_output[resnet18_convs];
  QuantizedValueVerifier<FieldT> relu_output[resnet18_relus];
  QuantizedValueVerifier<FieldT> add_output[resnet18_skips];
  QuantizedValueVerifier<FieldT> add_output1[resnet18_skips];
  QuantizedValueVerifier<FieldT> add_output2[resnet18_skips];
  QuantizedValueVerifier<FieldT> max_output;
  QuantizedValueVerifier<FieldT> avg_output;
  QuantizedValueVerifier<FieldT> fc_output;

  ResNet18VerData(const char *dir) {
    LOAD_DATA_ARR(dir, conv, output, i, conv_output);
    LOAD_DATA_ARR(dir, relu, output, i, relu_output);
    LOAD_DATA_ARR(dir, skip_add, output, i, add_output);
    LOAD_DATA_ARR(dir, skip_add, output1, i, add_output1);
    LOAD_DATA_ARR(dir, skip_add, output2, i, add_output2);

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
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
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

template <typename FieldT>
class ResNet50VerModel {
 public:
  ConvVerModel<FieldT> conv_model[resnet50_convs];
  SkipAddVerModel<FieldT> skip_add_model[resnet50_skips];
  FCVerModel<FieldT> fc_model;

  ResNet50VerModel(const char *dir) {
    FOR_EACH_ASSIGN(conv_model, (ConvVerModel<FieldT>(dir, i)));
    fc_model = FCVerModel<FieldT>(dir, resnet50_fc_idx);
  }

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    FOR_EACH_DO(conv_model, auth(ostriple));
    FOR_EACH_DO(skip_add_model, auth(ostriple));
    fc_model.auth(ostriple);
  }
};

template <typename FieldT>
class ResNet50VerData {
 public:
  QuantizedValueVerifier<FieldT> input;
  QuantizedValueVerifier<FieldT> conv_output[resnet50_convs];
  QuantizedValueVerifier<FieldT> relu_output[resnet50_relus];
  QuantizedValueVerifier<FieldT> add_output[resnet50_skips];
  QuantizedValueVerifier<FieldT> add_output1[resnet50_skips];
  QuantizedValueVerifier<FieldT> add_output2[resnet50_skips];
  QuantizedValueVerifier<FieldT> max_output;
  QuantizedValueVerifier<FieldT> avg_output;
  QuantizedValueVerifier<FieldT> fc_output;

  ResNet50VerData(const char *dir) {
    LOAD_DATA_ARR(dir, conv, output, i, conv_output);
    LOAD_DATA_ARR(dir, relu, output, i, relu_output);
    LOAD_DATA_ARR(dir, skip_add, output, i, add_output);
    LOAD_DATA_ARR(dir, skip_add, output1, i, add_output1);
    LOAD_DATA_ARR(dir, skip_add, output2, i, add_output2);

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
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
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

template <typename FieldT>
class ResNet101VerModel {
 public:
  ConvVerModel<FieldT> conv_model[resnet101_convs];
  SkipAddVerModel<FieldT> skip_add_model[resnet101_skips];
  FCVerModel<FieldT> fc_model;

  ResNet101VerModel(const char *dir) {
    FOR_EACH_ASSIGN(conv_model, (ConvVerModel<FieldT>(dir, i)));
    fc_model = FCVerModel<FieldT>(dir, resnet101_fc_idx);
  }

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    FOR_EACH_DO(conv_model, auth(ostriple));
    FOR_EACH_DO(skip_add_model, auth(ostriple));
    fc_model.auth(ostriple);
  }
};

template <typename FieldT>
class ResNet101VerData {
 public:
  QuantizedValueVerifier<FieldT> input;
  QuantizedValueVerifier<FieldT> conv_output[resnet101_convs];
  QuantizedValueVerifier<FieldT> relu_output[resnet101_relus];
  QuantizedValueVerifier<FieldT> add_output[resnet101_skips];
  QuantizedValueVerifier<FieldT> add_output1[resnet101_skips];
  QuantizedValueVerifier<FieldT> add_output2[resnet101_skips];
  QuantizedValueVerifier<FieldT> max_output;
  QuantizedValueVerifier<FieldT> avg_output;
  QuantizedValueVerifier<FieldT> fc_output;

  ResNet101VerData(const char *dir) {
    LOAD_DATA_ARR(dir, conv, output, i, conv_output);
    LOAD_DATA_ARR(dir, relu, output, i, relu_output);
    LOAD_DATA_ARR(dir, skip_add, output, i, add_output);
    LOAD_DATA_ARR(dir, skip_add, output1, i, add_output1);
    LOAD_DATA_ARR(dir, skip_add, output2, i, add_output2);

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
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
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

template <typename FieldT>
class VGG11VerModel {
 public:
  ConvVerModel<FieldT> conv_model[vgg11_convs];
  FCVerModel<FieldT> fc_model[vgg11_fcs];

  VGG11VerModel(const char *dir) {
    FOR_EACH_ASSIGN(conv_model, (ConvVerModel<FieldT>(dir, i)));
    FOR_EACH_ASSIGN(fc_model, (FCVerModel<FieldT>(dir, vgg11_convs + i)));
  }

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
    FOR_EACH_DO(conv_model, auth(ostriple));
    FOR_EACH_DO(fc_model, auth(ostriple));
  }
};

template <typename FieldT>
class VGG11VerData {
 public:
  QuantizedValueVerifier<FieldT> input;
  QuantizedValueVerifier<FieldT> conv_output[vgg11_convs];
  QuantizedValueVerifier<FieldT> relu_output[vgg11_relus];
  QuantizedValueVerifier<FieldT> max_output[vgg11_maxs];
  QuantizedValueVerifier<FieldT> avg_output;
  QuantizedValueVerifier<FieldT> fc_output[vgg11_fcs];

  VGG11VerData(const char *dir) {
    LOAD_DATA_ARR(dir, conv, output, i, conv_output);
    LOAD_DATA_ARR(dir, relu, output, i, relu_output);
    LOAD_DATA_ARR(dir, maxpool, output, i, max_output);
    FOR_EACH(fc_output,
             LOAD_DATA(dir, conv, output, (vgg11_convs + i), fc_output[i]));

    char path[PATH_BUF_LEN];
    sprintf(path, "%s/input", dir);
    LOAD_DATA_PATH(path, input);

    sprintf(path, "%s/avg_output", dir);
    LOAD_DATA_PATH(path, avg_output);
  }

  template <typename IO>
  void auth(FpOSTripleVer<IO, FieldT> *ostriple) {
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

#endif  // __CNN_VERIFIER_H__