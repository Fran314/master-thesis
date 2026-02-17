#ifndef ZK_FP_EXECUTIION_VER_H__
#define ZK_FP_EXECUTIION_VER_H__

#include <vector>

#include "zknn/zknn-arith/cnn-verifier.h"
#include "zknn/zknn-arith/ostriple-verifier.h"
#include "zknn/zknn-arith/transformer-verifier.h"
using namespace std;

template <typename IO, typename FieldT>
class ZKFpExecVer {
 public:
  FpOSTripleVer<IO, FieldT> *ostriple;
  IO *io = nullptr;

  ZKFpExecVer(IO **ios, FpOSTripleVer<IO, FieldT> *ostriple) {
    this->io = ios[0];
    this->ostriple = ostriple;
    // this->ostriple = new FpOSTripleVer<IO, FieldT>(ios);
  }

  ~ZKFpExecVer() { delete ostriple; }

  template <typename Int>
  bool ResNet18(const ResNet18VerModel<FieldT> &model,
                const ResNet18VerData<FieldT> &data) {
    ostriple->template zkp_conv<Int>(model.conv_model[0], data.input,
                                     data.conv_output[0]);
    ostriple->template zkp_relu<Int>(data.conv_output[0], data.relu_output[0]);
    ostriple->template zkp_maxpool<Int>(data.relu_output[0], data.max_output);

    // layer 1
    ostriple->template zkp_conv<Int>(model.conv_model[1], data.max_output,
                                     data.conv_output[1]);
    ostriple->template zkp_relu<Int>(data.conv_output[1], data.relu_output[1]);
    ostriple->template zkp_conv<Int>(model.conv_model[2], data.relu_output[1],
                                     data.conv_output[2]);
    ostriple->template zkp_skip_add<Int>(
        model.skip_add_model[0], data.max_output, data.conv_output[2],
        data.add_output1[0], data.add_output2[0], data.add_output[0]);
    ostriple->template zkp_relu<Int>(data.add_output[0], data.relu_output[2]);

    ostriple->template zkp_conv<Int>(model.conv_model[3], data.relu_output[2],
                                     data.conv_output[3]);
    ostriple->template zkp_relu<Int>(data.conv_output[3], data.relu_output[3]);
    ostriple->template zkp_conv<Int>(model.conv_model[4], data.relu_output[3],
                                     data.conv_output[4]);
    ostriple->template zkp_skip_add<Int>(
        model.skip_add_model[1], data.relu_output[2], data.conv_output[4],
        data.add_output1[1], data.add_output2[1], data.add_output[1]);
    ostriple->template zkp_relu<Int>(data.add_output[1], data.relu_output[4]);

    // layer 2 ~ layer 4
    for (int j = 1; j < 4; j++) {
      ostriple->template zkp_conv<Int>(model.conv_model[5 * j],
                                       data.relu_output[4 * j],
                                       data.conv_output[5 * j]);
      ostriple->template zkp_conv<Int>(model.conv_model[5 * j + 1],
                                       data.relu_output[4 * j],
                                       data.conv_output[5 * j + 1]);
      ostriple->template zkp_relu<Int>(data.conv_output[5 * j + 1],
                                       data.relu_output[4 * j + 1]);
      ostriple->template zkp_conv<Int>(model.conv_model[5 * j + 2],
                                       data.relu_output[4 * j + 1],
                                       data.conv_output[5 * j + 2]);
      ostriple->template zkp_skip_add<Int>(
          model.skip_add_model[2 * j], data.conv_output[5 * j],
          data.conv_output[5 * j + 2], data.add_output1[2 * j],
          data.add_output2[2 * j], data.add_output[2 * j]);
      ostriple->template zkp_relu<Int>(data.add_output[2 * j],
                                       data.relu_output[4 * j + 2]);

      ostriple->template zkp_conv<Int>(model.conv_model[5 * j + 3],
                                       data.relu_output[4 * j + 2],
                                       data.conv_output[5 * j + 3]);
      ostriple->template zkp_relu<Int>(data.conv_output[5 * j + 3],
                                       data.relu_output[4 * j + 3]);
      ostriple->template zkp_conv<Int>(model.conv_model[5 * j + 4],
                                       data.relu_output[4 * j + 3],
                                       data.conv_output[5 * j + 4]);
      ostriple->template zkp_skip_add<Int>(
          model.skip_add_model[2 * j + 1], data.relu_output[4 * j + 2],
          data.conv_output[5 * j + 4], data.add_output1[2 * j + 1],
          data.add_output2[2 * j + 1], data.add_output[2 * j + 1]);
      ostriple->template zkp_relu<Int>(data.add_output[2 * j + 1],
                                       data.relu_output[4 * j + 4]);
    }

    ostriple->template zkp_fc<Int>(model.fc_model, data.avg_output,
                                   data.fc_output);

    ostriple->zkp_range_batch(FieldT((1ul << 26) + 1));
    bool ret = ostriple->batch_check(false, false, true);
    return ret;
  }

  template <typename Int>
  bool ResNet50(const ResNet50VerModel<FieldT> &model,
                const ResNet50VerData<FieldT> &data) {
    ostriple->template zkp_conv<Int>(model.conv_model[0], data.input,
                                     data.conv_output[0]);
    ostriple->template zkp_relu<Int>(data.conv_output[0], data.relu_output[0]);
    ostriple->template zkp_maxpool<Int>(data.relu_output[0], data.max_output);

    int conv_cnt = 1, relu_cnt = 1, skip_cnt = 0;

    int seq[] = {3, 4, 6, 3};
    for (int j = 0; j < ARRLEN(seq); j++) {
      for (int seq_id = 0; seq_id < seq[j]; seq_id++) {
        const QuantizedValueVerifier<FieldT> *skip_add_in = NULL;
        if (seq_id == 0) {
          ostriple->template zkp_conv<Int>(
              model.conv_model[conv_cnt],
              j == 0 ? data.max_output : data.relu_output[relu_cnt - 1],
              data.conv_output[conv_cnt]);
          conv_cnt++;
          skip_add_in = &(data.conv_output[conv_cnt - 1]);
        } else {
          skip_add_in = &(data.relu_output[relu_cnt - 1]);
        }
        ostriple->template zkp_conv<Int>(model.conv_model[conv_cnt],
                                         j == 0 && seq_id == 0
                                             ? data.max_output
                                             : data.relu_output[relu_cnt - 1],
                                         data.conv_output[conv_cnt]);
        conv_cnt++;
        ostriple->template zkp_relu<Int>(data.conv_output[conv_cnt - 1],
                                         data.relu_output[relu_cnt]);
        relu_cnt++;
        ostriple->template zkp_conv<Int>(model.conv_model[conv_cnt],
                                         data.relu_output[relu_cnt - 1],
                                         data.conv_output[conv_cnt]);
        conv_cnt++;
        ostriple->template zkp_relu<Int>(data.conv_output[conv_cnt - 1],
                                         data.relu_output[relu_cnt]);
        relu_cnt++;
        ostriple->template zkp_conv<Int>(model.conv_model[conv_cnt],
                                         data.relu_output[relu_cnt - 1],
                                         data.conv_output[conv_cnt]);
        conv_cnt++;
        ostriple->template zkp_skip_add<Int>(
            model.skip_add_model[skip_cnt], *skip_add_in,
            data.conv_output[conv_cnt - 1], data.add_output1[skip_cnt],
            data.add_output2[skip_cnt], data.add_output[skip_cnt]);
        skip_cnt++;
        ostriple->template zkp_relu<Int>(data.add_output[skip_cnt - 1],
                                         data.relu_output[relu_cnt]);
        relu_cnt++;
      }
    }
    ostriple->template zkp_fc<Int>(model.fc_model, data.avg_output,
                                   data.fc_output);
    ostriple->zkp_range_batch(FieldT((1ul << 26) + 1));
    bool ret = ostriple->batch_check(false, false, true);
    return ret;
  }

  template <typename Int>
  bool ResNet101(const ResNet101VerModel<FieldT> &model,
                 const ResNet101VerData<FieldT> &data) {
    ostriple->template zkp_conv<Int>(model.conv_model[0], data.input,
                                     data.conv_output[0]);
    ostriple->template zkp_relu<Int>(data.conv_output[0], data.relu_output[0]);
    ostriple->template zkp_maxpool<Int>(data.relu_output[0], data.max_output);

    int conv_cnt = 1, relu_cnt = 1, skip_cnt = 0;

    int seq[] = {3, 4, 23, 3};
    for (int j = 0; j < ARRLEN(seq); j++) {
      for (int seq_id = 0; seq_id < seq[j]; seq_id++) {
        const QuantizedValueVerifier<FieldT> *skip_add_in = NULL;
        if (seq_id == 0) {
          ostriple->template zkp_conv<Int>(
              model.conv_model[conv_cnt],
              j == 0 ? data.max_output : data.relu_output[relu_cnt - 1],
              data.conv_output[conv_cnt]);
          conv_cnt++;
          skip_add_in = &(data.conv_output[conv_cnt - 1]);
        } else {
          skip_add_in = &(data.relu_output[relu_cnt - 1]);
        }
        ostriple->template zkp_conv<Int>(model.conv_model[conv_cnt],
                                         j == 0 && seq_id == 0
                                             ? data.max_output
                                             : data.relu_output[relu_cnt - 1],
                                         data.conv_output[conv_cnt]);
        conv_cnt++;
        ostriple->template zkp_relu<Int>(data.conv_output[conv_cnt - 1],
                                         data.relu_output[relu_cnt]);
        relu_cnt++;
        ostriple->template zkp_conv<Int>(model.conv_model[conv_cnt],
                                         data.relu_output[relu_cnt - 1],
                                         data.conv_output[conv_cnt]);
        conv_cnt++;
        ostriple->template zkp_relu<Int>(data.conv_output[conv_cnt - 1],
                                         data.relu_output[relu_cnt]);
        relu_cnt++;
        ostriple->template zkp_conv<Int>(model.conv_model[conv_cnt],
                                         data.relu_output[relu_cnt - 1],
                                         data.conv_output[conv_cnt]);
        conv_cnt++;
        ostriple->template zkp_skip_add<Int>(
            model.skip_add_model[skip_cnt], *skip_add_in,
            data.conv_output[conv_cnt - 1], data.add_output1[skip_cnt],
            data.add_output2[skip_cnt], data.add_output[skip_cnt]);
        skip_cnt++;
        ostriple->template zkp_relu<Int>(data.add_output[skip_cnt - 1],
                                         data.relu_output[relu_cnt]);
        relu_cnt++;
      }
    }
    ostriple->template zkp_fc<Int>(model.fc_model, data.avg_output,
                                   data.fc_output);
    ostriple->zkp_range_batch(FieldT((1ul << 26) + 1));
    bool ret = ostriple->batch_check(false, false, true);
    return ret;
  }

  template <typename Int>
  bool VGG11(const VGG11VerModel<FieldT> &model,
             const VGG11VerData<FieldT> &data) {
    int conv_cnt = 0, relu_cnt = 0, maxpool_cnt = 0, linear_cnt = 0;
    int features[] = {
        0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2,
    };
    const QuantizedValueVerifier<FieldT> *conv_in = &data.input;
    for (int i = 0; i < ARRLEN(features); i++) {
      if (features[i] == 0) {
        ostriple->template zkp_conv<Int>(model.conv_model[conv_cnt], *conv_in,
                                         data.conv_output[conv_cnt]);
        conv_cnt++;
      } else if (features[i] == 1) {
        ostriple->template zkp_relu<Int>(data.conv_output[conv_cnt - 1],
                                         data.relu_output[relu_cnt]);
        conv_in = &data.relu_output[relu_cnt];
        relu_cnt++;
      } else if (features[i] == 2) {
        ostriple->template zkp_maxpool<Int>(data.relu_output[relu_cnt - 1],
                                            data.max_output[maxpool_cnt], 0, 2,
                                            2);
        conv_in = &data.max_output[maxpool_cnt];
        maxpool_cnt++;
      } else
        assert(0);
    }
    int classifier[] = {3, 1, 3, 1, 3};
    for (int i = 0; i < ARRLEN(classifier); i++) {
      if (classifier[i] == 1) {
        ostriple->template zkp_relu<Int>(data.fc_output[linear_cnt - 1],
                                         data.relu_output[relu_cnt]);
        relu_cnt++;
      } else if (classifier[i] == 3) {
        ostriple->template zkp_fc<Int>(
            model.fc_model[linear_cnt],
            i == 0 ? data.avg_output : data.relu_output[relu_cnt - 1],
            data.fc_output[linear_cnt]);
        linear_cnt++;
      }
    }

    ostriple->zkp_range_batch(FieldT((1ul << 26) + 1));
    bool ret = ostriple->batch_check(false, false, true);
    return ret;
  }

  template <typename Int>
  bool GPT2(const GPT2VerModel<FieldT> &model,
            GPT2VerData<FieldT> &data) {
    TIME_STATS_BEG
    ostriple->template zkp_trans<Int>(model.trans, data);
    ostriple->template zkp_layer_norm<Int>(
        model.layer_norm, data.res_out[23], data.mean[24], data.var[24],
        data.std[24], data.std_max[24], data.sub[24], data.norm[24],
        data.layer_norm_out[24]);

    ostriple->zkp_range_batch(FieldT((1ul << 26) + 1));
    bool ret = ostriple->batch_check(false, false, true);
    TIME_STATS_END
    return ret;
  }

#undef SIDE_DECL
#undef SIDE_PUSH_VAR
#undef SIDE_PUSH
#undef ZKP_POLY2
};
#endif
