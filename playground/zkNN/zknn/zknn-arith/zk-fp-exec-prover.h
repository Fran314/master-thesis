#ifndef ZK_FP_EXECUTIION_PRV_H__
#define ZK_FP_EXECUTIION_PRV_H__

#include <emp-tool/utils/block.h>

#include <vector>

#include "zknn/zknn-arith/cnn-prover.h"
#include "zknn/zknn-arith/ostriple-common.h"
#include "zknn/zknn-arith/ostriple-prover.h"
#include "zknn/zknn-arith/transformer-prover.h"
using namespace std;

template <typename IO, typename FieldT>
class ZKFpExecPrv {
 public:
  FpOSTriplePrv<IO, FieldT> *ostriple;
  IO *io = nullptr;

  ZKFpExecPrv(IO **ios, FpOSTriplePrv<IO, FieldT> *ostriple) {
    this->io = ios[0];
    this->ostriple = ostriple;
    // this->ostriple = new FpOSTriplePrv<IO, FieldT>(ios);
  }

  ~ZKFpExecPrv() { delete ostriple; }

  template <typename Int>
  void ResNet18(const ResNet18PrvModel<FieldT, Int> &model,
                const ResNet18PrvData<FieldT, Int> &data) {
    ostriple->zkp_conv(model.conv_model[0], data.input, data.conv_output[0]);
    ostriple->zkp_relu(data.conv_output[0], data.relu_output[0]);
    ostriple->zkp_maxpool(data.relu_output[0], data.max_output);

    // layer 1
    ostriple->zkp_conv(model.conv_model[1], data.max_output,
                       data.conv_output[1]);
    ostriple->zkp_relu(data.conv_output[1], data.relu_output[1]);
    ostriple->zkp_conv(model.conv_model[2], data.relu_output[1],
                       data.conv_output[2]);
    ostriple->zkp_skip_add(model.skip_add_model[0], data.max_output,
                           data.conv_output[2], data.add_output1[0],
                           data.add_output2[0], data.add_output[0]);
    ostriple->zkp_relu(data.add_output[0], data.relu_output[2]);

    ostriple->zkp_conv(model.conv_model[3], data.relu_output[2],
                       data.conv_output[3]);
    ostriple->zkp_relu(data.conv_output[3], data.relu_output[3]);
    ostriple->zkp_conv(model.conv_model[4], data.relu_output[3],
                       data.conv_output[4]);
    ostriple->zkp_skip_add(model.skip_add_model[1], data.relu_output[2],
                           data.conv_output[4], data.add_output1[1],
                           data.add_output2[1], data.add_output[1]);
    ostriple->zkp_relu(data.add_output[1], data.relu_output[4]);

    // layer 2 ~ layer 4
    for (int j = 1; j < 4; j++) {
      ostriple->zkp_conv(model.conv_model[5 * j], data.relu_output[4 * j],
                         data.conv_output[5 * j]);
      ostriple->zkp_conv(model.conv_model[5 * j + 1], data.relu_output[4 * j],
                         data.conv_output[5 * j + 1]);
      ostriple->zkp_relu(data.conv_output[5 * j + 1],
                         data.relu_output[4 * j + 1]);
      ostriple->zkp_conv(model.conv_model[5 * j + 2],
                         data.relu_output[4 * j + 1],
                         data.conv_output[5 * j + 2]);
      ostriple->zkp_skip_add(
          model.skip_add_model[2 * j], data.conv_output[5 * j],
          data.conv_output[5 * j + 2], data.add_output1[2 * j],
          data.add_output2[2 * j], data.add_output[2 * j]);
      ostriple->zkp_relu(data.add_output[2 * j], data.relu_output[4 * j + 2]);

      ostriple->zkp_conv(model.conv_model[5 * j + 3],
                         data.relu_output[4 * j + 2],
                         data.conv_output[5 * j + 3]);
      ostriple->zkp_relu(data.conv_output[5 * j + 3],
                         data.relu_output[4 * j + 3]);
      ostriple->zkp_conv(model.conv_model[5 * j + 4],
                         data.relu_output[4 * j + 3],
                         data.conv_output[5 * j + 4]);
      ostriple->zkp_skip_add(
          model.skip_add_model[2 * j + 1], data.relu_output[4 * j + 2],
          data.conv_output[5 * j + 4], data.add_output1[2 * j + 1],
          data.add_output2[2 * j + 1], data.add_output[2 * j + 1]);
      ostriple->zkp_relu(data.add_output[2 * j + 1],
                         data.relu_output[4 * j + 4]);
    }

    ostriple->zkp_fc(model.fc_model, data.avg_output, data.fc_output);

    ostriple->zkp_range_batch(FieldT((1ul << 26) + 1));
    ostriple->batch_check(false, false, true);
  }

  template <typename Int>
  void ResNet50(const ResNet50PrvModel<FieldT, Int> &model,
                const ResNet50PrvData<FieldT, Int> &data) {
    ostriple->zkp_conv(model.conv_model[0], data.input, data.conv_output[0]);
    ostriple->zkp_relu(data.conv_output[0], data.relu_output[0]);
    ostriple->zkp_maxpool(data.relu_output[0], data.max_output);

    int conv_cnt = 1, relu_cnt = 1, skip_cnt = 0;

    int seq[] = {3, 4, 6, 3};
    for (int j = 0; j < ARRLEN(seq); j++) {
      for (int seq_id = 0; seq_id < seq[j]; seq_id++) {
        const QuantizedValueProver<FieldT, Int> *skip_add_in = NULL;
        if (seq_id == 0) {
          ostriple->zkp_conv(
              model.conv_model[conv_cnt],
              j == 0 ? data.max_output : data.relu_output[relu_cnt - 1],
              data.conv_output[conv_cnt]);
          conv_cnt++;
          skip_add_in = &(data.conv_output[conv_cnt - 1]);
        } else {
          skip_add_in = &(data.relu_output[relu_cnt - 1]);
        }
        ostriple->zkp_conv(model.conv_model[conv_cnt],
                           j == 0 && seq_id == 0
                               ? data.max_output
                               : data.relu_output[relu_cnt - 1],
                           data.conv_output[conv_cnt]);
        conv_cnt++;
        ostriple->zkp_relu(data.conv_output[conv_cnt - 1],
                           data.relu_output[relu_cnt]);
        relu_cnt++;
        ostriple->zkp_conv(model.conv_model[conv_cnt],
                           data.relu_output[relu_cnt - 1],
                           data.conv_output[conv_cnt]);
        conv_cnt++;
        ostriple->zkp_relu(data.conv_output[conv_cnt - 1],
                           data.relu_output[relu_cnt]);
        relu_cnt++;
        ostriple->zkp_conv(model.conv_model[conv_cnt],
                           data.relu_output[relu_cnt - 1],
                           data.conv_output[conv_cnt]);
        conv_cnt++;
        ostriple->zkp_skip_add(
            model.skip_add_model[skip_cnt], *skip_add_in,
            data.conv_output[conv_cnt - 1], data.add_output1[skip_cnt],
            data.add_output2[skip_cnt], data.add_output[skip_cnt]);
        skip_cnt++;
        ostriple->zkp_relu(data.add_output[skip_cnt - 1],
                           data.relu_output[relu_cnt]);
        relu_cnt++;
      }
    }
    ostriple->zkp_fc(model.fc_model, data.avg_output, data.fc_output);
    ostriple->zkp_range_batch(FieldT((1ul << 26) + 1));
    ostriple->batch_check(false, false, true);
  }

  template <typename Int>
  void ResNet101(const ResNet101PrvModel<FieldT, Int> &model,
                 const ResNet101PrvData<FieldT, Int> &data) {
    ostriple->zkp_conv(model.conv_model[0], data.input, data.conv_output[0]);
    ostriple->zkp_relu(data.conv_output[0], data.relu_output[0]);
    ostriple->zkp_maxpool(data.relu_output[0], data.max_output);

    int conv_cnt = 1, relu_cnt = 1, skip_cnt = 0;

    int seq[] = {3, 4, 23, 3};
    for (int j = 0; j < ARRLEN(seq); j++) {
      for (int seq_id = 0; seq_id < seq[j]; seq_id++) {
        const QuantizedValueProver<FieldT, Int> *skip_add_in = NULL;
        if (seq_id == 0) {
          ostriple->zkp_conv(
              model.conv_model[conv_cnt],
              j == 0 ? data.max_output : data.relu_output[relu_cnt - 1],
              data.conv_output[conv_cnt]);
          conv_cnt++;
          skip_add_in = &(data.conv_output[conv_cnt - 1]);
        } else {
          skip_add_in = &(data.relu_output[relu_cnt - 1]);
        }
        ostriple->zkp_conv(model.conv_model[conv_cnt],
                           j == 0 && seq_id == 0
                               ? data.max_output
                               : data.relu_output[relu_cnt - 1],
                           data.conv_output[conv_cnt]);
        conv_cnt++;
        ostriple->zkp_relu(data.conv_output[conv_cnt - 1],
                           data.relu_output[relu_cnt]);
        relu_cnt++;
        ostriple->zkp_conv(model.conv_model[conv_cnt],
                           data.relu_output[relu_cnt - 1],
                           data.conv_output[conv_cnt]);
        conv_cnt++;
        ostriple->zkp_relu(data.conv_output[conv_cnt - 1],
                           data.relu_output[relu_cnt]);
        relu_cnt++;
        ostriple->zkp_conv(model.conv_model[conv_cnt],
                           data.relu_output[relu_cnt - 1],
                           data.conv_output[conv_cnt]);
        conv_cnt++;
        ostriple->zkp_skip_add(
            model.skip_add_model[skip_cnt], *skip_add_in,
            data.conv_output[conv_cnt - 1], data.add_output1[skip_cnt],
            data.add_output2[skip_cnt], data.add_output[skip_cnt]);
        skip_cnt++;
        ostriple->zkp_relu(data.add_output[skip_cnt - 1],
                           data.relu_output[relu_cnt]);
        relu_cnt++;
      }
    }
    ostriple->zkp_fc(model.fc_model, data.avg_output, data.fc_output);
    ostriple->zkp_range_batch(FieldT((1ul << 26) + 1));
    ostriple->batch_check(false, false, true);
  }

  template <typename Int>
  void VGG11(const VGG11PrvModel<FieldT, Int> &model,
             const VGG11PrvData<FieldT, Int> &data) {
    int conv_cnt = 0, relu_cnt = 0, maxpool_cnt = 0, linear_cnt = 0;
    int features[] = {
        0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2,
    };
    const QuantizedValueProver<FieldT, Int> *conv_in = &data.input;
    for (int i = 0; i < ARRLEN(features); i++) {
      if (features[i] == 0) {
        ostriple->zkp_conv(model.conv_model[conv_cnt], *conv_in,
                           data.conv_output[conv_cnt]);
        conv_cnt++;
      } else if (features[i] == 1) {
        ostriple->zkp_relu(data.conv_output[conv_cnt - 1],
                           data.relu_output[relu_cnt]);
        conv_in = &data.relu_output[relu_cnt];
        relu_cnt++;
      } else if (features[i] == 2) {
        ostriple->zkp_maxpool(data.relu_output[relu_cnt - 1],
                              data.max_output[maxpool_cnt], 0, 2, 2);
        conv_in = &data.max_output[maxpool_cnt];
        maxpool_cnt++;
      } else
        assert(0);
    }
    int classifier[] = {3, 1, 3, 1, 3};
    for (int i = 0; i < ARRLEN(classifier); i++) {
      if (classifier[i] == 1) {
        ostriple->zkp_relu(data.fc_output[linear_cnt - 1],
                           data.relu_output[relu_cnt]);
        relu_cnt++;
      } else if (classifier[i] == 3) {
        ostriple->zkp_fc(
            model.fc_model[linear_cnt],
            i == 0 ? data.avg_output : data.relu_output[relu_cnt - 1],
            data.fc_output[linear_cnt]);
        linear_cnt++;
      }
    }

    ostriple->zkp_range_batch(FieldT((1ul << 26) + 1));
    ostriple->batch_check(false, false, true);
  }

  template <typename Int>
  void GPT2(const GPT2PrvModel<FieldT, Int> &model,
            GPT2PrvData<FieldT, Int> &data) {
    TIME_STATS_BEG
    ostriple->zkp_trans(model.trans, data);
    ostriple->zkp_layer_norm(model.layer_norm, data.res_out[23], data.mean[24],
                             data.var[24], data.std[24], data.std_max[24],
                             data.sub[24], data.norm[24],
                             data.layer_norm_out[24]);

    ostriple->zkp_range_batch(FieldT((1ul << 26) + 1));
    ostriple->batch_check(false, false, true);
    TIME_STATS_END
  }
};
#endif
