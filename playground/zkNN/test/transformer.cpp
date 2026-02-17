#include <omp.h>

#include <cstdio>

#include "emp-tool/io/net_io_channel.h"
#include "libff/algebra/curves/alt_bn128/alt_bn128_pp.hpp"
#include "zknn/zknn-arith/ostriple-prover.h"
#include "zknn/zknn-arith/ostriple-verifier.h"
#include "zknn/zknn-arith/transformer-common.h"
#include "zknn/zknn-arith/transformer-prover.h"
#include "zknn/zknn-arith/transformer-verifier.h"
#include "zknn/zknn-arith/zk-fp-exec-prover.h"
#include "zknn/zknn-arith/zk-fp-exec-verifier.h"

#define DATA_PATH "model/dump/gpt2"

int port, party;
const int threads = 128;

#define B FieldT((1ll << 26) + 1)

static void stats_beg(const char *role) {
  printf("%s begin\n", role);
  time_stats.clear();
  recv_size = 0;
  run_time = 0;
  s_time = omp_get_wtime();
}

static void stats_end(const char *role) {
  auto et = omp_get_wtime();
  run_time += et - s_time;

  printf("%s over, calc time: %f\n", role, run_time);
}

template <typename FieldT, typename Int>
static void gelu_prover(FpOSTriplePrv<NetIO, FieldT> *fp,
                        GPT2PrvModel<FieldT, Int> &model,
                        GPT2PrvData<FieldT, Int> &data) {
  stats_beg(__FUNCTION__);
  for (size_t i = 0; i < TRANS_NUM; i++) {
    fp->zkp_gelu(model.trans[i].ffn.gelu, data.linear_out[4 * i + 2], data.sign[i],
            data.q_abs[i], data.zb[i], data.q_min[i], data.gelu_poly_out1[i],
            data.gelu_poly_out2[i], data.gelu_poly_out3[i], data.erf_l[i],
            data.erf_out[i], data.gelu_out1[i], data.gelu_out2[i],
            data.gelu_out[i]);
  }
  fp->zkp_range_batch(B);
  fp->batch_check(false, false, true);
  stats_end(__FUNCTION__);
}

template <typename FieldT, typename Int>
static void softmax_prover(FpOSTriplePrv<NetIO, FieldT> *fp,
                        GPT2PrvModel<FieldT, Int> &model,
                        GPT2PrvData<FieldT, Int> &data) {
  stats_beg(__FUNCTION__);
  for (size_t i = 0; i < TRANS_NUM; i++) {
    for (size_t j = 0; j < N_HEAD; j++) {
      fp->zkp_softmax(model.trans[i].mha.attn[j].softmax, data, i * N_HEAD + j);
    }
  }
  fp->zkp_range_batch(B);
  fp->batch_check(false, false, true);
  stats_end(__FUNCTION__);
}

template <typename FieldT, typename Int>
static void layernorm_prover(FpOSTriplePrv<NetIO, FieldT> *fp,
                        GPT2PrvModel<FieldT, Int> &model,
                        GPT2PrvData<FieldT, Int> &data) {
  stats_beg(__FUNCTION__);
  const QuantizedValueProver<FieldT, Int> *in = &data.embd_out;
  for (size_t i = 0; i < TRANS_NUM; i++) {
    fp->zkp_layer_norm(model.trans[i].layer_norm[0], *in, data.mean[2 * i],
                     data.var[2 * i], data.std[2 * i], data.std_max[2 * i],
                     data.sub[2 * i], data.norm[2 * i]);
    fp->zkp_layer_norm(model.trans[i].layer_norm[1], data.res_out[2 * i],
                     data.mean[2 * i + 1], data.var[2 * i + 1],
                     data.std[2 * i + 1], data.std_max[2 * i + 1],
                     data.sub[2 * i + 1], data.norm[2 * i + 1]);
    in = &data.res_out[2 * i + 1];
  }
  fp->zkp_layer_norm(model.layer_norm, data.res_out[23], data.mean[24],
                            data.var[24], data.std[24], data.std_max[24],
                            data.sub[24], data.norm[24]);
  fp->zkp_range_batch(B);
  fp->batch_check(false, false, true);
  stats_end(__FUNCTION__);
}

template <typename FieldT, typename Int>
static void linear_prover(FpOSTriplePrv<NetIO, FieldT> *fp,
                        GPT2PrvModel<FieldT, Int> &model,
                        GPT2PrvData<FieldT, Int> &data) {
  stats_beg(__FUNCTION__);

  const QuantizedValueProver<FieldT, Int> *in = &data.embd_out;
  for (size_t i = 0; i < TRANS_NUM; i++) {
    fp->zkp_layer_norm_out(model.trans[i].layer_norm[0], data.norm[2 * i], data.layer_norm_out[2 * i]);
    fp->zkp_layer_norm_out(model.trans[i].layer_norm[1], data.norm[2 * i + 1], data.layer_norm_out[2 * i + 1]);
    fp->zkp_ffn_without_gelu(model.trans[i].ffn, data, i);
    fp->zkp_mha_without_softmax(model.trans[i].mha, data, i);;

    in = &data.res_out[2 * i + 1];
  }
  fp->zkp_layer_norm_out(model.layer_norm, data.norm[24], data.layer_norm_out[24]);

  fp->zkp_range_batch(B);
  fp->batch_check(false, false, true);
  stats_end(__FUNCTION__);
}

template <typename FieldT, typename Int>
static void gelu_verifier(FpOSTripleVer<NetIO, FieldT> *fv,
                          GPT2VerModel<FieldT> &model,
                          GPT2VerData<FieldT> &data) {
  stats_beg(__FUNCTION__);
  for (size_t i = 0; i < TRANS_NUM; i++) {
    fv->template zkp_gelu<Int>(model.trans[i].ffn.gelu, data.linear_out[4 * i + 2], data.sign[i],
            data.q_abs[i], data.zb[i], data.q_min[i], data.gelu_poly_out1[i],
            data.gelu_poly_out2[i], data.gelu_poly_out3[i], data.erf_l[i],
            data.erf_out[i], data.gelu_out1[i], data.gelu_out2[i],
            data.gelu_out[i]);
  }
  fv->zkp_range_batch(B);
  bool check = fv->batch_check(false, false, true);
  stats_end(__FUNCTION__);
  printf("proof size: %f MB\n", (float)recv_size / 1024 / 1024);
  printf("check: %d\n", check);
}

template <typename FieldT, typename Int>
static void softmax_verifier(FpOSTripleVer<NetIO, FieldT> *fv,
                          GPT2VerModel<FieldT> &model,
                          GPT2VerData<FieldT> &data) {
  stats_beg(__FUNCTION__);
  for (size_t i = 0; i < TRANS_NUM; i++) {
    for (size_t j = 0; j < N_HEAD; j++) {
      fv->template zkp_softmax<Int>(model.trans[i].mha.attn[j].softmax, data, i * N_HEAD + j);
    }
  }
  fv->zkp_range_batch(B);
  bool check = fv->batch_check(false, false, true);
  stats_end(__FUNCTION__);
  printf("proof size: %f MB\n", (float)recv_size / 1024 / 1024);
  printf("check: %d\n", check);
}

template <typename FieldT, typename Int>
static void layernorm_verifier(FpOSTripleVer<NetIO, FieldT> *fv,
                          GPT2VerModel<FieldT> &model,
                          GPT2VerData<FieldT> &data) {
  stats_beg(__FUNCTION__);
  const QuantizedValueVerifier<FieldT> *in = &data.embd_out;
  for (size_t i = 0; i < TRANS_NUM; i++) {
    fv->template zkp_layer_norm<Int>(model.trans[i].layer_norm[0], *in, data.mean[2 * i],
                     data.var[2 * i], data.std[2 * i], data.std_max[2 * i],
                     data.sub[2 * i], data.norm[2 * i]);
    fv->template zkp_layer_norm<Int>(model.trans[i].layer_norm[1], data.res_out[2 * i],
                     data.mean[2 * i + 1], data.var[2 * i + 1],
                     data.std[2 * i + 1], data.std_max[2 * i + 1],
                     data.sub[2 * i + 1], data.norm[2 * i + 1]);
    in = &data.res_out[2 * i + 1];
  }
  fv->template zkp_layer_norm<Int>(model.layer_norm, data.res_out[23], data.mean[24],
                            data.var[24], data.std[24], data.std_max[24],
                            data.sub[24], data.norm[24]);
  fv->zkp_range_batch(B);
  bool check = fv->batch_check(false, false, true);
  stats_end(__FUNCTION__);
  printf("proof size: %f MB\n", (float)recv_size / 1024 / 1024);
  printf("check: %d\n", check);
}

template <typename FieldT, typename Int>
static void linear_verifier(FpOSTripleVer<NetIO, FieldT> *fv,
                          GPT2VerModel<FieldT> &model,
                          GPT2VerData<FieldT> &data) {
  stats_beg(__FUNCTION__);

  const QuantizedValueVerifier<FieldT> *in = &data.embd_out;
  for (size_t i = 0; i < TRANS_NUM; i++) {
    fv->template zkp_layer_norm_out<Int>(model.trans[i].layer_norm[0], data.norm[2 * i], data.layer_norm_out[2 * i]);
    fv->template zkp_layer_norm_out<Int>(model.trans[i].layer_norm[1], data.norm[2 * i + 1], data.layer_norm_out[2 * i + 1]);
    fv->template zkp_ffn_without_gelu<Int>(model.trans[i].ffn, data, i);
    fv->template zkp_mha_without_softmax<Int>(model.trans[i].mha, data, i);;

    in = &data.res_out[2 * i + 1];
  }
  fv-> template zkp_layer_norm_out<Int>(model.layer_norm, data.norm[24], data.layer_norm_out[24]);

  fv->zkp_range_batch(B);
  bool check = fv->batch_check(false, false, true);
  stats_end(__FUNCTION__);
  printf("proof size: %f MB\n", (float)recv_size / 1024 / 1024);
  printf("check: %d\n", check);
}

template <typename FieldT>
void test_circuit_zk_prover(NetIO *ios[threads]) {
  auto fp = new FpOSTriplePrv<NetIO, FieldT>(ios, threads);

  printf("Prover read model & data begin\n");

  GPT2PrvModel<FieldT, long> pm(DATA_PATH);
  pm.auth(fp);

  GPT2PrvData<FieldT, long> pd(DATA_PATH);
  pd.auth(fp);

  printf("Prover read model & data end\n");
  puts("prover start");

  gelu_prover(fp, pm, pd);
  softmax_prover(fp, pm, pd);
  layernorm_prover(fp, pm, pd);
  linear_prover(fp, pm, pd);
 }

template <typename FieldT>
void test_circuit_zk_verifier(NetIO *ios[threads]) {
  auto fv = new FpOSTripleVer<NetIO, FieldT>(ios, threads);

  printf("Verifier read model & data begin\n");

  GPT2VerModel<FieldT> vm(DATA_PATH);
  vm.auth(fv);

  GPT2VerData<FieldT> vd(DATA_PATH);
  vd.auth(fv);

  printf("Verifier read model & data end\n");

  puts("verifier start");
  gelu_verifier<FieldT, long>(fv, vm, vd);
  softmax_verifier<FieldT, long>(fv, vm, vd);
  layernorm_verifier<FieldT, long>(fv, vm, vd);
  linear_verifier<FieldT, long>(fv, vm, vd);
}

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  NetIO *ios[threads + 1];
  for (int i = 0; i < threads + 1; ++i)
    ios[i] = new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i);

  std::cout << std::endl
            << "------------ circuit zero-knowledge proof test ------------"
            << std::endl;
  libff::alt_bn128_pp::init_public_params();

  if (party == ALICE) {
    libff::Fr<libff::alt_bn128_pp>::three_square_decomp_opti_init(
        1ul << 32, "two_square.bin");
    test_circuit_zk_prover<libff::Fr<libff::alt_bn128_pp>>(ios);
  } else
    test_circuit_zk_verifier<libff::Fr<libff::alt_bn128_pp>>(ios);

  for (int i = 0; i < threads + 1; ++i) {
    delete ios[i];
  }

  return 0;
}