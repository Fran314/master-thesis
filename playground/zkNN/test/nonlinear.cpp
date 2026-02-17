#include <omp.h>

#include <cstdio>

#include "emp-tool/io/net_io_channel.h"
#include "libff/algebra/curves/alt_bn128/alt_bn128_pp.hpp"
#include "zknn/zknn-arith/ostriple-prover.h"
#include "zknn/zknn-arith/ostriple-verifier.h"
#include "zknn/zknn-arith/transformer-prover.h"
#include "zknn/zknn-arith/transformer-verifier.h"
#include "zknn/zknn-arith/zk-fp-exec-prover.h"
#include "zknn/zknn-arith/zk-fp-exec-verifier.h"

int port, party;
const int threads = 128;

#define PROVER_DATA(data, filename, read)           \
  {                                                 \
    size_t sz = read(filename, &data);              \
    data.scale = NormFPProver<FieldT>(data.fscale); \
    data.value.resize(sz);                          \
    for (size_t i = 0; i < sz; i++) {               \
      data.value[i] = FieldT(data.ivalue[i]);       \
    }                                               \
  }

#define VERIFIER_DATA(data, filename, read) \
  {                                         \
    size_t sz = read(filename, &data);      \
    data.key_value.resize(sz);              \
  }

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)

#define READ_AUTH(var, dir, file, role, f, read) \
  T var;                                            \
  role##_DATA(var, STRINGIFY(dir/file), read);         \
  var.auth(f);

#define READ_AUTH_NUM_PROVER(var, dir, file, f) \
  T var;                                       \
{                                              \
  long t;                                      \
  read_number(STRINGIFY(dir/file), t);         \
  var.shape.push_back(1);                      \
  var.shape.push_back(1);                      \
  var.value.push_back(FieldT(t));              \
}                                              \
  var.auth(f);

#define READ_AUTH_NUM_VERIFIER(var, dir, file, f) \
  T var;                                          \
  var.key_value.resize(1);                        \
  var.shape.push_back(1);                         \
  var.shape.push_back(1);                         \
  var.auth(f);

#define PREPARE_DATA_RELU(path, role, f, read) \
  READ_AUTH(relu_in, path, relu_in, role, f, (read)); \
  READ_AUTH(relu_out, path, relu_out, role, f, (read));


#define PREPARE_DATA_SOFTMAX(path, role, f, read) \
  READ_AUTH(softmax_in, path, i_softmax_in_0, role, f, (read)); \
  READ_AUTH(x_max, path, i_softmax_x_max_0, role, f, (read));   \
  READ_AUTH(softmax_out, path, i_softmax_out_0, role, f, (read)); \
  READ_AUTH(softmax_out1, path, i_softmax_out1_0, role, f, (read)); \
  READ_AUTH(z, path, i_exp_z_0, role, f, (read)); \
  READ_AUTH(p, path, i_exp_p_0, role, f, (read)); \
  READ_AUTH(p1, path, i_exp_p1_0, role, f, (read));             \
  READ_AUTH(p2, path, i_exp_p2_0, role, f, (read));             \
  READ_AUTH_NUM_##role(exp_poly_out3, path, i_exp_x3_0, f);     \
  READ_AUTH(l, path, i_exp_l_0, role, f, (read)); \
  READ_AUTH(exp_poly_out1, path, i_exp_x1_0, role, f, (read));  \
  READ_AUTH(exp_poly_out2, path, i_exp_x2_0, role, f, (read));  \
  READ_AUTH(exp_out, path, i_exp_out_0, role, f, (read));

#define PREPARE_DATA_GELU(path, role, f, read) \
  READ_AUTH(gelu_in, path, gelu_in_0, role, f, (read)); \
  READ_AUTH(gelu_out, path, gelu_out_0, role, f, (read)); \
  READ_AUTH(gelu_out1, path, gelu_out1_0, role, f, (read)); \
  READ_AUTH(gelu_out2, path, gelu_out2_0, role, f, (read)); \
  READ_AUTH(sign, path, i_erf_sign_0, role, f, (read)); \
  READ_AUTH(q_abs, path, i_erf_q_abs_0, role, f, (read)); \
  READ_AUTH_NUM_##role(zb, path, i_erf_zb_0, f); \
  READ_AUTH(q_min, path, i_erf_q_min_0, role, f, (read)); \
  READ_AUTH(erf_l, path, i_erf_l_0, role, f, (read));   \
  READ_AUTH(gelu_poly_out1, path, i_erf_x1_0, role, f, (read)); \
  READ_AUTH(gelu_poly_out2, path, i_erf_x2_0, role, f, (read)); \
  READ_AUTH_NUM_##role(gelu_poly_out3, path, i_erf_x3_0, f);\
  READ_AUTH(erf_out, path, i_erf_out_0, role, f, (read));

#define PREPARE_DATA_LAYERNORM(path, role, f, read) \
  READ_AUTH(layer_norm_in, path, embd_out, role, f, (read)); \
  READ_AUTH(mean, path, layer_norm_mean_0, role, f, (read));        \
  READ_AUTH(var, path, layer_norm_var_0, role, f, (read));          \
  READ_AUTH(std, path, layer_norm_std_0, role, f, (read));          \
  READ_AUTH(std_max, path, layer_norm_std_max_0, role, f, (read));  \
  READ_AUTH(sub, path, layer_norm_sub_0, role, f, (read));          \
  READ_AUTH(norm, path, layer_norm_norm_0, role, f, (read));

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

template <typename FieldT>
static void relu_prover(NetIO *ios[threads]) {
  auto fp = new FpOSTriplePrv<NetIO, FieldT>(ios, threads);
  printf("Prover read model & data begin\n");

  #define T QuantizedValueProver<FieldT, int>
  // relu
  PREPARE_DATA_RELU(model/dump/nonlinear/relu, PROVER, fp, cnn::read_vec);

  printf("Prover read model & data end\n");
  puts("prover start");

  stats_beg(__FUNCTION__);
  fp->zkp_relu(relu_in, relu_out);
  fp->zkp_range_batch(B);
  fp->batch_check(false, false, true);
  stats_end(__FUNCTION__);
}

template <typename FieldT>
static void softmax_prover(NetIO *ios[threads]) {
  auto fp = new FpOSTriplePrv<NetIO, FieldT>(ios, threads);
  printf("Prover read model & data begin\n");

  #define T QuantizedValueProver<FieldT, long>
  SftMaxPrvModel<FieldT> sft("model/dump/nonlinear/softmax", 0);
  sft.auth(fp);
  PREPARE_DATA_SOFTMAX(model/dump/nonlinear/softmax, PROVER, fp, transformer::read_vec);

  printf("Prover read model & data end\n");
  puts("prover start");

  stats_beg(__FUNCTION__);
  fp->zkp_softmax(sft, softmax_in, x_max, z, p, p1, p2, l, exp_out, exp_poly_out1, exp_poly_out2, exp_poly_out3, softmax_out, softmax_out1);
  fp->zkp_range_batch(B);
  fp->batch_check(false, false, true);
  stats_end(__FUNCTION__);
}

template <typename FieldT>
static void gelu_prover(NetIO *ios[threads]) {
  auto fp = new FpOSTriplePrv<NetIO, FieldT>(ios, threads);
  printf("Prover read model & data begin\n");

  #define T QuantizedValueProver<FieldT, long>
  GeluPrvModel<FieldT> gelu("model/dump/nonlinear/gelu", 0);
  gelu.auth(fp);
  PREPARE_DATA_GELU(model/dump/nonlinear/gelu, PROVER, fp, transformer::read_vec);

  printf("Prover read model & data end\n");
  puts("prover start");

  stats_beg(__FUNCTION__);
  fp->zkp_gelu(gelu, gelu_in, sign, q_abs, zb, q_min, gelu_poly_out1, gelu_poly_out2, gelu_poly_out3, erf_l, erf_out, gelu_out1, gelu_out2, gelu_out);
  fp->zkp_range_batch(B);
  fp->batch_check(false, false, true);
  stats_end(__FUNCTION__);
}

template <typename FieldT>
static void layernorm_prover(NetIO *ios[threads]) {
  auto fp = new FpOSTriplePrv<NetIO, FieldT>(ios, threads);
  printf("Prover read model & data begin\n");

  #define T QuantizedValueProver<FieldT, long>
  LayerNormPrvModel<FieldT, long> layernorm("model/dump/nonlinear/norm", 0);
  layernorm.auth(fp);
  PREPARE_DATA_LAYERNORM(model/dump/nonlinear/norm, PROVER, fp, transformer::read_vec);

  printf("Prover read model & data end\n");
  puts("prover start");

  stats_beg(__FUNCTION__);
  fp->zkp_layer_norm(layernorm, layer_norm_in, mean, var, std, std_max, sub, norm);
  fp->zkp_range_batch(B);
  fp->batch_check(false, false, true);
  stats_end(__FUNCTION__);
}

template <typename FieldT>
static void relu_verifier(NetIO *ios[threads]) {
  auto fv = new FpOSTripleVer<NetIO, FieldT>(ios, threads);
  printf("Verifier read model & data begin\n");

  #define T QuantizedValueVerifier<FieldT>

  PREPARE_DATA_RELU(model/dump/nonlinear/relu, VERIFIER, fv, cnn::read_shape);

  printf("Verifier read model & data end\n");

  puts("verifier start");

  stats_beg(__FUNCTION__);
  fv->template zkp_relu<int>(relu_in, relu_out);
  fv->zkp_range_batch(B);
  bool check = fv->batch_check(false, false, true);
  stats_end(__FUNCTION__);
  printf("proof size: %f MB\n", (float)recv_size / 1024 / 1024);
  printf("check: %d\n", check);
}

template <typename FieldT>
static void softmax_verifier(NetIO *ios[threads]) {
  auto fv = new FpOSTripleVer<NetIO, FieldT>(ios, threads);
  printf("Verifier read model & data begin\n");

  #define T QuantizedValueVerifier<FieldT>

  SftMaxVerModel<FieldT> sft;
  sft.auth(fv);
  PREPARE_DATA_SOFTMAX(model/dump/nonlinear/softmax, VERIFIER, fv, transformer::read_shape);

  printf("Verifier read model & data end\n");

  puts("verifier start");

  stats_beg(__FUNCTION__);
  fv->template zkp_softmax<long>(sft, softmax_in, x_max, z, p, p1, p2, l, exp_out, exp_poly_out1, exp_poly_out2, exp_poly_out3, softmax_out, softmax_out1);
  fv->zkp_range_batch(B);
  bool check = fv->batch_check(false, false, true);
  stats_end(__FUNCTION__);
  printf("proof size: %f MB\n", (float)recv_size / 1024 / 1024);
  printf("check: %d\n", check);
}

template <typename FieldT>
static void gelu_verifier(NetIO *ios[threads]) {
  auto fv = new FpOSTripleVer<NetIO, FieldT>(ios, threads);
  printf("Verifier read model & data begin\n");

  #define T QuantizedValueVerifier<FieldT>

  GeluVerModel<FieldT> gelu;
  gelu.auth(fv);
  PREPARE_DATA_GELU(model/dump/nonlinear/gelu, VERIFIER, fv, transformer::read_shape);

  printf("Verifier read model & data end\n");

  puts("verifier start");

  stats_beg(__FUNCTION__);
  fv->template zkp_gelu<long>(gelu, gelu_in, sign, q_abs, zb, q_min, gelu_poly_out1, gelu_poly_out2, gelu_poly_out3, erf_l, erf_out, gelu_out1, gelu_out2, gelu_out);
  fv->zkp_range_batch(B);
  bool check = fv->batch_check(false, false, true);
  stats_end(__FUNCTION__);
  printf("proof size: %f MB\n", (float)recv_size / 1024 / 1024);
  printf("check: %d\n", check);
}

template <typename FieldT>
static void layernorm_verifier(NetIO *ios[threads]) {
  auto fv = new FpOSTripleVer<NetIO, FieldT>(ios, threads);
  printf("Verifier read model & data begin\n");

  #define T QuantizedValueVerifier<FieldT>

  LayerNormVerModel<FieldT> layernorm("model/dump/nonlinear/norm", 0);
  layernorm.auth(fv);
  PREPARE_DATA_LAYERNORM(model/dump/nonlinear/norm, VERIFIER, fv, transformer::read_shape);

  printf("Verifier read model & data end\n");

  puts("verifier start");

  stats_beg(__FUNCTION__);
  fv->template zkp_layer_norm<long>(layernorm, layer_norm_in, mean, var, std, std_max, sub, norm);
  fv->zkp_range_batch(B);
  bool check = fv->batch_check(false, false, true);
  stats_end(__FUNCTION__);
  printf("proof size: %f MB\n", (float)recv_size / 1024 / 1024);
  printf("check: %d\n", check);
}


template <typename FieldT>
void test_circuit_zk_prover(NetIO *ios[threads]) {
  relu_prover<FieldT>(ios);
  softmax_prover<FieldT>(ios);
  gelu_prover<FieldT>(ios);
  layernorm_prover<FieldT>(ios);
}

template <typename FieldT>
void test_circuit_zk_verifier(NetIO *ios[threads]) {

  relu_verifier<FieldT>(ios);
  softmax_verifier<FieldT>(ios);
  gelu_verifier<FieldT>(ios);
  layernorm_verifier<FieldT>(ios);
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
