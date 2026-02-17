#include <time.h>
#include <zknn/utility.h>

#include <iostream>
#include <libff/libff/algebra/curves/alt_bn128/alt_bn128_pp.hpp>

#include "emp-tool/emp-tool.h"
#include "zknn/zknn-arith/cnn-prover.h"
#include "zknn/zknn-arith/cnn-verifier.h"
#include "zknn/zknn-arith/ostriple-prover.h"
#include "zknn/zknn-arith/ostriple-verifier.h"
#include "zknn/zknn-arith/zk-fp-exec-prover.h"
#include "zknn/zknn-arith/zk-fp-exec-verifier.h"

using namespace emp;
using namespace std;
using namespace libff;

#ifdef RESNET18
#define DATA_PATH "model/dump/resnet18"
#define CNNPrvModel ResNet18PrvModel
#define CNNVerModel ResNet18VerModel
#define CNNPrvData ResNet18PrvData
#define CNNVerData ResNet18VerData
#define ZKP ResNet18
#elif RESNET50
#define DATA_PATH "model/dump/resnet50"
#define CNNPrvModel ResNet50PrvModel
#define CNNVerModel ResNet50VerModel
#define CNNPrvData ResNet50PrvData
#define CNNVerData ResNet50VerData
#define ZKP ResNet50
#elif RESNET101
#define DATA_PATH "model/dump/resnet101"
#define CNNPrvModel ResNet101PrvModel
#define CNNVerModel ResNet101VerModel
#define CNNPrvData ResNet101PrvData
#define CNNVerData ResNet101VerData
#define ZKP ResNet101
#elif VGG_11
#define DATA_PATH "model/dump/vgg11"
#define CNNPrvModel VGG11PrvModel
#define CNNVerModel VGG11VerModel
#define CNNPrvData VGG11PrvData
#define CNNVerData VGG11VerData
#define ZKP VGG11
#else
#define DATA_PATH "model/dump/resnet18"
#define CNNPrvModel ResNet18PrvModel
#define CNNVerModel ResNet18VerModel
#define CNNPrvData ResNet18PrvData
#define CNNVerData ResNet18VerData
#define ZKP ResNet18
#endif

#define STRING_(X) #X
#define STRING(X) STRING_(X)
#define CNN_NAME STRING(ZKP)

int port, party;
const int threads = 128;

extern size_t recv_size;
extern double run_time;

template <typename FieldT>
void test_circuit_zk_prover(NetIO* ios[threads + 1], int input_sz_lg) {
  FpOSTriplePrv<NetIO, FieldT>* fp =
      new FpOSTriplePrv<NetIO, FieldT>(ios, threads);

  printf("Prover read model & data begin\n");
  // read model & data
  CNNPrvModel<FieldT, int> pm(DATA_PATH);
  pm.auth(fp);

  CNNPrvData<FieldT, int> pd(DATA_PATH);
  pd.auth(fp);

  printf("Prover read model & data end\n");

  ios[0]->flush();
  double itime, ftime, exec_time;
  itime = omp_get_wtime();
  s_time = omp_get_wtime();
  run_time = 0;

  ZKFpExecPrv<NetIO, FieldT>* pe = new ZKFpExecPrv<NetIO, FieldT>(ios, fp);
  pe->ZKP(pm, pd);

  run_time += omp_get_wtime() - s_time;
  printf("prover Run time: %f s\n", run_time);
}

template <typename FieldT>
void test_circuit_zk_verifier(NetIO* ios[threads + 1], int input_sz_lg) {
  FpOSTripleVer<NetIO, FieldT>* fv =
      new FpOSTripleVer<NetIO, FieldT>(ios, threads);

  printf("Verifier read model & data begin\n");
  // model and data
  CNNVerModel<FieldT> vm(DATA_PATH);
  vm.auth(fv);

  CNNVerData<FieldT> vd(DATA_PATH);
  vd.auth(fv);

  printf("Verifier read model & data end\n");

  recv_size = 0;
  s_time = omp_get_wtime();
  run_time = 0;

  ios[0]->flush();
  ZKFpExecVer<NetIO, FieldT>* ve = new ZKFpExecVer<NetIO, FieldT>(ios, fv);
  bool check = ve->template ZKP<int>(vm, vd);
  run_time += omp_get_wtime() - s_time;
  printf("verifier Run time: %fs\n", run_time);
  printf("Proof size: %f MB\n", (float)recv_size / 1024 / 1024);
  printf("check: %d\n", check);
}

int main(int argc, char** argv) {
  parse_party_and_port(argv, &party, &port);
  NetIO* ios[threads + 1];
  for (int i = 0; i < threads + 1; ++i) {
    ios[i] = new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i);
    ios[i]->set_nodelay();
  }

  std::cout << std::endl
            << "------------ circuit zero-knowledge proof test ------------"
            << std::endl;
  puts(CNN_NAME);

  size_t num = 20;
  alt_bn128_pp::init_public_params();

  if (party == ALICE) {
    libff::Fr<libff::alt_bn128_pp>::three_square_decomp_opti_init(1ul << 32, "two_square.bin");
    test_circuit_zk_prover<Fr<alt_bn128_pp>>(ios, num);
  } else test_circuit_zk_verifier<Fr<alt_bn128_pp>>(ios, num);

  // free ios
  for (int i = 0; i < threads + 1; ++i) {
    delete ios[i];
  }

  return 0;
}
