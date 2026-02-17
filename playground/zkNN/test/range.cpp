#include <time.h>
#include <zknn/utility.h>
#include <zknn/zknn-prover.h>
#include <zknn/zknn-verifier.h>

#include <iostream>
#include <libff/libff/algebra/curves/alt_bn128/alt_bn128_pp.hpp>
#include <system_error>
#include <vector>

#include "emp-tool/emp-tool.h"
using namespace emp;
using namespace std;

using namespace libff;

int port, party;
const int threads = 128;
const int ITER = 10;

extern size_t send_size;
extern size_t recv_size;
extern double run_time;

static void print_stats() {
  int width = 20;
  printf("%*s%*s%*s%*s%*s%*s\n", width, "function", width, "calc_time", width,
         "execute_count", width, "average_time", width, "percentage", width,
         "recv_size");
  std::vector<std::pair<std::string, TimeStats>> sorted(time_stats.begin(),
                                                        time_stats.end());
  sort(sorted.begin(), sorted.end(),
       [](const std::pair<std::string, TimeStats> &a,
          const std::pair<std::string, TimeStats> &b) {
         return a.second.tot_time < b.second.tot_time;
       });
  for (auto p : sorted) {
    printf("%*s%*f%*d%*f%*f%%%*lu\n", width, p.first.c_str(), width,
           p.second.tot_time / ITER, width, p.second.exec_cnt / ITER, width,
           p.second.average_time(), width,
           p.second.percentage(run_time) * 100, width,
           p.second.transfer_sz / ITER);
  }
}

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

  printf("%s over, calc time: %f\n", role, run_time / ITER);
  printf("%s recv: %f MB\n", role, (float)recv_size / 1024 / 1024 / ITER);
  print_stats();
}

template <typename FieldT>
void test_range_prover(NetIO *ios[threads + 1], int input_sz_lg) {
  FpOSTriplePrv<NetIO, FieldT> *fp =
      new FpOSTriplePrv<NetIO, FieldT>(ios, threads);
  size_t num = 1 << input_sz_lg;
  vector<FieldT> v(num);
  vector<FieldT> mac_v(num);
  FieldT B = (FieldT(2) ^ 24) - FieldT::one();
  for (size_t i = 0; i < num; i++) v[i] = FieldT(i);
  fp->authenticated_val_input(mac_v, v, num);

  printf("prover range proof begin\n");

  stats_beg(__FUNCTION__);
  for (int i = 0; i < ITER; i++) {
    fp->zkp_range(mac_v, v, B, num);
    fp->zkp_range_batch(B);
    fp->batch_check(false, false, true);
  }
  stats_end(__FUNCTION__);

  run_time += omp_get_wtime() - s_time;
  printf("prover Run time: %f s\n", run_time);
}

template <typename FieldT>
void test_range_verifier(NetIO *ios[threads + 1], int input_sz_lg) {
  FpOSTripleVer<NetIO, FieldT> *fv =
      new FpOSTripleVer<NetIO, FieldT>(ios, threads);
  size_t num = 1 << input_sz_lg;
  vector<FieldT> key_v(num);
  fv->authenticated_val_input(key_v, num);
  FieldT B = (FieldT(2) ^ 24) - FieldT::one();

  printf("verifier range proof begin\n");

  stats_beg(__FUNCTION__);
  for (int i = 0; i < ITER; i++) {
    fv->zkp_range(key_v, B, num);
    fv->zkp_range_batch(B);
    bool check = fv->batch_check(false, false, true);
    assert(check);
  }
  stats_end(__FUNCTION__);

  run_time += omp_get_wtime() - s_time;
  printf("verifier Run time: %f\n", run_time);

  printf("proof size: %f MB\n", (float)recv_size / 1024 / 1024 / ITER);
}

int main(int argc, char **argv) {
  alt_bn128_pp::init_public_params();

  parse_party_and_port(argv, &party, &port);
  NetIO *ios[threads + 1];
  for (int i = 0; i < threads + 1; ++i) {
    ios[i] = new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i);
  }

  std::cout << std::endl
            << "------------ circuit zero-knowledge proof test ------------"
            << std::endl;

  if (party == ALICE)
    libff::Fr<libff::alt_bn128_pp>::three_square_decomp_opti_init(
        1ul << 32, "two_square.bin");

  for (int i = 14; i <= 18; i += 2) {
    printf("N = %d\n", 1 << i);
    if (party == ALICE) {
      test_range_prover<Fr<alt_bn128_pp>>(ios, i);
    } else {
      test_range_verifier<Fr<alt_bn128_pp>>(ios, i);
    }
  }

  // free ios
  for (int i = 0; i < threads + 1; ++i) {
    delete ios[i];
  }

  return 0;
}
