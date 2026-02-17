#include <iostream>
#include <zknn/utility.h>
#include <zknn/zknn-prover.h>
#include <zknn/zknn-verifier.h>
#include <libff/libff/algebra/curves/alt_bn128/alt_bn128_pp.hpp>
#include <fstream>
#include <system_error>
#include <vector>
#include <time.h>
#include "emp-tool/emp-tool.h"
using namespace emp;
using namespace std;

using namespace libff;

int port, party;
const int threads = 128;

extern size_t send_size;
extern size_t recv_size;
extern double run_time;



template<typename FieldT>
void test_lookup_prover(NetIO *ios[threads+1], int input_sz_lg) 
{
    FpOSTriplePrv<NetIO, FieldT>* fp = new FpOSTriplePrv<NetIO, FieldT>(ios, threads);
    size_t num = 1<<input_sz_lg;
	vector<vector<FieldT>> v;
	vector<vector<FieldT>> mac_v;
	vector<vector<FieldT>> T;
	size_t m = 64;
	for(size_t i=0; i<m; i++)
	{
		vector<FieldT> exp(1);
		exp[0] = FieldT(i) + FieldT(64) * (FieldT(2) ^ i);
		T.push_back(exp);
	}

	vector<FieldT> exp(num);
	vector<FieldT> mac_exp(num);
	for(size_t i=0; i<num; i++)
	{
		exp[0] = FieldT(i % m) + FieldT(64) * (FieldT(2) ^ (i%m));
	}
	fp->authenticated_val_input(mac_exp, exp, num);

	for(size_t i=0; i<num; i++)
	{
		vector<FieldT> vv(1);
		vector<FieldT> mac_vv(1);
		vv[0] = exp[i];
		mac_vv[0] = mac_exp[i];
		v.push_back(vv);
		mac_v.push_back(mac_vv);
	}

	printf("prover lookup proof begin\n");

	recv_size = 0;
	s_time = omp_get_wtime();
	run_time = 0;

	fp->zkp_lookup(mac_v, v, T);
	fp->batch_check(true, true, true);

	run_time += omp_get_wtime() - s_time;
	printf("prover Run time: %f s\n", run_time);
}

template<typename FieldT>
void test_lookup_verifier(NetIO *ios[threads+1], int input_sz_lg)
{
    FpOSTripleVer<NetIO, FieldT>* fv = new FpOSTripleVer<NetIO, FieldT>(ios, threads);
    size_t num = 1<<input_sz_lg;
	vector<vector<FieldT>> key_v;
	vector<vector<FieldT>> T;
	size_t m = 64;
	for(size_t i=0; i<m; i++)
	{
		vector<FieldT> exp(1);
		exp[0] = FieldT(i) + FieldT(64) * (FieldT(2) ^ i);
		T.push_back(exp);
	}

	vector<FieldT> key_exp(num);
	fv->authenticated_val_input(key_exp, num);
	for(size_t i=0; i<num; i++)
	{
		vector<FieldT> key_vv(1);
		key_vv[0] = key_exp[i];
		key_v.push_back(key_vv);
	}

	printf("verifier lookup proof begin\n");

	recv_size = 0;
	s_time = omp_get_wtime();
	run_time = 0;

	fv->zkp_lookup(key_v, T);
	bool check = fv->batch_check(true, true, true);
	printf("check: %lu\n", check);

	run_time += omp_get_wtime() - s_time;
	printf("verifier Run time: %f\n", run_time);
	printf("proof size: %f MB\n", (float)recv_size / 1024 / 1024);
}


int main(int argc, char** argv) {
	parse_party_and_port(argv, &party, &port);
	NetIO* ios[threads+1];
	for(int i = 0; i < threads+1; ++i)
	{
		ios[i] = new NetIO(party == ALICE?nullptr:"127.0.0.1",port+i);
		ios[i]->set_nodelay();
	}

	std::cout << std::endl << "------------ circuit zero-knowledge proof test ------------" << std::endl;

	size_t num = 18;
	alt_bn128_pp::init_public_params();

	if(party == ALICE) test_lookup_prover<Fr<alt_bn128_pp>>(ios, num);
	else test_lookup_verifier<Fr<alt_bn128_pp>>(ios, num);

	// free ios
	for(int i = 0; i < threads+1; ++i) {
		delete ios[i];
	}

	return 0;
}
