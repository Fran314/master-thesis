#ifndef __VOLE_PRV_H__
#define __VOLE_PRV_H__

#include <cstddef>
#include <cstring>
#include <emp-tool/emp-tool.h>
#include <emp-ot/emp-ot.h>
#include <emp-tool/utils/block.h>
#include <libff/libff/algebra/fields/bigint.hpp>
#include "zknn/vole/base-vole-prover.h"
#include "zknn/vole/sps-vole-prover.h"
#include "zknn/utility.h"

template<typename IO, typename FieldT>
class VOLE_Prv {
public:
    IO* io;
    IO** ios;
    size_t threads;
    size_t depth;
    size_t leave_n;
	size_t tree_n;
    size_t k;
	Base_VOLE_Prv<IO, FieldT>* bvole = nullptr;
    Sps_VOLE_Prv<IO, FieldT>* spsvole = nullptr;

    vector<FieldT> pre_mac_u;
    vector<FieldT> pre_u;
    vector<FieldT> pre_mac_sps;
    vector<FieldT> pre_sps;
    vector<FieldT> mac_u;
    vector<FieldT> u;

    size_t offset;
    size_t end;
    size_t round_total;

    vector<FieldT> pA;

	VOLE_Prv(IO **ios, size_t threads, size_t depth = 12, size_t tree_n = 512, size_t k = 1) {
        this->io = ios[0];
        this->ios = ios;
        this->threads = threads;
        this->depth = depth;
        this->leave_n = 1<<(depth-1);
		this->tree_n = tree_n;
        this->k = k;
		bvole = new Base_VOLE_Prv<IO, FieldT>(io);
        spsvole = new Sps_VOLE_Prv<IO, FieldT>(ios, depth, tree_n, threads);

        end = leave_n * tree_n;
        round_total = end - k - tree_n;
	}

    ~VOLE_Prv(){
        if(bvole != nullptr) delete bvole;
    }

    void setup(){
        pA.resize(leave_n * tree_n * k);
        PRG g0;
        g0.reseed(&zero_block);
        vector<libff::bigint<FieldT::num_limbs>> tmp(pA.size());
        g0.random_data(&tmp[0], pA.size() * sizeof(tmp[0]));
        for(size_t i=0; i < pA.size(); i++) pA[i] = FieldT(tmp[i]);

        bvole->triple_gen(pre_mac_u, pre_u, k);
        bvole->triple_gen(pre_mac_sps, pre_sps, tree_n);

        extend();
        offset = k + tree_n;
    }

    void extend(){
        spsvole->extend(mac_u, u, pre_mac_sps, pre_sps);

        #pragma omp parallel for
        for(size_t i=0; i < leave_n * tree_n; i++) {
            FieldT mac_mid = FieldT::zero();
            FieldT mid = FieldT::zero();
            for(size_t j=0; j < k; j++){
                mac_mid += pA[i*k + j] * pre_mac_u[j];
                mid += pA[i*k + j] * pre_u[j];
            }
            mac_u[i] = mac_mid + mac_u[i];
            u[i] = mid + u[i];
        }

        memcpy(&pre_mac_u[0], &mac_u[0], k * sizeof(FieldT));
        memcpy(&pre_u[0], &u[0], k * sizeof(FieldT));
        memcpy(&pre_mac_sps[0], &mac_u[k], tree_n * sizeof(FieldT));
        memcpy(&pre_sps[0], &u[k], tree_n * sizeof(FieldT));
        offset = k + tree_n;
    }

    void extend(vector<FieldT>& mac, vector<FieldT>& x, size_t num)
    {
        mac.resize(num);
        x.resize(num);
        size_t left = end - offset;
        if(num <= left){
            memcpy(&mac[0], &mac_u[offset], num * sizeof(FieldT));
            memcpy(&x[0], &u[offset], num * sizeof(FieldT));
            offset += num;
            return;
        }

        memcpy(&mac[0], &mac_u[offset], left * sizeof(FieldT));
        memcpy(&x[0], &u[offset], left * sizeof(FieldT));
        offset += left;

        num -= left;
        size_t round_num = num / round_total;
        for(size_t i=0; i<round_num; i++){
            extend();
            memcpy(&mac[i * round_total + left], &mac_u[offset], round_total * sizeof(FieldT));
            memcpy(&x[i * round_total + left], &u[offset], round_total * sizeof(FieldT));
            offset = end;
        }
        num -= round_total * round_num;
        if(num != 0)
        {
            extend();
            memcpy(&mac[round_num * round_total + left], &mac_u[offset], num * sizeof(FieldT));
            memcpy(&x[round_num * round_total + left], &u[offset], num * sizeof(FieldT));
            offset += num;
        }
    }



};

#endif