#ifndef __VOLE_VER_H__
#define __VOLE_VER_H__


#include <emp-tool/emp-tool.h>
#include <emp-ot/emp-ot.h>
#include <emp-tool/utils/block.h>
#include <libff/libff/algebra/fields/bigint.hpp>
#include "zknn/vole/base-vole-verifier.h"
#include "zknn/vole/sps-vole-verifier.h"
#include "zknn/utility.h"

template<typename IO, typename FieldT>
class VOLE_Ver {
public:
    IO* io;
    IO** ios; 
    size_t threads;
    size_t depth;
    size_t leave_n;
    size_t tree_n;
    FieldT delta;
    size_t k;
    Base_VOLE_Ver<IO, FieldT>* bvole = nullptr;
    Sps_VOLE_Ver<IO, FieldT>* spsvole = nullptr;

    vector<FieldT> pre_key_u;
    vector<FieldT> pre_key_sps;
    vector<FieldT> key_u;

    size_t offset;
    size_t end;
    size_t round_total;

    vector<FieldT> pA;


	VOLE_Ver(IO **ios, size_t threads, size_t depth = 12, size_t tree_n = 512, size_t k = 1) {
        this->io = ios[0];
        this->ios = ios;
        this->threads = threads;
        this->depth = depth;
        this->leave_n = 1<<(depth-1);
        this->tree_n = tree_n;
        bvole = new Base_VOLE_Ver<IO, FieldT>(io);
        this->delta = bvole->delta;
        spsvole = new Sps_VOLE_Ver<IO, FieldT>(ios, depth, tree_n, delta, threads);
        this->k = k;

        end = leave_n * tree_n;
        round_total = end - k - tree_n;
	}

    ~VOLE_Ver(){
        if(bvole != nullptr) delete bvole;
    }

    void setup(){
        pA.resize(leave_n * tree_n * k);
        PRG g0;
        g0.reseed(&zero_block);
        vector<libff::bigint<FieldT::num_limbs>> tmp(pA.size());
        g0.random_data(&tmp[0], pA.size() * sizeof(tmp[0]));
        for(size_t i=0; i < pA.size(); i++) pA[i] = FieldT(tmp[i]);

        bvole->triple_gen(pre_key_u, k);
        bvole->triple_gen(pre_key_sps, tree_n);

        extend();
        offset = k + tree_n;
    }

    void extend(){
        spsvole->extend(key_u, pre_key_sps);
        //
        #pragma omp parallel for
        for(size_t i=0; i < leave_n * tree_n; i++) {
            FieldT key_mid = FieldT::zero();
            for(size_t j=0; j < k; j++){
                key_mid += pA[i*k + j] * pre_key_u[j];
            }
            key_u[i] = key_mid + key_u[i];
        }

        // 
        memcpy(&pre_key_u[0], &key_u[0], k * sizeof(FieldT));
        memcpy(&pre_key_sps[0], &key_u[k], tree_n * sizeof(FieldT));
        offset = k + tree_n;
    }


    void extend(vector<FieldT>& key, size_t num) {
        key.resize(num);
        size_t left = end - offset;
        if(num <= left){
            memcpy(&key[0], &key_u[offset], num * sizeof(FieldT));
            offset += num;
            return;
        }

        memcpy(&key[0], &key_u[offset], left * sizeof(FieldT));
        offset += left;

        num -= left;
        size_t round_num = num / round_total;
        for(size_t i=0; i<round_num; i++){
            extend();
            memcpy(&key[i * round_total + left], &key_u[offset], round_total * sizeof(FieldT));
            offset = end;
        }
        num -= round_total * round_num;
        if(num != 0)
        {
            extend();
            memcpy(&key[round_num * round_total + left], &key_u[offset], num * sizeof(FieldT));
            offset += num;
        }
    }
};

#endif