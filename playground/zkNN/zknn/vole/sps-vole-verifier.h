#ifndef SPS_VOLE_VER_H__
#define SPS_VOLE_VER_H__

#include <emp-tool/emp-tool.h>
#include <emp-ot/emp-ot.h>
#include <emp-tool/utils/block.h>
#include <libff/libff/algebra/fields/bigint.hpp>
#include "zknn/vole/base-vole-verifier.h"
#include "zknn/utility.h"

template<typename IO, typename FieldT>
class Sps_VOLE_Ver {
public:
    IO* io;
    IO** ios;
    size_t threads;
    size_t depth;
    size_t leave_n;
    size_t tree_n;
    FieldT delta;
    vector<PRG> g;
    PRG prg;
    Base_VOLE_Ver<IO, FieldT>* bvole = nullptr;

	Sps_VOLE_Ver(IO **ios, size_t depth, size_t tree_n, const FieldT delta, size_t threads) {
        this->io = ios[0];
        this->ios = ios;
        this->threads = threads;
        this->depth = depth;
        this->leave_n = 1<<(depth-1);
        this->tree_n = tree_n;
        this->delta = delta;
        this->g.resize(tree_n * leave_n);
	}

    void extend(vector<FieldT>& key, const vector<FieldT>& key_a)
    {
        key.resize(tree_n * leave_n);

        vector<FieldT> key_beta(tree_n);
        vector<FieldT> a_(tree_n);

		run_time += omp_get_wtime() - s_time;
		io->recv_data(&a_[0], tree_n * sizeof(FieldT));
		recv_size += tree_n * sizeof(FieldT);
		s_time = omp_get_wtime();

        for(size_t i=0; i<tree_n; i++){
            key_beta[i] = key_a[i] + a_[i] * delta;
        }

        vector<block> seed(tree_n);
        prg.random_block(&seed[0], tree_n);

        vector<vector<FieldT>> v(tree_n);
        vector<vector<block>> K0(tree_n);
        vector<vector<block>> K1(tree_n);
        #pragma omp parallel for
        for(size_t i=0; i<tree_n; i++){
            ggm_tree_gen(K0[i], K1[i], v[i], seed[i], i);
        }

        run_time += omp_get_wtime() - s_time;
        vector<OTCO<IO>*> otcos(threads);
        for(size_t i=0; i<threads; i++) {
            otcos[i] = new OTCO<IO>(ios[i+1]);
        }
        io->flush();
        #pragma omp parallel for
        for(size_t i=0; i<tree_n; i++){
            size_t tid = omp_get_thread_num();
            otcos[tid]->send(&K0[i][0], &K1[i][0], depth - 1);
            ios[tid+1]->flush();
        }
        for(size_t i=0; i<threads; i++) {
			delete otcos[i];
		}
        s_time = omp_get_wtime();
		recv_size += tree_n * (depth - 1) * sizeof(FieldT);

        for(size_t i=0; i<tree_n; i++){
            FieldT d = key_beta[i];
            for(size_t j=0; j<leave_n; j++) d -= v[i][j];
            io->send_data(&d, sizeof(FieldT));
            memcpy(&key[i*leave_n], &v[i][0], leave_n * sizeof(FieldT));
        }

    }


    // generate GGM tree from the top
	void ggm_tree_gen(vector<block>& K0, vector<block>& K1, vector<FieldT>& v, block seed, size_t tree_idx) {
        K0.resize(depth-1);
        K1.resize(depth-1);
        v.resize(leave_n);

        vector<block> ggm_tree_mem(leave_n);
        TwoKeyPRP prp(makeBlock(0, 0), makeBlock(0, 1));

        block* ggm_tree = (block*)&ggm_tree_mem[0];
		prp.node_expand_1to2(ggm_tree, seed);
		K0[0] = ggm_tree[0];
		K1[0] = ggm_tree[1];
		for(size_t h = 1; h < depth-1; ++h) {
			K0[h] = K1[h] = zero_block;
			int sz = 1<<h;
			for(int i = sz-2; i >=0; i-=2) {
				prp.node_expand_2to4(&ggm_tree[i*2], &ggm_tree[i]);
				K0[h] = K0[h] ^ ggm_tree[i*2];
				K0[h] = K0[h] ^ ggm_tree[i*2+2];
				K1[h] = K1[h] ^ ggm_tree[i*2+1];
				K1[h] = K1[h] ^ ggm_tree[i*2+3];
			}
		}

        vector<libff::bigint<FieldT::num_limbs>> tmp(leave_n);
        for(size_t i=0; i<leave_n; i++) 
        {
            g[tree_idx * leave_n + i].reseed(ggm_tree+i);
            g[tree_idx * leave_n + i].random_data(&tmp[i], sizeof(tmp[i]));
            v[i] = FieldT(tmp[i]);
        }
	}

};

#endif