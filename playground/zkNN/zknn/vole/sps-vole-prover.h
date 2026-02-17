#ifndef SPS_VOLE_PRV_H__
#define SPS_VOLE_PRV_H__

#include <emp-tool/emp-tool.h>
#include <emp-ot/emp-ot.h>
#include <libff/libff/algebra/fields/bigint.hpp>
#include <thread>
#include "zknn/vole/base-vole-prover.h"
#include "zknn/utility.h"

template<typename IO, typename FieldT>
class Sps_VOLE_Prv {
public:
    IO* io;
	IO** ios;
	size_t threads;
    size_t depth;
    size_t leave_n;
	size_t tree_n;
	vector<PRG> g;

	Sps_VOLE_Prv(IO **ios, size_t depth, size_t tree_n, size_t threads) {
        this->io = ios[0];
		this->ios = ios;
		this->threads = threads;
        this->depth = depth;
        this->leave_n = 1<<(depth-1);
		this->tree_n = tree_n;
		g.resize(leave_n * tree_n);
	}

	size_t get_pos(bool* b) {
		size_t choice_pos = 0;
		for(size_t i = 0; i < depth-1; ++i) {
			choice_pos<<=1;
			if(!b[i])
				choice_pos +=1;
		}
		return choice_pos;
	}

	void extend(vector<FieldT>& mac, vector<FieldT>& u, const vector<FieldT>& mac_a, const vector<FieldT>& a)
	{

		mac.resize(tree_n * leave_n);
		u.resize(tree_n * leave_n);
		
		vector<FieldT> mac_beta(tree_n);
		vector<FieldT> beta(tree_n);
		vector<FieldT> a_(tree_n);
		for(size_t i=0; i < tree_n; i++){
			beta[i] = FieldT::random_element();
			a_[i] = beta[i] - a[i];
			mac_beta[i] = mac_a[i];
		}
		io->send_data(&a_[0], tree_n * sizeof(FieldT));

		bool* b = new bool[depth - 1];
		for(size_t i=0; i < depth - 1; i++) b[i] = false;

		vector<vector<FieldT>> v(tree_n);
        vector<vector<block>> K(tree_n);

		run_time += omp_get_wtime() - s_time;
        vector<OTCO<IO>*> otcos(threads);
        for(size_t i=0; i<threads; i++) {
            otcos[i] = new OTCO<IO>(ios[i+1]);
        }

		io->flush();
		#pragma omp parallel for
		for(size_t i=0; i<tree_n; i++){
			size_t tid = omp_get_thread_num();
			K[i].resize(depth - 1);
			otcos[tid]->recv(&K[i][0], b, depth - 1);
			ios[tid+1]->flush();
		}
		for(size_t i=0; i<threads; i++) {
			delete otcos[i];
		}
		
		s_time = omp_get_wtime();
		recv_size += tree_n * (depth - 1) * sizeof(FieldT);

		#pragma omp parallel for
		for(size_t i=0; i<tree_n; i++){
			ggm_tree_recover(v[i], b, K[i], i);
		}

		for(size_t i=0; i<tree_n; i++){
			memcpy(&mac[i*leave_n], &v[i][0], leave_n * sizeof(FieldT));
			FieldT d;

			run_time += omp_get_wtime() - s_time;
			io->recv_data(&d, sizeof(FieldT));
			recv_size += sizeof(FieldT);
			s_time = omp_get_wtime();
			
			size_t alpha = get_pos(b);
			for(size_t j=0; j < leave_n; j++) {
				if(j != alpha) d += v[i][j];
			}
			mac[i*leave_n + alpha] = mac_beta[i] - d;
			for(size_t j=0; j < leave_n; j++) u[i*leave_n + j] = FieldT::zero();
			u[i*leave_n + alpha] = beta[i];
        }
	}
	
    // generate GGM tree from the top
	void ggm_tree_recover(vector<FieldT>& v, bool* b, const vector<block>& K, size_t tree_idx) {
		v.resize(leave_n);
		
        size_t to_fill_idx = 0;
        vector<block> ggm_tree_mem(leave_n);
        block* ggm_tree = (block*)&ggm_tree_mem[0];
		TwoKeyPRP prp(zero_block, makeBlock(0, 1));

        for(size_t i = 1; i < depth; ++i) {
			to_fill_idx = to_fill_idx * 2;
			ggm_tree[to_fill_idx] = ggm_tree[to_fill_idx+1] = zero_block;
			if(b[i-1] == false) {
				layer_recover(i, 0, to_fill_idx, K[i-1], &prp, ggm_tree);
				to_fill_idx += 1;
			} else layer_recover(i, 1, to_fill_idx+1, K[i-1], &prp, ggm_tree);
		}

        vector<libff::bigint<FieldT::num_limbs>> tmp(leave_n);
        for(size_t i=0; i<leave_n; i++) 
        {
            g[tree_idx * leave_n + i].reseed(ggm_tree+i);
            g[tree_idx * leave_n + i].random_data(&tmp[i], sizeof(tmp[i]));
            v[i] = FieldT(tmp[i]);
        }
	}

    void layer_recover(int depth, int lr, int to_fill_idx, block sum, TwoKeyPRP *prp, block* ggm_tree) {
		int layer_start = 0;
		int item_n = 1<<depth;
		block nodes_sum = zero_block;
		int lr_start = lr==0?layer_start:(layer_start+1);
		
		for(int i = lr_start; i < item_n; i+=2)
			nodes_sum = nodes_sum ^ ggm_tree[i];
		ggm_tree[to_fill_idx] = nodes_sum ^ sum;
		if((size_t)depth == this->depth-1) return;
		for(int i = item_n-2; i >= 0; i-=2)
			prp->node_expand_2to4(&ggm_tree[i*2], &ggm_tree[i]);
	}
};


#endif