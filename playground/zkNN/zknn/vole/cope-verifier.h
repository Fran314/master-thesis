#ifndef COPE_VER_H__
#define COPE_VER_H__

#include "emp-tool/emp-tool.h"
#include "emp-ot/emp-ot.h"
#include "zknn/utility.h"
#include <cstddef>
#include <emp-tool/utils/block.h>
#include <vector>
#include <libff/libff/algebra/fields/bigint.hpp>

template<typename IO, typename FieldT>
class Cope_Ver {
public:

    IO *io;
    size_t m;
    FieldT delta;
    bool* delta_bool;
    vector<block> K;
	vector<PRG> G0;
	vector<FieldT> gm;

	Cope_Ver(IO *io) {
		delta_bool = nullptr;
        m = FieldT::size_in_bits();
        this->io = io;
	}

	~Cope_Ver() {
		if(delta_bool != nullptr) delete[] delta_bool;
	}

	void initialize() {
		this->delta = FieldT::random_element();
        delta_bool = new bool[m];
		delta_to_bool(delta_bool, delta);

		K.resize(m);
		OTCO<IO> otco(io);
		otco.recv(&K[0], delta_bool, m);

		G0.resize(m);
		for(size_t i=0; i < m; i++) G0[i].reseed(&K[i]);

		gm.resize(m);
        gm[0] = FieldT::one();
		for(size_t i = 1; i < m; ++i) {
			gm[i] = gm[i-1] * FieldT(2);
		}
	}

    // sender
	void extend(FieldT& key) {
		vector<FieldT> w, v;
		w.resize(m);
		v.resize(m);
        
		libff::bigint<FieldT::num_limbs> tmp;
		for(size_t i = 0; i < m; ++i) {
			G0[i].random_data(&tmp, sizeof(tmp));
			w[i] = FieldT(tmp);
		}

		run_time += omp_get_wtime() - s_time;
		io->recv_data(&v[0], m*sizeof(FieldT));
		recv_size += m * sizeof(FieldT);
		s_time = omp_get_wtime();
		
		for(size_t i=0; i<m; i++) {
			if(delta_bool[i] == true) v[i] = w[i] - v[i];
			else v[i] = w[i];
		}

		prm2pr(key, v);
	}

	void extend(vector<FieldT>& key, size_t len) {
		vector<FieldT> w, v;
		w.resize(m * len);
		v.resize(m * len);
		vector<libff::bigint<FieldT::num_limbs>> tmp(len);
		for(size_t i = 0; i < m; i++) {
			G0[i].random_data(&tmp[0], len * sizeof(tmp[0]));
			for(size_t j=0; j < len; j++) w[i * len + j] = FieldT(tmp[j]);
		}

		run_time += omp_get_wtime() - s_time;
		io->recv_data(&v[0], m*len*sizeof(FieldT));
		recv_size += m*len * sizeof(FieldT);
		s_time = omp_get_wtime();

		for(size_t i = 0; i < m; i++) {
			for(size_t j=0; j<len; j++) {
				if(delta_bool[i] == true) v[i * len + j] = w[i * len + j] - v[i * len + j];
				else v[i * len + j] = w[i * len + j];
			}
		}
		key.resize(len);
		prm2pr(key, v);
	}


	void delta_to_bool(bool* bdata, const FieldT& data) {
        auto bd = data.as_bigint();
        for(size_t i=0; i<m; i++) bdata[i] = bd.test_bit(i);
	}

	void prm2pr(FieldT& res, const vector<FieldT>& a) {
		res = FieldT::zero();
		// FieldT g = FieldT::one();
		for(size_t i = 0; i < m; ++i) {
			res += a[i] * gm[i];
		}
	}

	void prm2pr(vector<FieldT>& res, const vector<FieldT>& a) {
        size_t len = res.size();
		for(size_t j = 0; j < len; j++)
        {
            res[j] = FieldT::zero();
            for(size_t i = 0; i < m; ++i) {
			    res[j] += a[i*len + j] * gm[i];
		    }
        }
	}

};
#endif
