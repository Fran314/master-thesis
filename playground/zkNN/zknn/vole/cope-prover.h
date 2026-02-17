#ifndef COPE_PRV_H__
#define COPE_PRV_H__

#include "emp-tool/emp-tool.h"
#include "emp-ot/emp-ot.h"
#include <cstddef>
#include <emp-tool/utils/block.h>
#include <vector>
#include <libff/libff/algebra/fields/bigint.hpp>

template<typename IO, typename FieldT>
class Cope_Prv {
public:

    IO *io;
    size_t m;
    vector<block> K;
    vector<PRG> G0, G1;
    vector<FieldT> gm;

	Cope_Prv(IO *io) {
        m = FieldT::size_in_bits();
        this->io = io;
	}

	void initialize() {
        K.resize(2 * m);
        PRG prg;
        prg.random_block(&K[0], 2*m);

		OTCO<IO> otco(io);
		otco.send(&K[0], &K[0]+m, m);

        G0.resize(m);
        G1.resize(m);
		for(size_t i=0; i < m; i++) G0[i].reseed(&K[i]);
        for(size_t i=0; i < m; i++) G1[i].reseed(&K[i+m]);

        gm.resize(m);
        gm[0] = FieldT::one();
		for(size_t i = 1; i < m; i++) {
			gm[i] = gm[i-1] * FieldT(2);
		}
	}

    // recver
	void extend(FieldT& mac, const FieldT& u) {
		vector<FieldT> w0, w1, tau;
        w0.resize(m);
        w1.resize(m);
        tau.resize(m);
        libff::bigint<FieldT::num_limbs> tmp;
		for(size_t i = 0; i < m; ++i) {
			G0[i].random_data(&tmp, sizeof(tmp));
			w0[i] = FieldT(tmp);
            G1[i].random_data(&tmp, sizeof(tmp));
			w1[i] = FieldT(tmp);
            tau[i] = w1[i] - w0[i] - u;
		}

		io->send_data(&tau[0], m*sizeof(FieldT));
		prm2pr(mac, w0);
	}

    void extend(vector<FieldT>& mac, const vector<FieldT>& u, size_t len) {
        vector<FieldT> w0, w1, tau;
        w0.resize(m * len);
        w1.resize(m * len);
        tau.resize(m * len);

        vector<libff::bigint<FieldT::num_limbs>> tmp(len);
        for(size_t i=0; i<m; i++)
        {
            G0[i].random_data(&tmp[0], len * sizeof(tmp[0]));
            for(size_t j=0; j<len; j++) w0[i * len + j] = FieldT(tmp[j]);
            G1[i].random_data(&tmp[0], len * sizeof(tmp[0]));
            for(size_t j=0; j<len; j++) w1[i * len + j] = FieldT(tmp[j]);
            for(size_t j=0; j<len; j++) tau[i * len + j] = w1[i * len + j] - w0[i * len + j] - u[j];
        }
        io->send_data(&tau[0], m*len*sizeof(FieldT));
        mac.resize(len);
        prm2pr(mac, w0);
    }

    void prm2pr(FieldT& res, const vector<FieldT>& a) {
		res = FieldT::zero();
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
