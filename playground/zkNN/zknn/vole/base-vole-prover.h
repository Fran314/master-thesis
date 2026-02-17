#ifndef BASE_VOLE_PRV_H__
#define BASE_VOLE_PRV_H__

#include "emp-tool/emp-tool.h"
#include "zknn/vole/cope-prover.h"
#include "zknn/utility.h"
#include <cstddef>

template<typename IO, typename FieldT>
class Base_VOLE_Prv {
public:

	size_t m;
	IO *io;
	Cope_Prv<IO, FieldT>* cope = nullptr;

	Base_VOLE_Prv(IO *io) {
        m = FieldT::size_in_bits();
        this->io = io;
        cope = new Cope_Prv<IO, FieldT>(io);
        cope->initialize();
	}

    ~Base_VOLE_Prv() {
        if(cope != nullptr) delete cope;
	}

	void triple_gen(vector<FieldT>& mac, vector<FieldT>& u, size_t num) {
        mac.resize(num);
        u.resize(num);
        FieldT mac_a, a;
        for(size_t i=0; i<num; i++) u[i] = FieldT::random_element();
        cope->extend(mac, u, num);
        cope->extend(mac_a, a);
		check(mac, u, mac_a, a, num);
	}
	
	// receiver check
	void check(const vector<FieldT>& mac, const vector<FieldT>& u, const FieldT& mac_a, const FieldT& a, size_t num) {
		FieldT seed;

		run_time += omp_get_wtime() - s_time;
		io->recv_data(&seed, sizeof(FieldT));
		recv_size += sizeof(FieldT);
		s_time = omp_get_wtime();

		vector<FieldT> chi;
		
		uni_hash_coeff_gen(chi, seed, num);
		FieldT xz[2];
		vector_inn_prdt(xz[0], mac, chi);
		vector_inn_prdt(xz[1], u, chi);
		xz[0] += mac_a;
		xz[1] += a;
		io->send_data(xz, 2*sizeof(FieldT));
	}
};

#endif