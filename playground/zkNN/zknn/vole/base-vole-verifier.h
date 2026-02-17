#ifndef BASE_VOLE_VER_H__
#define BASE_VOLE_VER_H__

#include "emp-tool/emp-tool.h"
#include "zknn/vole/cope-verifier.h"
#include "zknn/utility.h"

template<typename IO, typename FieldT>
class Base_VOLE_Ver {
public:

	size_t m;
	IO *io;
	Cope_Ver<IO, FieldT>* cope = nullptr;
    FieldT delta;

	Base_VOLE_Ver(IO *io) {
        m = FieldT::size_in_bits();
        this->io = io;
        cope = new Cope_Ver<IO, FieldT>(io);
        cope->initialize();
        this->delta = cope->delta;
	}

    ~Base_VOLE_Ver() {
        if(cope != nullptr) delete cope;
	}

    void triple_gen(vector<FieldT>& key, size_t num) {
        key.resize(num);
        FieldT key_a, a;
        cope->extend(key, num);
        cope->extend(key_a);
        check(key, key_a, num);
	}

	void check(const vector<FieldT>& key, FieldT key_a, size_t num) {
        FieldT seed = FieldT::random_element();
		io->send_data(&seed, sizeof(FieldT));
        vector<FieldT> chi;
		uni_hash_coeff_gen(chi, seed, num);
        FieldT y;
        vector_inn_prdt(y, key, chi);
        y += key_a;
        FieldT xz[2];

		run_time += omp_get_wtime() - s_time;
		io->recv_data(xz, 2*sizeof(FieldT));
		recv_size += 2 * sizeof(FieldT);
		s_time = omp_get_wtime();

        if(y != xz[0] + xz[1] * delta){
            printf("base vole error!\n");
        }
	}
};

#endif