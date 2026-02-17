#ifndef _VOLE_TRIPLE_PRV_H_
#define _VOLE_TRIPLE_PRV_H_

#include <cstddef>
#include <vector>
#include <omp.h>
#include "zknn/vole/vole-prover.h"

using namespace std;

template<typename IO, typename FieldT>
class VoleTriplePrv { 
public:
	IO* io;
	IO** ios;
	size_t threads;
	bool extend_initialized;
	VOLE_Prv<IO, FieldT>* vole = nullptr;

	VoleTriplePrv (IO **ios, size_t threads)
	{
		this->io = ios[0];
		this->ios = ios;
		this->threads = threads;
		this->extend_initialized = false;
		vole = new VOLE_Prv<IO, FieldT>(ios, threads);
	}

	~VoleTriplePrv () {
		if(vole != nullptr) delete vole;
	}
	
	void setup() 
	{
		vole->setup();
		extend_initialized = true;
	}

	void extend(FieldT& mac, FieldT& x) 
	{
		if(extend_initialized == false) return ;
		vector<FieldT> vmac(1);
		vector<FieldT> vx(1);
		vole->extend(vmac, vx, 1);
		mac = vmac[0];
		x  = vx[0];
	}

	void extend(vector<FieldT>& mac, vector<FieldT>& x, size_t num) 
	{
		if(extend_initialized == false) return ;
		vole->extend(mac, x, num);
	}
};
#endif
