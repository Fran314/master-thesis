#ifndef _VOLE_TRIPLE_VER_H_
#define _VOLE_TRIPLE_VER_H_

#include <vector>
#include <omp.h>
#include "zknn/vole/vole-verifier.h"
using namespace std;

template<typename IO, typename FieldT>
class VoleTripleVer { 
public:
	IO* io;
	IO** ios;
	size_t threads;
	bool extend_initialized;
	FieldT delta;
	VOLE_Ver<IO, FieldT>* vole = nullptr;

	VoleTripleVer (IO **ios, size_t threads) 
	{
		this->io = ios[0];
		this->ios = ios;
		this->threads = threads;
		this->extend_initialized = false;
		vole = new VOLE_Ver<IO, FieldT>(ios, threads);
	}

	~VoleTripleVer () {
		if(vole != nullptr) delete vole;
	}

	void setup() {
		vole->setup();
		this->delta = vole->delta;
		this->extend_initialized = true;
	}

	void extend(FieldT& key) 
	{
		if(extend_initialized == false) return;
		vector<FieldT> vkey(1);
		vole->extend(vkey, 1);
		key = vkey[0];
	}

	void extend(vector<FieldT>& key, size_t num) {
		if(extend_initialized == false) return;
		vole->extend(key, num);
	}
};
#endif
