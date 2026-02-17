# zkNN: A Efficient and Extensible ZKP framework for neural networks

This library is an efficient and extensible ZKP framework for neural networks. It contains source code of the paper **ZKNN**

## Dependency Statement

We utilize [libff](https://github.com/scipr-lab/libff.git) to provide the finite field operations.

We use [OpenMP](https://www.openmp.org) to implement multi-core parallelism for our scheme.

We refer to [emp-toolkit](https://github.com/emp-toolkit) to provide basic MPC components, such as OT.

## Setup

Our experiments are finished in the following setup.

1. Ubuntu 20.04
2. g++ 7.5.0
3. AMD Ryzen 3700X 8-Core CPU 

## Neural Networks Models and Dataset

The CNN models we tested include VGG-11, ResNet-50, and ResNet-101. The Transformer model we tested is GPT-2. 

For CNNs, we use dataset [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
For GPT-2, use the [dataset](https://github.com/openai/gpt-2-output-dataset) from OpenAI.

Details on how to train, use, and evaluate these models in this library is described [here](./model/README.md) (./model/README.md).


## Build

Install dependent libraries
```
apt install build-essential libboost-all-dev cmake libgmp3-dev libssl-dev libprocps-dev pkg-config libsodium-dev
```

Install emp-toolkit.
```
wget https://raw.githubusercontent.com/emp-toolkit/emp-readme/master/scripts/install.py
python3 install.py --deps --tool --ot --zk
```

After that, we build our library.
```
cmake -B build
make -C build -j8
```

Pre-computation preparation for two square decomposition. (around 10 mins). After this step, make sure **two_square.bin** are generated.
```
./build/bin/test_three_square
```

Data preparation.

The steps of data preparation (CNNs, Transformers) are described [here](./model/README.md) (./model/README.md).
```
# For a quick start (not include Transformers)
tar -vxjf ./model/data.tar.bz2 -C ./model
```

## Run

For example, run a test of the range proof.
Test on one machine., run:
```
./run ./build/bin/test_range
```

For example, to run a test of the ResNet-50 proof, run:

```
# this step should be performed after data preparation, details see (./model/README.md).
# or a quick start. (not include Transformers)
tar -vxjf ./model/data.tar.bz2 -C ./model
./run ./build/bin/test_resnet50
```

Test on two machine. Modify IP address in File: 
```
# ./test/range.cpp
main()
{
    ...

    ios[i] = new NetIO(party == ALICE?nullptr:"XXX.XXX.XXX.XXX",port+i);   # prover's IP address 

    ...
}
```
Then, run:
```
./build/bin/test_range 1 [port]   # prover 
./build/bin/test_range 2 [port]   # verifier
```


