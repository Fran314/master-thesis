#!/usr/bin/env bash

set -e

circom sudoku.circom --r1cs --c --wasm --sym

# Compile both with nodejs and with c++ just to prove that they both work

# nodejs
node sudoku_js/generate_witness.js sudoku_js/sudoku.wasm sudoku.input.json witness.wtns

# c++
pushd sudoku_cpp
make
popd
./sudoku_cpp/sudoku sudoku.input.json witness.wtns

snarkjs powersoftau new bn128 12 pot12_0000.ptau -v

snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau --name="First contribution" -v

snarkjs powersoftau prepare phase2 pot12_0001.ptau pot12_final.ptau -v

snarkjs groth16 setup sudoku.r1cs pot12_final.ptau sudoku_0000.zkey

snarkjs zkey contribute sudoku_0000.zkey sudoku_0001.zkey --name="1st Contributor Name" -v

snarkjs zkey export verificationkey sudoku_0001.zkey verification_key.json

snarkjs groth16 prove sudoku_0001.zkey witness.wtns proof.json public.json

snarkjs groth16 verify verification_key.json public.json proof.json
