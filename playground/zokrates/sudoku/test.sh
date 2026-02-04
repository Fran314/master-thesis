#!/usr/bin/env bash

set -e

# compile
zokrates compile -i sudoku_checker.zok

# perform the setup phase
zokrates setup

# execute the program
zokrates compute-witness --abi --stdin <./sudoku_checker.input.json

# generate a proof of computation
zokrates generate-proof

# export a solidity verifier
zokrates export-verifier

# or verify natively
zokrates verify
