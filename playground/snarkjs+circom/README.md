# Snarkjs + circom

Playground to test ZKP for R1CS with snarkjs + circom

## Install

### Generic

Refer to the [`snarkjs`](https://github.com/iden3/snarkjs) and
[`circom`](https://docs.circom.io/getting-started/installation/) documentation
for installing on generic systems

### NixOS

There is already a `flake.nix`. Just run

```bash
nix develop
```

which will put you in a shell with the binaries `circom` and `snarkjs`

## Usage

This is just a short summary of the instructions given
[here](https://docs.circom.io/getting-started/writing-circuits/)

It really is just a summary of the instructions, I do not claim to understand
precisely the commands just yet

1. Create a circom circuit ending in `.circom` (eg `sudoku.circom`)
2. Compile it to a `.r1cs` and obtain a program to compute witnesses by running

    ```bash
    circom your-circuit.circom --r1cs --wasm --sym --c
    ```

    This will both compile the circuit into an `.r1cs` and will also produce a
    program to compute the witness

    `--wasm` creates a witness-computer in `js`, `--c` creates a
    witness-computer in `c++`

    `--sym` is used to create an auxiliary file for displaying the R1CS if you
    want to display it

3. write the input into a `.json` file

4. Compute the witness.

    To do it with js, run

    ```bash
    node your-circuit_js/generate_witness.js your-circuit_js/your-circuit.wasm your-circuit.input.json witness.wtns
    ```

    To do this with c++, go into the generated folder and run

    ```bash
    make
    ```

    Then run

    ```bash
    ./your-circuit your-circuit.input.json witness.wtns
    ```

5. Generate the keys

    (This uses the bn128 elliptic curve, and the parameter `12` depends on how
    big the R1CS is)

    ```bash
    snarkjs powersoftau new bn128 12 pot12_0000.ptau -v
    ```

    ```bash
    snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau --name="First contribution" -v
    ```

    ```bash
    snarkjs powersoftau prepare phase2 pot12_0001.ptau pot12_final.ptau -v
    ```

    ```bash
    snarkjs groth16 setup your-circuit.r1cs pot12_final.ptau your-circuit_0000.zkey
    ```

    ```bash
    snarkjs zkey contribute your-circuit_0000.zkey your-circuit_0001.zkey --name="1st Contributor Name" -v
    ```

    ```bash
    snarkjs zkey export verificationkey your-circuit_0001.zkey verification_key.json
    ```

6. Generate a proof

    ```bash
    snarkjs groth16 prove your-circuit_0001.zkey witness.wtns proof.json public.json
    ```

7. Verify a proof

    ```bash
    snarkjs groth16 verify verification_key.json public.json proof.json
    ```
