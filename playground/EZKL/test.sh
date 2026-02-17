#!/usr/bin/env bash

set -e

rm -f *.onnx *.onnx.data *.json *.key *.srs *.compiled

# Export model and generate dummy input data
python3 network.py

ezkl gen-settings -M network.onnx

# Not necessary but optimizes and speeds up the process
ezkl calibrate-settings -M network.onnx -D input.json

ezkl compile-circuit -M network.onnx

# Generate dummy SRS for testing
LOGROWS=$(jq '.run_args.logrows' settings.json)
ezkl gen-srs --srs-path kzg.srs --logrows "$LOGROWS"

# Setup (generate proving and verification keys)
ezkl setup --srs-path kzg.srs

ezkl gen-witness -D input.json

ezkl prove --srs-path kzg.srs

ezkl verify --srs-path kzg.srs