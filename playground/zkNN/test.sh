#!/usr/bin/env bash

set -e

# -DCMAKE_POLICY_VERSION_MINIMUM=3.5 overrides the minimum version required for cmake, otherwise it complains
# -DWITH_PROCPS=OFF disables some memory profiling part. It's necessary because it requires `libprocps` but on NixOS only `libprocp2` is available
# cmake -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DWITH_PROCPS=OFF
#
# make -C build -j"$(nproc)"

# # 4. Generate precomputation table (takes ~5 minutes, creates two_square.bin)
# ./build/bin/test_three_square
#
# # 5. Extract model data for CNN tests
# tar -vxjf ./model/data.tar.bz2 -C ./model

# 6. Run tests
./run ./build/bin/test_range # Range proof (quick, no model data needed)
# ./run ./build/bin/test_lookup         # Lookup table test
# ./run ./build/bin/test_resnet18       # ResNet-18 (needs step 5)
# ./run ./build/bin/test_resnet50       # ResNet-50 (needs step 5)
# ./run ./build/bin/test_resnet101      # ResNet-101 (needs step 5)
# ./run ./build/bin/test_vgg11          # VGG-11 (needs step 5)
# ./run ./build/bin/test_transformer    # GPT-2 (needs separate data prep, see model/README.md)
