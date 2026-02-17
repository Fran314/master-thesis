
#include "zknn/utility.h"
#include "libff/algebra/curves/alt_bn128/alt_bn128_pp.hpp"
#include "libff/algebra/curves/public_params.hpp"
#include <cassert>
#include <omp.h>
#include <vector>


template <typename FieldT>
void test_three_square_decomp(int n, int b){
    const int ITER = 10;

    vector<FieldT> x;
    for (int i = 0; i < n; i++) {
        FieldT a = FieldT(4) * FieldT(i) * FieldT(b - i) + FieldT(1);
        x.push_back(a);
    }

    vector<vector<FieldT>> r1(3, vector<FieldT>(x.size()));
    vector<vector<FieldT>> r2(3, vector<FieldT>(x.size()));
    auto t1 = omp_get_wtime();
    for (int k = 0; k < ITER; k++) {
        #pragma omp parallel for
        for (int i = 0; i < x.size(); i++) {
            three_square_decomp(r1[0][i], r1[1][i], r1[2][i], x[i]);
        }
    }
    auto t2 = omp_get_wtime();
    for (int k = 0; k < ITER; k++) {
        #pragma omp parallel for
        for (int i = 0; i < x.size(); i++) {
            FieldT::three_square_decomp_opti(r2[0][i], r2[1][i], r2[2][i], x[i]);
        }
    }
    auto t3 = omp_get_wtime();
    for (int k = 0; k < ITER; k++) {
        #pragma omp parallel for
        for (int i = 0; i < x.size(); i++) {
            FieldT::three_square_decomp_long(r2[0][i], r2[1][i], r2[2][i], x[i]);
        }
    }
    auto t4 = omp_get_wtime();
    printf("%d %d: %f -> %f -> %f\n", n, b, (t2-t1)*1000 / ITER, (t3-t2)*1000 / ITER, (t4-t3)*1000 / ITER);
    assert(r1 == r2);
    printf("All results match.\n");
}

int main() {
    libff::alt_bn128_pp::init_public_params();
    libff::Fr<libff::alt_bn128_pp>::three_square_decomp_opti_init(1ul << 32, "two_square.bin", true);
    for (int i = 16; i <= 20; i += 1) {
        test_three_square_decomp<libff::Fr<libff::alt_bn128_pp>>(1 << i, 1 << 24);
    }
    return 0;
}
