/** @file
 *****************************************************************************
 Implementation of arithmetic in the finite field F[p], for prime p of fixed length.
 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#ifndef FP_TCC_
#define FP_TCC_
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <gmp.h>
#include <limits>
#include <fstream>

#include <libff/algebra/fields/field_utils.hpp>
#include <libff/algebra/fields/fp_aux.tcc>

#include <omp.h>
#include <random>
#include <utility>

namespace libff {

template<mp_size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::mul_reduce(const bigint<n> &other)
{
    /* stupid pre-processor tricks; beware */
// #if defined(__x86_64__) && defined(USE_ASM)

    if (n == 2)
    {
        mp_limb_t res[2*n];
        mp_limb_t c0, c1;
        COMBA_2_BY_2_MUL(c0, c1, res, this->mont_repr.data, other.data);

        mp_limb_t k;
        mp_limb_t tmp1, tmp2, tmp3;
        // REDUCE_6_LIMB_PRODUCT(k, tmp1, tmp2, tmp3, inv, res, modulus.data);
        REDUCE_4_LIMB_PRODUCT(k, tmp1, tmp2, tmp3, inv, res, modulus.data);

        /* subtract t > mod */
        __asm__
            ("/* check for overflow */        \n\t"
             MONT_CMP(8)
             MONT_CMP(0)

             "/* subtract mod if overflow */  \n\t"
             "subtract%=:                     \n\t"
             MONT_FIRSTSUB
             MONT_NEXTSUB(8)

             "done%=:                         \n\t"
             :
             : [tmp] "r" (res+n), [M] "r" (modulus.data)
             : "cc", "memory", "%rax");
        mpn_copyi(this->mont_repr.data, res+n, n);
    }
    else if (n == 3)
    { // Use asm-optimized Comba multiplication and reduction
        mp_limb_t res[2*n];
        mp_limb_t c0, c1, c2;
        COMBA_3_BY_3_MUL(c0, c1, c2, res, this->mont_repr.data, other.data);

        mp_limb_t k;
        mp_limb_t tmp1, tmp2, tmp3;
        REDUCE_6_LIMB_PRODUCT(k, tmp1, tmp2, tmp3, inv, res, modulus.data);

        /* subtract t > mod */
        __asm__
            ("/* check for overflow */        \n\t"
             MONT_CMP(16)
             MONT_CMP(8)
             MONT_CMP(0)

             "/* subtract mod if overflow */  \n\t"
             "subtract%=:                     \n\t"
             MONT_FIRSTSUB
             MONT_NEXTSUB(8)
             MONT_NEXTSUB(16)
             "done%=:                         \n\t"
             :
             : [tmp] "r" (res+n), [M] "r" (modulus.data)
             : "cc", "memory", "%rax");
        mpn_copyi(this->mont_repr.data, res+n, n);
    }
    else if (n == 4)
    { // use asm-optimized "CIOS method"

        mp_limb_t tmp[n+1];
        mp_limb_t T0=0, T1=1, cy=2, u=3; // TODO: fix this

        __asm__ (MONT_PRECOMPUTE
                 MONT_FIRSTITER(1)
                 MONT_FIRSTITER(2)
                 MONT_FIRSTITER(3)
                 MONT_FINALIZE(3)
                 MONT_ITERFIRST(1)
                 MONT_ITERITER(1, 1)
                 MONT_ITERITER(1, 2)
                 MONT_ITERITER(1, 3)
                 MONT_FINALIZE(3)
                 MONT_ITERFIRST(2)
                 MONT_ITERITER(2, 1)
                 MONT_ITERITER(2, 2)
                 MONT_ITERITER(2, 3)
                 MONT_FINALIZE(3)
                 MONT_ITERFIRST(3)
                 MONT_ITERITER(3, 1)
                 MONT_ITERITER(3, 2)
                 MONT_ITERITER(3, 3)
                 MONT_FINALIZE(3)
                 "/* check for overflow */        \n\t"
                 MONT_CMP(24)
                 MONT_CMP(16)
                 MONT_CMP(8)
                 MONT_CMP(0)

                 "/* subtract mod if overflow */  \n\t"
                 "subtract%=:                     \n\t"
                 MONT_FIRSTSUB
                 MONT_NEXTSUB(8)
                 MONT_NEXTSUB(16)
                 MONT_NEXTSUB(24)
                 "done%=:                         \n\t"
                 :
                 : [tmp] "r" (tmp), [A] "r" (this->mont_repr.data), [B] "r" (other.data), [inv] "r" (inv), [M] "r" (modulus.data),
                   [T0] "r" (T0), [T1] "r" (T1), [cy] "r" (cy), [u] "r" (u)
                 : "cc", "memory", "%rax", "%rdx"
        );
        mpn_copyi(this->mont_repr.data, tmp, n);
    }
    else if (n == 5)
    { // use asm-optimized "CIOS method"

        mp_limb_t tmp[n+1];
        mp_limb_t T0=0, T1=1, cy=2, u=3; // TODO: fix this

        __asm__ (MONT_PRECOMPUTE
                 MONT_FIRSTITER(1)
                 MONT_FIRSTITER(2)
                 MONT_FIRSTITER(3)
                 MONT_FIRSTITER(4)
                 MONT_FINALIZE(4)
                 MONT_ITERFIRST(1)
                 MONT_ITERITER(1, 1)
                 MONT_ITERITER(1, 2)
                 MONT_ITERITER(1, 3)
                 MONT_ITERITER(1, 4)
                 MONT_FINALIZE(4)
                 MONT_ITERFIRST(2)
                 MONT_ITERITER(2, 1)
                 MONT_ITERITER(2, 2)
                 MONT_ITERITER(2, 3)
                 MONT_ITERITER(2, 4)
                 MONT_FINALIZE(4)
                 MONT_ITERFIRST(3)
                 MONT_ITERITER(3, 1)
                 MONT_ITERITER(3, 2)
                 MONT_ITERITER(3, 3)
                 MONT_ITERITER(3, 4)
                 MONT_FINALIZE(4)
                 MONT_ITERFIRST(4)
                 MONT_ITERITER(4, 1)
                 MONT_ITERITER(4, 2)
                 MONT_ITERITER(4, 3)
                 MONT_ITERITER(4, 4)
                 MONT_FINALIZE(4)
                 "/* check for overflow */        \n\t"
                 MONT_CMP(32)
                 MONT_CMP(24)
                 MONT_CMP(16)
                 MONT_CMP(8)
                 MONT_CMP(0)

                 "/* subtract mod if overflow */  \n\t"
                 "subtract%=:                     \n\t"
                 MONT_FIRSTSUB
                 MONT_NEXTSUB(8)
                 MONT_NEXTSUB(16)
                 MONT_NEXTSUB(24)
                 MONT_NEXTSUB(32)
                 "done%=:                         \n\t"
                 :
                 : [tmp] "r" (tmp), [A] "r" (this->mont_repr.data), [B] "r" (other.data), [inv] "r" (inv), [M] "r" (modulus.data),
                   [T0] "r" (T0), [T1] "r" (T1), [cy] "r" (cy), [u] "r" (u)
                 : "cc", "memory", "%rax", "%rdx"
        );
        mpn_copyi(this->mont_repr.data, tmp, n);
    }
    else
// #endif
    {
        mp_limb_t res[2*n];
        mpn_mul_n(res, this->mont_repr.data, other.data, n);

        /*
          The Montgomery reduction here is based on Algorithm 14.32 in
          Handbook of Applied Cryptography
          <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.
         */
        for (size_t i = 0; i < n; ++i)
        {
            mp_limb_t k = inv * res[i];
            /* calculate res = res + k * mod * b^i */
            mp_limb_t carryout = mpn_addmul_1(res+i, modulus.data, n, k);
            carryout = mpn_add_1(res+n+i, res+n+i, n-i, carryout);
            assert(carryout == 0);
        }

        if (mpn_cmp(res+n, modulus.data, n) >= 0)
        {
            const mp_limb_t borrow = mpn_sub(res+n, res+n, n, modulus.data, n);
#ifndef NDEBUG
            assert(borrow == 0);
#else
            UNUSED(borrow);
#endif
        }

        mpn_copyi(this->mont_repr.data, res+n, n);
    }
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>::Fp_model(const bigint<n> &b)
{
    mpn_copyi(this->mont_repr.data, Rsquared.data, n);
    mul_reduce(b);
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>::Fp_model(const mpz_t &z)
{
    if(mpz_sgn(z)>=0)
    {
        *this = Fp_model(bigint<n>(z));
    }
    else
    {
        mpz_t nz;
        mpz_init(nz);
        mpz_neg(nz, z);
        *this = Fp_model(bigint<n>(nz));
        *this = -(*this);
        mpz_clear(nz);
    }
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>::Fp_model(const long x, const bool is_unsigned)
{
    static_assert(std::numeric_limits<mp_limb_t>::max() >= static_cast<unsigned long>(std::numeric_limits<long>::max()), "long won't fit in mp_limb_t");
    if (is_unsigned || x >= 0)
    {
        this->mont_repr.data[0] = (mp_limb_t)x;
    }
    else
    {
        const mp_limb_t borrow = mpn_sub_1(this->mont_repr.data, modulus.data, n, (mp_limb_t)-x);
#ifndef NDEBUG
            assert(borrow == 0);
#else
            UNUSED(borrow);
#endif
    }

    mul_reduce(Rsquared);
}

template<mp_size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::set_ulong(const unsigned long x)
{
    this->mont_repr.clear();
    this->mont_repr.data[0] = x;
    mul_reduce(Rsquared);
}

template<mp_size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::clear()
{
    this->mont_repr.clear();
}

template<mp_size_t n, const bigint<n>& modulus>
bigint<n> Fp_model<n,modulus>::as_bigint() const
{
    bigint<n> one;
    one.clear();
    one.data[0] = 1;

    Fp_model<n, modulus> res(*this);
    res.mul_reduce(one);

    return (res.mont_repr);
}

template<mp_size_t n, const bigint<n>& modulus>
unsigned long Fp_model<n,modulus>::as_ulong() const
{
    return this->as_bigint().as_ulong();
}

template<mp_size_t n, const bigint<n>& modulus>
bool Fp_model<n,modulus>::operator==(const Fp_model& other) const
{
    return (this->mont_repr == other.mont_repr);
}

template<mp_size_t n, const bigint<n>& modulus>
bool Fp_model<n,modulus>::operator!=(const Fp_model& other) const
{
    return (this->mont_repr != other.mont_repr);
}

template<mp_size_t n, const bigint<n>& modulus>
bool Fp_model<n,modulus>::operator>=(const Fp_model& other) const
{
    if(this->is_positive() && other.is_positive())
    {
        return this->as_bigint() >= other.as_bigint();
    }
    else if((!this->is_positive()) && (!other.is_positive()))
    {
        return this->as_bigint() <= other.as_bigint();
    }

    return this->is_positive();
}

template<mp_size_t n, const bigint<n>& modulus>
bool Fp_model<n,modulus>::operator>(const Fp_model& other) const
{
    if(this->is_positive() && other.is_positive())
    {
        return this->as_bigint() > other.as_bigint();
    }
    else if((!this->is_positive()) && (!other.is_positive()))
    {
        return this->as_bigint() < other.as_bigint();
    }

    return this->is_positive();
}

template<mp_size_t n, const bigint<n>& modulus>
bool Fp_model<n,modulus>::operator<=(const Fp_model& other) const
{
    if(this->is_positive() && other.is_positive())
    {
        return this->as_bigint() <= other.as_bigint();
    }
    else if((!this->is_positive()) && (!other.is_positive()))
    {
        return this->as_bigint() >= other.as_bigint();
    }

    return other.is_positive();
}

template<mp_size_t n, const bigint<n>& modulus>
bool Fp_model<n,modulus>::operator<(const Fp_model& other) const
{
    if(this->is_positive() && other.is_positive())
    {
        return this->as_bigint() < other.as_bigint();
    }
    else if((!this->is_positive()) && (!other.is_positive()))
    {
        return this->as_bigint() > other.as_bigint();
    }

    return other.is_positive();
}

template<mp_size_t n, const bigint<n>& modulus>
bool Fp_model<n,modulus>::is_zero() const
{
    return (this->mont_repr.is_zero()); // zero maps to zero
}

template<mp_size_t n, const bigint<n>& modulus>
bool Fp_model<n,modulus>::is_prime(size_t reps) const
{
    mpz_t zx;
    mpz_init(zx);
    this->as_bigint().to_mpz(zx);
    int ret = mpz_probab_prime_p(zx, reps);
    mpz_clear(zx);
    return ret >= 1;
}


template<mp_size_t n, const bigint<n>& modulus>
bool Fp_model<n,modulus>::is_positive() const
{
    return this->as_bigint() < euler && !this->is_zero();
}

template<mp_size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::print() const
{
    Fp_model<n,modulus> tmp;
    tmp.mont_repr.data[0] = 1;
    tmp.mul_reduce(this->mont_repr);

    tmp.mont_repr.print();
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::zero()
{
    Fp_model<n,modulus> res;
    res.mont_repr.clear();
    return res;
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::one()
{
    Fp_model<n,modulus> res;
    res.mont_repr.data[0] = 1;
    res.mul_reduce(Rsquared);
    return res;
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::geometric_generator()
{
    Fp_model<n,modulus> res;
    res.mont_repr.data[0] = 2;
    res.mul_reduce(Rsquared);
    return res;
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::arithmetic_generator()
{
    Fp_model<n,modulus> res;
    res.mont_repr.data[0] = 1;
    res.mul_reduce(Rsquared);
    return res;
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>& Fp_model<n,modulus>::operator+=(const Fp_model<n,modulus>& other)
{
#ifdef PROFILE_OP_COUNTS
    this->add_cnt++;
#endif
// #if defined(__x86_64__) && defined(USE_ASM)
    if (n == 2)
    {
        __asm__
            ("/* perform bignum addition */   \n\t"
             ADD_FIRSTADD
             ADD_NEXTADD(8)
             "/* if overflow: subtract     */ \n\t"
             "/* (tricky point: if A and B are in the range we do not need to do anything special for the possible carry flag) */ \n\t"
             "jc      subtract%=              \n\t"

             "/* check for overflow */        \n\t"
             ADD_CMP(8)
             ADD_CMP(0)

             "/* subtract mod if overflow */  \n\t"
             "subtract%=:                     \n\t"
             ADD_FIRSTSUB
             ADD_NEXTSUB(8)
             "done%=:                         \n\t"
             :
             : [A] "r" (this->mont_repr.data), [B] "r" (other.mont_repr.data), [mod] "r" (modulus.data)
             : "cc", "memory", "%rax");
    }
    else if (n == 3)
    {
        __asm__
            ("/* perform bignum addition */   \n\t"
             ADD_FIRSTADD
             ADD_NEXTADD(8)
             ADD_NEXTADD(16)
             "/* if overflow: subtract     */ \n\t"
             "/* (tricky point: if A and B are in the range we do not need to do anything special for the possible carry flag) */ \n\t"
             "jc      subtract%=              \n\t"

             "/* check for overflow */        \n\t"
             ADD_CMP(16)
             ADD_CMP(8)
             ADD_CMP(0)

             "/* subtract mod if overflow */  \n\t"
             "subtract%=:                     \n\t"
             ADD_FIRSTSUB
             ADD_NEXTSUB(8)
             ADD_NEXTSUB(16)
             "done%=:                         \n\t"
             :
             : [A] "r" (this->mont_repr.data), [B] "r" (other.mont_repr.data), [mod] "r" (modulus.data)
             : "cc", "memory", "%rax");
    }
    else if (n == 4)
    {
        __asm__
            ("/* perform bignum addition */   \n\t"
             ADD_FIRSTADD
             ADD_NEXTADD(8)
             ADD_NEXTADD(16)
             ADD_NEXTADD(24)
             "/* if overflow: subtract     */ \n\t"
             "/* (tricky point: if A and B are in the range we do not need to do anything special for the possible carry flag) */ \n\t"
             "jc      subtract%=              \n\t"

             "/* check for overflow */        \n\t"
             ADD_CMP(24)
             ADD_CMP(16)
             ADD_CMP(8)
             ADD_CMP(0)

             "/* subtract mod if overflow */  \n\t"
             "subtract%=:                     \n\t"
             ADD_FIRSTSUB
             ADD_NEXTSUB(8)
             ADD_NEXTSUB(16)
             ADD_NEXTSUB(24)
             "done%=:                         \n\t"
             :
             : [A] "r" (this->mont_repr.data), [B] "r" (other.mont_repr.data), [mod] "r" (modulus.data)
             : "cc", "memory", "%rax");
    }
    else if (n == 5)
    {
        __asm__
            ("/* perform bignum addition */   \n\t"
             ADD_FIRSTADD
             ADD_NEXTADD(8)
             ADD_NEXTADD(16)
             ADD_NEXTADD(24)
             ADD_NEXTADD(32)
             "/* if overflow: subtract     */ \n\t"
             "/* (tricky point: if A and B are in the range we do not need to do anything special for the possible carry flag) */ \n\t"
             "jc      subtract%=              \n\t"

             "/* check for overflow */        \n\t"
             ADD_CMP(32)
             ADD_CMP(24)
             ADD_CMP(16)
             ADD_CMP(8)
             ADD_CMP(0)

             "/* subtract mod if overflow */  \n\t"
             "subtract%=:                     \n\t"
             ADD_FIRSTSUB
             ADD_NEXTSUB(8)
             ADD_NEXTSUB(16)
             ADD_NEXTSUB(24)
             ADD_NEXTSUB(32)
             "done%=:                         \n\t"
             :
             : [A] "r" (this->mont_repr.data), [B] "r" (other.mont_repr.data), [mod] "r" (modulus.data)
             : "cc", "memory", "%rax");
    }
    else
// #endif
    {
        mp_limb_t scratch[n+1];
        const mp_limb_t carry = mpn_add_n(scratch, this->mont_repr.data, other.mont_repr.data, n);
        scratch[n] = carry;

        if (carry || mpn_cmp(scratch, modulus.data, n) >= 0)
        {
            const mp_limb_t borrow = mpn_sub(scratch, scratch, n+1, modulus.data, n);
#ifndef NDEBUG
            assert(borrow == 0);
#else
            UNUSED(borrow);
#endif
        }

        mpn_copyi(this->mont_repr.data, scratch, n);
    }

    return *this;
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>& Fp_model<n,modulus>::operator-=(const Fp_model<n,modulus>& other)
{
#ifdef PROFILE_OP_COUNTS
    this->sub_cnt++;
#endif
// #if defined(__x86_64__) && defined(USE_ASM)
    if (n == 2)
    {
        __asm__
            (SUB_FIRSTSUB
             SUB_NEXTSUB(8)

             "jnc     done%=\n\t"

             SUB_FIRSTADD
             SUB_NEXTADD(8)

             "done%=:\n\t"
             :
             : [A] "r" (this->mont_repr.data), [B] "r" (other.mont_repr.data), [mod] "r" (modulus.data)
             : "cc", "memory", "%rax");
    }
    else if (n == 3)
    {
        __asm__
            (SUB_FIRSTSUB
             SUB_NEXTSUB(8)
             SUB_NEXTSUB(16)

             "jnc     done%=\n\t"

             SUB_FIRSTADD
             SUB_NEXTADD(8)
             SUB_NEXTADD(16)

             "done%=:\n\t"
             :
             : [A] "r" (this->mont_repr.data), [B] "r" (other.mont_repr.data), [mod] "r" (modulus.data)
             : "cc", "memory", "%rax");
    }
    else if (n == 4)
    {
        __asm__
            (SUB_FIRSTSUB
             SUB_NEXTSUB(8)
             SUB_NEXTSUB(16)
             SUB_NEXTSUB(24)

             "jnc     done%=\n\t"

             SUB_FIRSTADD
             SUB_NEXTADD(8)
             SUB_NEXTADD(16)
             SUB_NEXTADD(24)

             "done%=:\n\t"
             :
             : [A] "r" (this->mont_repr.data), [B] "r" (other.mont_repr.data), [mod] "r" (modulus.data)
             : "cc", "memory", "%rax");
    }
    else if (n == 5)
    {
        __asm__
            (SUB_FIRSTSUB
             SUB_NEXTSUB(8)
             SUB_NEXTSUB(16)
             SUB_NEXTSUB(24)
             SUB_NEXTSUB(32)

             "jnc     done%=\n\t"

             SUB_FIRSTADD
             SUB_NEXTADD(8)
             SUB_NEXTADD(16)
             SUB_NEXTADD(24)
             SUB_NEXTADD(32)

             "done%=:\n\t"
             :
             : [A] "r" (this->mont_repr.data), [B] "r" (other.mont_repr.data), [mod] "r" (modulus.data)
             : "cc", "memory", "%rax");
    }
    else
// #endif
    {
        mp_limb_t scratch[n+1];
        if (mpn_cmp(this->mont_repr.data, other.mont_repr.data, n) < 0)
        {
            const mp_limb_t carry = mpn_add_n(scratch, this->mont_repr.data, modulus.data, n);
            scratch[n] = carry;
        }
        else
        {
            mpn_copyi(scratch, this->mont_repr.data, n);
            scratch[n] = 0;
        }

        const mp_limb_t borrow = mpn_sub(scratch, scratch, n+1, other.mont_repr.data, n);
#ifndef NDEBUG
        assert(borrow == 0);
#else
        UNUSED(borrow);
#endif

        mpn_copyi(this->mont_repr.data, scratch, n);
    }
    return *this;
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>& Fp_model<n,modulus>::operator*=(const Fp_model<n,modulus>& other)
{
#ifdef PROFILE_OP_COUNTS
    this->mul_cnt++;
#endif

    mul_reduce(other.mont_repr);
    return *this;
}


template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>& Fp_model<n,modulus>::operator/=(const Fp_model<n,modulus>& other)
{
    if(this->is_positive() || this->is_zero())
    {
        mpz_t zr;
        mpz_t zx;
        mpz_t zn;
        mpz_init(zx);
        mpz_init(zr);
        mpz_init(zn);
        this->as_bigint().to_mpz(zx);
        other.as_bigint().to_mpz(zn);
        mpz_fdiv_q(zr, zx, zn);
        *this = Fp_model(zr);
        mpz_clear(zx);
        mpz_clear(zr);
        mpz_clear(zn);
        return *this;
    }
    else
    {
        mpz_t zr;
        mpz_t zx;
        mpz_t zn;
        mpz_t zmod;
        mpz_init(zx);
        mpz_init(zr);
        mpz_init(zn);
        mpz_init(zmod);
        this->as_bigint().to_mpz(zx);
        other.as_bigint().to_mpz(zn);
        modulus.to_mpz(zmod);
        mpz_sub(zr, zx, zmod);
        mpz_fdiv_q(zr, zr, zn);
        *this = Fp_model(zr);
        mpz_clear(zx);
        mpz_clear(zr);
        mpz_clear(zn);
        mpz_clear(zmod);
        return *this;
    }
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>& Fp_model<n,modulus>::operator%=(const Fp_model<n,modulus>& other)
{
    if(this->is_positive() || this->is_zero())
    {
        mpz_t zr;
        mpz_t zx;
        mpz_t zn;
        mpz_init(zx);
        mpz_init(zr);
        mpz_init(zn);
        this->as_bigint().to_mpz(zx);
        other.as_bigint().to_mpz(zn);
        mpz_mod(zr, zx, zn);
        *this = Fp_model(zr);
        mpz_clear(zx);
        mpz_clear(zr);
        mpz_clear(zn);
        return *this;
    }
    else
    {
        mpz_t zr;
        mpz_t zx;
        mpz_t zn;
        mpz_t zmod;
        mpz_init(zx);
        mpz_init(zr);
        mpz_init(zn);
        mpz_init(zmod);
        this->as_bigint().to_mpz(zx);
        other.as_bigint().to_mpz(zn);
        modulus.to_mpz(zmod);
        mpz_sub(zr, zx, zmod);
        mpz_mod(zr, zr, zn);
        *this = Fp_model(zr);
        mpz_clear(zx);
        mpz_clear(zr);
        mpz_clear(zn);
        mpz_clear(zmod);
        return *this;
    }
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>& Fp_model<n,modulus>::operator^=(const unsigned long pow)
{
    (*this) = power<Fp_model<n, modulus> >(*this, pow);
    return (*this);
}

template<mp_size_t n, const bigint<n>& modulus>
template<mp_size_t m>
Fp_model<n,modulus>& Fp_model<n,modulus>::operator^=(const bigint<m> &pow)
{
    (*this) = power<Fp_model<n, modulus>, m>(*this, pow);
    return (*this);
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::operator+(const Fp_model<n,modulus>& other) const
{
    Fp_model<n, modulus> r(*this);
    return (r += other);
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::operator-(const Fp_model<n,modulus>& other) const
{
    Fp_model<n, modulus> r(*this);
    return (r -= other);
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::operator*(const Fp_model<n,modulus>& other) const
{
    Fp_model<n, modulus> r(*this);
    return (r *= other);
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::operator/(const Fp_model<n,modulus>& other) const
{
    Fp_model<n, modulus> r(*this);
    return (r /= other);
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::operator%(const Fp_model<n,modulus>& other) const
{
    Fp_model<n, modulus> r(*this);
    return (r %= other);
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::operator^(const unsigned long pow) const
{
    Fp_model<n, modulus> r(*this);
    return (r ^= pow);
}

template<mp_size_t n, const bigint<n>& modulus>
template<mp_size_t m>
Fp_model<n,modulus> Fp_model<n,modulus>::operator^(const bigint<m> &pow) const
{
    Fp_model<n, modulus> r(*this);
    return (r ^= pow);
}



template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::pow_mod(const Fp_model<n,modulus>& a, const Fp_model<n,modulus>& e, const Fp_model<n,modulus>& mod)
{

    mpz_t zr;
    mpz_t za;
    mpz_t ze;
    mpz_t zmod;
    mpz_init(zr);
    mpz_init(za);
    mpz_init(ze);
    mpz_init(zmod);
    a.as_bigint().to_mpz(za);
    e.as_bigint().to_mpz(ze);
    mod.as_bigint().to_mpz(zmod);
    mpz_powm(zr, za, ze, zmod);
    Fp_model<n,modulus> r(zr);
    mpz_clear(zr);
    mpz_clear(za);
    mpz_clear(ze);
    mpz_clear(zmod);
    return r;
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::sqrt_int(const Fp_model<n,modulus>& a) {
    mpz_t zr;
    mpz_t za;
    mpz_init(zr);
    mpz_init(za);
    a.as_bigint().to_mpz(za);
    mpz_sqrt(zr, za);
    Fp_model<n,modulus> r(zr);
    mpz_clear(zr);
    mpz_clear(za);
    return r;
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::divexact(const Fp_model<n,modulus>& a, const Fp_model<n,modulus>& b) {
    if(a.is_positive() || a.is_zero())
    {
        mpz_t zr;
        mpz_t zb;
        mpz_t za;
        mpz_init(zr);
        mpz_init(zb);
        mpz_init(za);
        a.as_bigint().to_mpz(za);
        b.as_bigint().to_mpz(zb);
        mpz_divexact(zr, za, zb);
        Fp_model<n,modulus> r(zr);
        mpz_clear(zr);
        mpz_clear(zb);
        mpz_clear(za);
        return r;
    }
    else
    {
        mpz_t zr;
        mpz_t za;
        mpz_t zb;
        mpz_t zmod;
        mpz_init(zr);
        mpz_init(za);
        mpz_init(zb);
        mpz_init(zmod);
        a.as_bigint().to_mpz(za);
        b.as_bigint().to_mpz(zb);
        modulus.to_mpz(zmod);
        mpz_sub(za, za, zmod);
        mpz_divexact(zr, za, zb);
        Fp_model<n,modulus> r(zr);
        mpz_clear(zr);
        mpz_clear(za);
        mpz_clear(zb);
        mpz_clear(zmod);
        return r;
    }
}

template<mp_size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::mods(mpz_t a, mpz_t b, mpz_t out) {
    mpz_t t;
    mpz_init(t);

    mpz_mod(out, a, b);
    mpz_mul_ui(t, out, 2);
    if (mpz_cmp(t, b) > 0) mpz_sub(out, out, b);

    mpz_clear(t);
}

template<mp_size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::quos(mpz_t a, mpz_t b, mpz_t out) {
    mpz_t t;
    mpz_init(t);

    // t = a - mods(a, b)
    mods(a, b, t);
    mpz_sub(t, a, t);

    mpz_divexact(out, t, b);

    mpz_clear(t);
}

template<mp_size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::grem(mpz_t r0, mpz_t r1, mpz_t w0, mpz_t w1, mpz_t z0, mpz_t z1) {
    mpz_t t0, t1, u0, u1, n0;
    mpz_init(t0);
    mpz_init(t1);
    mpz_init(u0);
    mpz_init(u1);
    mpz_init(n0);
    
	// FieldT n = z0 * z0 + z1 * z1;
    mpz_mul(t0, z0, z0);
    mpz_mul(t1, z1, z1);
    mpz_add(n0, t0, t1);
    

	// FieldT u0 = quos(w0 * z0 + w1 * z1, n);
    mpz_mul(t0, w0, z0);
    mpz_mul(t1, w1, z1);
    mpz_add(t0, t0, t1);
    quos(t0, n0, u0);

	// FieldT u1 = quos(w1 * z0 - w0 * z1, n);
    mpz_mul(t0, w1, z0);
    mpz_mul(t1, w0, z1);
    mpz_sub(t0, t0, t1);
    quos(t0, n0, u1);

	// r0 = w0 - z0 * u0 + z1 * u1;
    mpz_mul(t0, z0, u0);
    mpz_mul(t1, z1, u1);
    mpz_sub(r0, w0, t0);
    mpz_add(r0, r0, t1);
    
	// r1 = w1 - z0 * u1 - z1 * u0;
    mpz_mul(t0, z0, u1);
    mpz_mul(t1, z1, u0);
    mpz_sub(r1, w1, t0);
    mpz_sub(r1, r1, t1);

    mpz_clear(t0);
    mpz_clear(t1);
    mpz_clear(u0);
    mpz_clear(u1);
    mpz_clear(n0);
}

template<mp_size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::ggcd(mpz_t r0, mpz_t r1, mpz_t w0, mpz_t w1, mpz_t z0, mpz_t z1) {
    while (mpz_cmp_ui(z0, 0) != 0 || mpz_cmp_ui(z1, 0) != 0) {
        mpz_set(r0, z0);
        mpz_set(r1, z1);
        grem(z0, z1, w0, w1, r0, r1);
        mpz_set(w0, r0);
        mpz_set(w1, r1);
    }
}

template<mp_size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::root4(mpz_t p, mpz_t out) {
    mpz_t k, j, a, b;
    mpz_init(k);
    mpz_init(j);
    mpz_init(a);
    mpz_init(b);
    mpz_div_ui(k, p, 4);
    mpz_set_ui(j, 2);
    
    while (true) {
        mpz_powm(a, j, k, p);
        mpz_mul(b, a, a);
        mods(b, p, b);
        if (mpz_cmp_si(b, -1) == 0) {
            mpz_set(out, a);
            break;
        }
        mpz_add_ui(j, j, 1);
    }

    mpz_clear(k);
    mpz_clear(j);
    mpz_clear(a);
    mpz_clear(b);
}

template<mp_size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::two_square_decomp(Fp_model<n, modulus>& y1, Fp_model<n, modulus>& y2, mpz_t x) {
    mpz_t a, zy1, zy2, w1, z1;
    mpz_init(a);
    mpz_init(zy1);
    mpz_init(zy2);
    mpz_init(w1);
    mpz_init(z1);

    root4(x, a);
    mpz_set_ui(w1, 0);
    mpz_set_ui(z1, 1);
    ggcd(zy1, zy2, x, w1, a, z1);
    
    if (mpz_cmp_si(zy1, 0) < 0) {
        mpz_neg(zy1, zy1);
    }
    if (mpz_cmp_si(zy2, 0) < 0) {
        mpz_neg(zy2, zy2);
    }
    y1 = Fp_model(zy1);
    y2 = Fp_model(zy2);

    mpz_clear(a);
    mpz_clear(zy1);
    mpz_clear(zy2);
    mpz_clear(w1);
    mpz_clear(z1);
}

template<mp_size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::three_square_decomp(Fp_model<n,modulus>& y1, Fp_model<n,modulus>& y2, Fp_model<n,modulus>& y3, const Fp_model<n,modulus>& in) {
    mpz_t x, r, p, t;
    mpz_init(x);
    mpz_init(r);
    mpz_init(p);
    mpz_init(t);

    in.as_bigint().to_mpz(x);

    mpz_sqrt(r, x);
    mpz_mul(t, r, r);
    mpz_sub(p, x, t);
    while (mpz_probab_prime_p(p, 15) < 1 && mpz_cmp_ui(p, 1) != 0 && mpz_cmp_ui(p, 0) != 0) {
        mpz_sub_ui(r, r, 1);
        mpz_mul(t, r, r);
        mpz_sub(p, x, t);
    }
    y1 = Fp_model(r);
    if (mpz_cmp_ui(p, 1) == 0) {
        y2 = one();
        y3 = zero();
        goto three_square_decomp_end;
    } else if (mpz_cmp_ui(p, 0) == 0) {
        y2 = zero();
        y2 = zero();
        goto three_square_decomp_end;
    }
    two_square_decomp(y2, y3, p);

three_square_decomp_end:
    mpz_clear(x);
    mpz_clear(r);
    mpz_clear(p);
    mpz_clear(t);
}

template<mp_size_t n, const bigint<n>& modulus>
std::vector<unsigned long> Fp_model<n, modulus>::linearSieve(unsigned long x) {
    is_prime_arr.resize(x + 1, true);
    std::vector<unsigned long> primes;

    for (int i = 2; i <= x; ++i) {
        if (is_prime_arr[i]) {
            primes.push_back(i);
        }
        for (int j = 0; j < primes.size() && i * primes[j] <= x; ++j) {
            is_prime_arr[i * primes[j]] = false;
            if (i % primes[j] == 0) {
                break;
            }
        }
    }
    return primes;
}

template<mp_size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::three_square_decomp_opti_init(unsigned long mx, char* filename, bool save) {
    double start = omp_get_wtime();
    table_mx = mx;
    if (filename && !save) {
      printf("load hashmap from %s\n", filename);
      FILE *fp = fopen(filename, "rb");

      unsigned long x, y1, y2;
      is_prime_arr.resize(mx + 1, false);
      while (fread(&x, 8, 1, fp)) {
        fread(&y1, 8, 1, fp);
        fread(&y2, 8, 1, fp);
        num_to_two_square.insert({x, {y1, y2}});
        is_prime_arr[x] = true;
      }
      fclose(fp);
    } else {
      auto primes = linearSieve(mx);

      num_to_two_square.insert({2, {1, 1}});
      mpz_t p;
      mpz_init(p);

      for (auto num: primes) {
          if (num % 4 != 1) continue;
          mpz_set_ui(p, num);
          Fp_model<n, modulus> y1, y2;
          two_square_decomp(y1, y2, p);
          // printf("%lu = %lu ^ 2 + %lu ^ 2\n", num, y1.as_ulong(), y2.as_ulong());
          num_to_two_square.insert({num, {y1.as_ulong(), y2.as_ulong()}});
      }
      mpz_clear(p);
    }
    double end = omp_get_wtime();
    printf("table size: %ld\n", num_to_two_square.size());
    printf("hash init time: %lf\n", end - start);

    if (filename && save) {
      FILE *fp = fopen(filename, "wb");
      for (auto &p: num_to_two_square) {
        fwrite(&p.first, 8, 1, fp);
        fwrite(&p.second.first, 8, 1, fp);
        fwrite(&p.second.second, 8, 1, fp);
      }
      printf("save hashmap to %s\n", filename);
    }
}

template<mp_size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::two_square_decomp_opti(Fp_model<n, modulus>& y1, Fp_model<n, modulus>& y2, mpz_t x) {
    unsigned long p = mpz_get_ui(x);
    if (num_to_two_square.find(p) != num_to_two_square.end()) {
        std::pair<unsigned long, unsigned long> pair = num_to_two_square[p];
        y1 = Fp_model(pair.first);
        y2 = Fp_model(pair.second);
    } else {
        // puts("miss");
        two_square_decomp(y1, y2, x);
    }
}

template<mp_size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::three_square_decomp_opti(Fp_model<n,modulus>& y1, Fp_model<n,modulus>& y2, Fp_model<n,modulus>& y3, const Fp_model<n,modulus>& in) {
    mpz_t x, r, p, t;
    mpz_init(x);
    mpz_init(r);
    mpz_init(p);
    mpz_init(t);

    in.as_bigint().to_mpz(x);

    mpz_sqrt(r, x);
    mpz_mul(t, r, r);
    mpz_sub(p, x, t);
    auto is_probab_prime = [&](mpz_t p) -> bool {
        unsigned long p_int = mpz_get_ui(p);
        if (p_int <= table_mx) {
            // puts("fast path");
            // return num_to_two_square.find(p_int) != num_to_two_square.end();
            return is_prime_arr[p_int];
        } else {
            return mpz_probab_prime_p(p, 15) >= 1;
        }
    };
    while (!is_probab_prime(p) && mpz_cmp_ui(p, 1) != 0 && mpz_cmp_ui(p, 0) != 0) {
        mpz_sub_ui(r, r, 1);
        mpz_mul(t, r, r);
        mpz_sub(p, x, t);
    }
    y1 = Fp_model(r);
    if (mpz_cmp_ui(p, 1) == 0) {
        y2 = one();
        y3 = zero();
    } else if (mpz_cmp_ui(p, 0) == 0) {
        y2 = zero();
        y3 = zero();
    } else {
        two_square_decomp_opti(y2, y3, p);
    }

    mpz_clear(x);
    mpz_clear(r);
    mpz_clear(p);
    mpz_clear(t);
}

template<mp_size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::three_square_decomp_long(Fp_model<n,modulus>& _y1, Fp_model<n,modulus>& _y2, Fp_model<n,modulus>& _y3, const Fp_model<n,modulus>& _in) {
    long long x = _in.as_ulong();
    long long r = (long long)std::sqrt((long double)x);
    long long t = r * r;
    long long p = x - t;
    mpz_t _p;
    mpz_init(_p);

    auto is_probab_prime = [&](long long p) -> bool {
        if (p <= table_mx) {
            return is_prime_arr[p];
        } else {
            mpz_set_ui(_p, p);
            return mpz_probab_prime_p(_p, 15) >= 1;
        }
    };
    while (!is_probab_prime(p) && p != 0 && p != 1) {
        t = t - 2 * r + 1;
        r = r - 1;
        p = x - t;
    }
    _y1 = Fp_model(r);
    if (p == 1) {
        _y2 = one();
        _y3 = zero();
    } else if (p == 0) {
        _y2 = zero();
        _y3 = zero();
    } else {
        mpz_set_ui(_p, p);
        two_square_decomp_opti(_y2, _y3, _p);
    }

    mpz_clear(_p);
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::operator-() const
{
#ifdef PROFILE_OP_COUNTS
    this->sub_cnt++;
#endif

    if (this->is_zero())
    {
        return (*this);
    }
    else
    {
        Fp_model<n, modulus> r;
        mpn_sub_n(r.mont_repr.data, modulus.data, this->mont_repr.data, n);
        return r;
    }
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::squared() const
{
#ifdef PROFILE_OP_COUNTS
    this->sqr_cnt++;
    this->mul_cnt--; // zero out the upcoming mul
#endif
    /* stupid pre-processor tricks; beware */
#if defined(__x86_64__) && defined(USE_ASM)
    if (n == 3)
    { // use asm-optimized Comba squaring
        mp_limb_t res[2*n];
        mp_limb_t c0, c1, c2;
        COMBA_3_BY_3_SQR(c0, c1, c2, res, this->mont_repr.data);

        mp_limb_t k;
        mp_limb_t tmp1, tmp2, tmp3;
        REDUCE_6_LIMB_PRODUCT(k, tmp1, tmp2, tmp3, inv, res, modulus.data);

        /* subtract t > mod */
        __asm__ volatile
            ("/* check for overflow */        \n\t"
             MONT_CMP(16)
             MONT_CMP(8)
             MONT_CMP(0)

             "/* subtract mod if overflow */  \n\t"
             "subtract%=:                     \n\t"
             MONT_FIRSTSUB
             MONT_NEXTSUB(8)
             MONT_NEXTSUB(16)
             "done%=:                         \n\t"
             :
             : [tmp] "r" (res+n), [M] "r" (modulus.data)
             : "cc", "memory", "%rax");

        Fp_model<n, modulus> r;
        mpn_copyi(r.mont_repr.data, res+n, n);
        return r;
    }
    else
#endif
    {
        Fp_model<n, modulus> r(*this);
        return (r *= r);
    }
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>& Fp_model<n,modulus>::invert()
{
#ifdef PROFILE_OP_COUNTS
    this->inv_cnt++;
#endif

    assert(!this->is_zero());

    bigint<n> g; /* gp should have room for vn = n limbs */

    mp_limb_t s[n+1]; /* sp should have room for vn+1 limbs */
    mp_size_t sn;

    bigint<n> v = modulus; // both source operands are destroyed by mpn_gcdext

    /* computes gcd(u, v) = g = u*s + v*t, so s*u will be 1 (mod v) */
    const mp_size_t gn = mpn_gcdext(g.data, s, &sn, this->mont_repr.data, n, v.data, n);
#ifndef NDEBUG
    assert(gn == 1 && g.data[0] == 1); /* inverse exists */
#else
    UNUSED(gn);
#endif

    mp_limb_t q; /* division result fits into q, as sn <= n+1 */
    /* sn < 0 indicates negative sn; will fix up later */

    if (std::abs(sn) >= n)
    {
        /* if sn could require modulus reduction, do it here */
        mpn_tdiv_qr(&q, this->mont_repr.data, 0, s, std::abs(sn), modulus.data, n);
    }
    else
    {
        /* otherwise just copy it over */
        mpn_zero(this->mont_repr.data, n);
        mpn_copyi(this->mont_repr.data, s, std::abs(sn));
    }

    /* fix up the negative sn */
    if (sn < 0)
    {
        const mp_limb_t borrow = mpn_sub_n(this->mont_repr.data, modulus.data, this->mont_repr.data, n);
#ifndef NDEBUG
        assert(borrow == 0);
#else
        UNUSED(borrow);
#endif
    }

    mul_reduce(Rcubed);
    return *this;
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::inverse() const
{
    Fp_model<n, modulus> r(*this);
    return (r.invert());
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n, modulus> Fp_model<n,modulus>::random_element() /// returns random element of Fp_model
{
//     /* note that as Montgomery representation is a bijection then
//        selecting a random element of {xR} is the same as selecting a
//        random element of {x} */
//     Fp_model<n, modulus> r;
//     do
//     {
//         r.mont_repr.randomize();

//         /* clear all bits higher than MSB of modulus */
//         size_t bitno = GMP_NUMB_BITS * n - 1;
//         while (modulus.test_bit(bitno) == false)
//         {
//             const std::size_t part = bitno/GMP_NUMB_BITS;
//             const std::size_t bit = bitno - (GMP_NUMB_BITS*part);

//             r.mont_repr.data[part] &= ~(1ul<<bit);

//             bitno--;
//         }
//     }
//    /* if r.data is still >= modulus -- repeat (rejection sampling) */
//     while (mpn_cmp(r.mont_repr.data, modulus.data, n) >= 0);
    Fp_model<n, modulus> r;
    std::random_device rd;
    r.set_ulong(rd());
    
    return r;
    // return nqr;
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n, modulus> Fp_model<n,modulus>::random_element(const Fp_model& other) /// returns random element of Fp_model
{
    Fp_model<n, modulus> r = random_element() % other;
    return r;
}

template<mp_size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::sqrt() const
{
    Fp_model<n,modulus> one = Fp_model<n,modulus>::one();

    size_t v = Fp_model<n,modulus>::s;
    Fp_model<n,modulus> z = Fp_model<n,modulus>::nqr_to_t;
    Fp_model<n,modulus> w = (*this)^Fp_model<n,modulus>::t_minus_1_over_2;
    Fp_model<n,modulus> x = (*this) * w;
    Fp_model<n,modulus> b = x * w; // b = (*this)^t

#if DEBUG
    // check if square with euler's criterion
    Fp_model<n,modulus> check = b;
    for (size_t i = 0; i < v-1; ++i)
    {
        check = check.squared();
    }
    if (check != one)
    {
        assert(0);
    }
#endif

    // compute square root with Tonelli--Shanks
    // (does not terminate if not a square!)

    while (b != one)
    {
        size_t m = 0;
        Fp_model<n,modulus> b2m = b;
        while (b2m != one)
        {
            /* invariant: b2m = b^(2^m) after entering this loop */
            b2m = b2m.squared();
            m += 1;
        }

        int j = v-m-1;
        w = z;
        while (j > 0)
        {
            w = w.squared();
            --j;
        } // w = z^2^(v-m-1)

        z = w.squared();
        b = b * z;
        x = x * w;
        v = m;
    }

    return x;
}

template<mp_size_t n, const bigint<n>& modulus>
std::ostream& operator<<(std::ostream &out, const Fp_model<n, modulus> &p)
{
#ifndef MONTGOMERY_OUTPUT
    Fp_model<n,modulus> tmp;
    tmp.mont_repr.data[0] = 1;
    tmp.mul_reduce(p.mont_repr);
    out << tmp.mont_repr;
#else
    out << p.mont_repr;
#endif
    return out;
}

template<mp_size_t n, const bigint<n>& modulus>
std::istream& operator>>(std::istream &in, Fp_model<n, modulus> &p)
{
#ifndef MONTGOMERY_OUTPUT
    in >> p.mont_repr;
    p.mul_reduce(Fp_model<n, modulus>::Rsquared);
#else
    in >> p.mont_repr;
#endif
    return in;
}

} // libff
#endif // FP_TCC_
