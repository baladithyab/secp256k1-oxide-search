// secp256k1_field.cuh — 256-bit field arithmetic mod p for secp256k1.
//
// Limb layout: 8 x uint32, little-endian limb order (limb[0] = least-significant 32 bits).
// Field prime p = 2^256 - 2^32 - 977.
// Reduction: schoolbook 256x256 -> 512-bit product, then fold the high 256 bits using
//   2^256 ≡ 2^32 + 977 (mod p). Two fold passes + a final conditional subtract suffice.
// All arithmetic is integer-only (NO float anywhere). 32x32->64 multiplies with explicit carry.
//
// This is the BIT-EXACT correctness reference for Wave 1.
#ifndef SECP256K1_FIELD_CUH
#define SECP256K1_FIELD_CUH

#include <stdint.h>

typedef struct {
    uint32_t v[8];  // little-endian limbs: v[0] = bits 0..31
} fe;               // field element in [0, p)

// p = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
__device__ __constant__ uint32_t P_LIMBS[8] = {
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// ---- basic helpers -------------------------------------------------------

__device__ __forceinline__ void fe_set_zero(fe *r) {
#pragma unroll
    for (int i = 0; i < 8; i++) r->v[i] = 0u;
}

__device__ __forceinline__ void fe_copy(fe *r, const fe *a) {
#pragma unroll
    for (int i = 0; i < 8; i++) r->v[i] = a->v[i];
}

__device__ __forceinline__ int fe_is_zero(const fe *a) {
    uint32_t x = 0u;
#pragma unroll
    for (int i = 0; i < 8; i++) x |= a->v[i];
    return x == 0u;
}

__device__ __forceinline__ int fe_is_odd(const fe *a) {
    return (int)(a->v[0] & 1u);
}

__device__ __forceinline__ int fe_equal(const fe *a, const fe *b) {
    uint32_t d = 0u;
#pragma unroll
    for (int i = 0; i < 8; i++) d |= (a->v[i] ^ b->v[i]);
    return d == 0u;
}

// Compare a vs p. Returns 1 if a >= p, else 0.
__device__ __forceinline__ int fe_ge_p(const uint32_t a[8]) {
#pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (a[i] > P_LIMBS[i]) return 1;
        if (a[i] < P_LIMBS[i]) return 0;
    }
    return 1;  // equal => a >= p
}

// r = a - p (assumes a >= p). Returns the borrow (should be 0).
__device__ __forceinline__ void fe_sub_p(uint32_t r[8], const uint32_t a[8]) {
    uint64_t borrow = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t t = (uint64_t)a[i] - (uint64_t)P_LIMBS[i] - borrow;
        r[i] = (uint32_t)t;
        borrow = (t >> 63) & 1ull;  // borrow if subtraction underflowed
    }
}

// conditional final reduce: if a >= p, a -= p (single subtract enough after fold).
__device__ __forceinline__ void fe_reduce_once(uint32_t a[8]) {
    if (fe_ge_p(a)) {
        uint32_t t[8];
        fe_sub_p(t, a);
#pragma unroll
        for (int i = 0; i < 8; i++) a[i] = t[i];
    }
}

// ---- add / sub mod p -----------------------------------------------------

__device__ __forceinline__ void fe_add(fe *r, const fe *a, const fe *b) {
    uint32_t t[8];
    uint64_t carry = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t s = (uint64_t)a->v[i] + (uint64_t)b->v[i] + carry;
        t[i] = (uint32_t)s;
        carry = s >> 32;
    }
    // sum may be up to 2p-2 (< 2^257). If there's a top carry, fold 2^256 = 2^32+977.
    if (carry) {
        // add (2^32 + 977) to t
        uint64_t c = (uint64_t)t[0] + 977ull;
        t[0] = (uint32_t)c; c >>= 32;
        c += (uint64_t)t[1] + 1ull;
        t[1] = (uint32_t)c; c >>= 32;
#pragma unroll
        for (int i = 2; i < 8; i++) {
            c += (uint64_t)t[i];
            t[i] = (uint32_t)c; c >>= 32;
        }
        // c could still produce another tiny carry; handle by another fold of 2^256.
        if (c) {
            uint64_t c2 = (uint64_t)t[0] + 977ull;
            t[0] = (uint32_t)c2; c2 >>= 32;
            c2 += (uint64_t)t[1] + 1ull;
            t[1] = (uint32_t)c2; c2 >>= 32;
#pragma unroll
            for (int i = 2; i < 8; i++) {
                c2 += (uint64_t)t[i];
                t[i] = (uint32_t)c2; c2 >>= 32;
            }
        }
    }
    fe_reduce_once(t);
#pragma unroll
    for (int i = 0; i < 8; i++) r->v[i] = t[i];
}

// r = a - b mod p
__device__ __forceinline__ void fe_sub(fe *r, const fe *a, const fe *b) {
    uint32_t t[8];
    int64_t borrow = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int64_t s = (int64_t)a->v[i] - (int64_t)b->v[i] - borrow;
        t[i] = (uint32_t)s;
        borrow = (s < 0) ? 1 : 0;
    }
    if (borrow) {
        // result negative: add p (mod 2^256), i.e. add p back.
        uint64_t carry = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            uint64_t s = (uint64_t)t[i] + (uint64_t)P_LIMBS[i] + carry;
            t[i] = (uint32_t)s;
            carry = s >> 32;
        }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) r->v[i] = t[i];
}

// ---- multiply mod p ------------------------------------------------------
//
// Schoolbook 256x256 -> 512, then fold high half via 2^256 = 2^32 + 977.

// Fold a 512-bit value (16 x uint32, little-endian) into 256-bit result mod p.
// hi = words[8..15], lo = words[0..7].  result = lo + hi*(2^32 + 977) repeatedly.
__device__ __forceinline__ void fe_fold512(uint32_t out[8], const uint32_t w[16]) {
    // First fold: acc(256) = lo + hi*977 + hi*2^32
    // We accumulate into a 9-limb buffer to capture overflow, then fold once more.
    uint64_t acc[9];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = (uint64_t)w[i];
    acc[8] = 0;

    // add hi*977 (hi limbs are w[8..15])
    uint64_t carry = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t prod = (uint64_t)w[8 + i] * 977ull + acc[i] + carry;
        acc[i] = (uint32_t)prod;
        carry = prod >> 32;
    }
    acc[8] += carry;

    // add hi*2^32  (hi shifted up by one limb): add w[8+i] into acc[i+1].
    // For i = 0..6 this writes 32-bit limbs acc[1..7]. For i = 7 the destination is
    // acc[8], the OVERFLOW word: it must keep full precision (the low 32 bits are at
    // weight 2^256 and the carry-out is at weight 2^288). Do NOT truncate acc[8] to
    // 32 bits — that was the bit-exact bug (it collapsed the 2^288 carry onto 2^256).
    carry = 0;
#pragma unroll
    for (int i = 0; i < 7; i++) {
        uint64_t s = acc[i + 1] + (uint64_t)w[8 + i] + carry;
        acc[i + 1] = (uint32_t)s;
        carry = s >> 32;
    }
    // i = 7: full-precision overflow word. acc[8] (<= ~977 from the 977 pass) + w[15]
    // (< 2^32) + carry (<= 1) fits comfortably in 64 bits and is the true value of all
    // bits >= 256. This whole quantity is the new "hi" to fold again.
    uint64_t extra = acc[8] + (uint64_t)w[15] + carry;

    // Now acc[0..7] is a 256-bit value, with `extra` representing bits >= 256.
    // Fold `extra` again: value += extra*(2^32 + 977).
    uint32_t t[8];
#pragma unroll
    for (int i = 0; i < 8; i++) t[i] = (uint32_t)acc[i];

    // extra is small: bounded by acc[8] (<= ~977) + w[15] (< 2^32) + carry (<= 1),
    // i.e. e < 2^32 + 1024 (~33 bits). Then e*977 < 2^43, well within 64 bits.
    uint64_t e = extra;
    if (e) {
        // add e*977
        uint64_t c = (uint64_t)t[0] + e * 977ull;
        t[0] = (uint32_t)c; c >>= 32;
#pragma unroll
        for (int i = 1; i < 8; i++) {
            c += (uint64_t)t[i];
            t[i] = (uint32_t)c; c >>= 32;
        }
        // add e*2^32 (i.e. add e into limb 1)
        uint64_t c2 = (uint64_t)t[1] + e + c;  // also absorb leftover c
        t[1] = (uint32_t)c2; c2 >>= 32;
#pragma unroll
        for (int i = 2; i < 8; i++) {
            c2 += (uint64_t)t[i];
            t[i] = (uint32_t)c2; c2 >>= 32;
        }
        // any remaining c2 carry: fold once more (tiny)
        if (c2) {
            uint64_t c3 = (uint64_t)t[0] + c2 * 977ull;
            t[0] = (uint32_t)c3; c3 >>= 32;
            c3 += (uint64_t)t[1] + c2;  // +c2*2^32
            t[1] = (uint32_t)c3; c3 >>= 32;
#pragma unroll
            for (int i = 2; i < 8; i++) {
                c3 += (uint64_t)t[i];
                t[i] = (uint32_t)c3; c3 >>= 32;
            }
        }
    }
    // After all folds, t < 2*p comfortably, but loop the conditional subtract to be
    // bit-exact-safe for the near-p operand class (where extra is largest). Each pass
    // removes one p; in practice at most one or two iterations are ever taken.
#pragma unroll 1
    for (int rep = 0; rep < 3; rep++) {
        if (!fe_ge_p(t)) break;
        fe_reduce_once(t);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) out[i] = t[i];
}

__device__ __forceinline__ void fe_mul(fe *r, const fe *a, const fe *b) {
    uint32_t prod[16];
#pragma unroll
    for (int i = 0; i < 16; i++) prod[i] = 0u;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            uint64_t t = (uint64_t)a->v[i] * (uint64_t)b->v[j]
                       + (uint64_t)prod[i + j] + carry;
            prod[i + j] = (uint32_t)t;
            carry = t >> 32;
        }
        prod[i + 8] += (uint32_t)carry;  // no further carry possible: prod[i+8] was 0
    }
    fe_fold512(r->v, prod);
}

__device__ __forceinline__ void fe_sqr(fe *r, const fe *a) {
    fe_mul(r, a, a);
}

// ---- modular inverse via Fermat: a^(p-2) mod p ---------------------------
// p-2 = 0xFFFFFFFF...FFFFFFFE FFFFFC2D
// We use a fixed addition chain expressed as square-and-multiply over the bits.
__device__ void fe_inv(fe *r, const fe *a) {
    // p-2 limbs (little-endian): same as p but limb0 = 0xFFFFFC2D
    const uint32_t e[8] = {
        0xFFFFFC2Du, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };
    fe result;
    // result = 1
    fe_set_zero(&result);
    result.v[0] = 1u;
    fe base;
    fe_copy(&base, a);
    // square-and-multiply, LSB to MSB
#pragma unroll
    for (int limb = 0; limb < 8; limb++) {
        uint32_t bits = e[limb];
        for (int b = 0; b < 32; b++) {
            if (bits & 1u) {
                fe tmp; fe_mul(&tmp, &result, &base); fe_copy(&result, &tmp);
            }
            fe sq; fe_sqr(&sq, &base); fe_copy(&base, &sq);
            bits >>= 1;
        }
    }
    fe_copy(r, &result);
}

#endif // SECP256K1_FIELD_CUH
