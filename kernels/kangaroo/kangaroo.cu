// kangaroo.cu — GPU Pollard's-kangaroo ECDLP solver for secp256k1 (Track D, ADR-0003 Regime 2).
//
// Bounded-interval discrete log: given the EXPOSED compressed pubkey Q = d*G with d in [lo, hi],
// recover d. Research artifact (no broadcast/spend); KATs are self-generated toy intervals.
//
// REUSES the bit-exact verified field/EC/affine primitives unchanged (porting bugs are the #1 risk):
//   secp256k1_field.cuh   fe add/sub/mul/sqr/inv mod p
//   secp256k1_ec.cuh      jpoint add/double, scalar_mul_G, jpoint_to_affine, compressed_pubkey
//   secp256k1_affine.cuh  affpt, aff_add_with_inv (precomputed-inverse affine add), aff_add_fallback
// The ONLY new field routine is a modular square root (a^((p+1)/4), valid since p = 3 mod 4) used to
// decompress Q. Everything else is the kangaroo bookkeeping on top of the trusted group ops.
//
// Method:
//   Interval-shift to Q' = Q - lo*G = (d-lo)*G, so the unknown d' = d-lo lies in [0, W=hi-lo] and
//   every tracked scalar/distance stays << 2^64 << n. The tame/wild collision identity
//        d' = T_tame - D_wild     (T = tame absolute shifted-scalar, D = wild distance incl. start offset)
//   then holds as an EXACT integer equation — no mod-n arithmetic needed (the field code only does
//   mod p). A handful-of-bits distinguished-point (DP) hash table on the host detects collisions.
//
// Build: /usr/local/cuda/bin/nvcc -O3 -arch=sm_120 -I../cuda-ref -o kangaroo kangaroo.cu
// Run:   ./kangaroo <lo_hex> <hi_hex> <Q_compressed_hex>      (prints "d=<hex>" on success)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unordered_map>
#include <string>
#include <vector>

#include "secp256k1_field.cuh"
#include "secp256k1_ec.cuh"
#include "secp256k1_affine.cuh"

#define K_JUMPS 32                 // number of jump scalars (power of 2 so index = x & (K-1))
#ifndef KANG_CAP
#define KANG_CAP (1u << 21)        // DP report buffer capacity per round
#endif

// (p+1)/4, little-endian u32 limbs — exponent for the modular square root (p = 3 mod 4).
// Verified against G: sqrt(Gx^3+7) == Gy. (see kangaroo notes / oracle cross-check)
__device__ __constant__ uint32_t SQRT_EXP[8] = {
    0xBFFFFF0Cu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x3FFFFFFFu
};

// ----- device globals (set once at setup) ---------------------------------
__device__ affpt   d_jump[K_JUMPS];     // jump points J[i] = s[i]*G (affine)
__device__ uint64_t d_jumpscalar[K_JUMPS]; // s[i] (kept on device so the walk accumulates distance)
__device__ affpt   d_Qp;                // shifted target Q' = Q - lo*G (affine)
__device__ uint32_t d_dpcount;          // atomic DP write index for the current round
__device__ unsigned long long d_escapes; // negmap diagnostic: count of fruitless-cycle escapes fired

// ----- field helper: negate mod p (y -> p - y), used by the negation map ----
__device__ __forceinline__ void fe_negate(fe *r, const fe *a) {
    fe feP;
#pragma unroll
    for (int i = 0; i < 8; i++) feP.v[i] = P_LIMBS[i];
    fe_sub(r, &feP, a);   // p - a (a in [1,p) => result in (0,p]; a=0 maps to 0 but never occurs for y)
}

// ----- new field helper: modular square root ------------------------------
__device__ void fe_sqrt(fe *r, const fe *a) {        // r = a^((p+1)/4) mod p
    fe result; fe_set_zero(&result); result.v[0] = 1u;
    fe base; fe_copy(&base, a);
#pragma unroll
    for (int limb = 0; limb < 8; limb++) {
        uint32_t bits = SQRT_EXP[limb];
        for (int b = 0; b < 32; b++) {
            if (bits & 1u) { fe t; fe_mul(&t, &result, &base); fe_copy(&result, &t); }
            fe sq; fe_sqr(&sq, &base); fe_copy(&base, &sq);
            bits >>= 1;
        }
    }
    fe_copy(r, &result);
}

// Decompress a 33-byte compressed pubkey into affine (x,y). Returns 1 on a valid curve point.
__device__ int decompress(affpt *out, const uint8_t c[33]) {
    fe x;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint8_t *p = c + 1 + (7 - i) * 4;  // limb i (v[i]) comes from the i-th 4-byte group from the END
        x.v[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
    }
    // rhs = x^3 + 7
    fe x2, x3, rhs, seven; fe_set_zero(&seven); seven.v[0] = 7u;
    fe_sqr(&x2, &x); fe_mul(&x3, &x2, &x); fe_add(&rhs, &x3, &seven);
    fe y; fe_sqrt(&y, &rhs);
    fe yy; fe_sqr(&yy, &y);
    if (!fe_equal(&yy, &rhs)) return 0;             // not a valid x on the curve
    // fix parity: compressed prefix 0x02 => even y, 0x03 => odd y
    int want_odd = (c[0] & 1);
    if (fe_is_odd(&y) != want_odd) {                // y := p - y
        fe feP;
#pragma unroll
        for (int i = 0; i < 8; i++) feP.v[i] = P_LIMBS[i];
        fe ny; fe_sub(&ny, &feP, &y); fe_copy(&y, &ny);
    }
    fe_copy(&out->x, &x); fe_copy(&out->y, &y);
    return 1;
}

__device__ __forceinline__ void u64_to_k(uint32_t k[8], uint64_t v) {
#pragma unroll
    for (int i = 0; i < 8; i++) k[i] = 0u;
    k[0] = (uint32_t)v; k[1] = (uint32_t)(v >> 32);
}

// affine point = scalar*G (via verified jpoint scalar mult + to-affine). scalar must be in [1,n).
__device__ int affine_scalar_mul_G(affpt *out, uint64_t scalar) {
    uint32_t k[8]; u64_to_k(k, scalar);
    jpoint R; scalar_mul_G(&R, k);
    fe ax, ay;
    if (!jpoint_to_affine(&ax, &ay, &R)) return 0;  // infinity (scalar % n == 0)
    fe_copy(&out->x, &ax); fe_copy(&out->y, &ay);
    return 1;
}

// ----- setup kernels -------------------------------------------------------

// Single-thread: build the jump table, decompress Q, and form Q' = Q - lo*G.
// neg_lo[8] = (n - lo) little-endian u32 limbs (so neg_lo*G = -(lo*G)); computed on host.
__global__ void k_init_global(const uint8_t *Q_comp, const uint32_t *neg_lo,
                              const uint64_t *jump_scalars) {
    if (threadIdx.x || blockIdx.x) return;
    // jump table
    for (int i = 0; i < K_JUMPS; i++) {
        d_jumpscalar[i] = jump_scalars[i];
        affine_scalar_mul_G(&d_jump[i], jump_scalars[i]);
    }
    // decompress Q, then Q' = Q + (-lo*G)
    affpt Q; decompress(&Q, Q_comp);
    jpoint jQ; fe_copy(&jQ.X, &Q.x); fe_copy(&jQ.Y, &Q.y); fe_set_zero(&jQ.Z); jQ.Z.v[0] = 1u;
    uint32_t nl[8];
#pragma unroll
    for (int i = 0; i < 8; i++) nl[i] = neg_lo[i];
    jpoint negloG; scalar_mul_G(&negloG, nl);
    jpoint jQp; jpoint_add(&jQp, &jQ, &negloG);
    fe ax, ay; jpoint_to_affine(&ax, &ay, &jQp);
    fe_copy(&d_Qp.x, &ax); fe_copy(&d_Qp.y, &ay);
}

// Initialise each kangaroo's start point and tracked distance.
//   thread g in [0, M_t):        TAME kangaroo, start scalar = stride_t/2 + g*stride_t, point=start*G,
//                                tracked T = start.
//   thread g in [M_t, M_t+M_w):  WILD kangaroo j=g-M_t, offset = stride_w/2 + j*stride_w,
//                                point = Q' + offset*G, tracked distance D = offset.
__global__ void k_init_kangaroos(affpt *px, uint64_t *dist, uint32_t M_t, uint32_t M_w,
                                 uint64_t stride_t, uint64_t stride_w) {
    uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= M_t + M_w) return;
    if (g < M_t) {
        uint64_t start = stride_t / 2 + (uint64_t)g * stride_t;
        if (start == 0) start = 1;
        affine_scalar_mul_G(&px[g], start);
        dist[g] = start;
    } else {
        uint32_t j = g - M_t;
        uint64_t off = stride_w / 2 + (uint64_t)j * stride_w;
        // point = Q' + off*G
        jpoint jQp; fe_copy(&jQp.X, &d_Qp.x); fe_copy(&jQp.Y, &d_Qp.y); fe_set_zero(&jQp.Z); jQp.Z.v[0] = 1u;
        affpt offG;
        if (off == 0) { fe_copy(&px[g].x, &d_Qp.x); fe_copy(&px[g].y, &d_Qp.y); dist[g] = 0; return; }
        affine_scalar_mul_G(&offG, off);
        jpoint joff; fe_copy(&joff.X, &offG.x); fe_copy(&joff.Y, &offG.y); fe_set_zero(&joff.Z); joff.Z.v[0] = 1u;
        jpoint jr; jpoint_add(&jr, &jQp, &joff);
        fe ax, ay; jpoint_to_affine(&ax, &ay, &jr);
        fe_copy(&px[g].x, &ax); fe_copy(&px[g].y, &ay);
        dist[g] = off;
    }
}

// Negation-map init: same start layout, but canonicalize each start to (x, even-y) and record the
// SIGNED dlog-state. dlog(start) for tame = +scalar (a=0), for wild = d' (a=+1). Canonicalizing a
// start with odd y negates it: dlog -> -dlog, so w -> -w and a -> -a.
//   tame: a=0 always; if start negated, w = -start.
//   wild: a = +1 normally, -1 if Q'+off*G had odd y; w = (negated ? -off : off).
__global__ void k_init_kangaroos_negmap(affpt *px, int64_t *dist, int8_t *sgn,
                                       uint32_t M_t, uint32_t M_w,
                                       uint64_t stride_t, uint64_t stride_w) {
    uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= M_t + M_w) return;
    affpt S; int64_t w; int a;
    if (g < M_t) {
        uint64_t start = stride_t / 2 + (uint64_t)g * stride_t;
        if (start == 0) start = 1;
        affine_scalar_mul_G(&S, start);
        a = 0; w = (int64_t)start;
    } else {
        uint32_t j = g - M_t;
        uint64_t off = stride_w / 2 + (uint64_t)j * stride_w;
        jpoint jQp; fe_copy(&jQp.X, &d_Qp.x); fe_copy(&jQp.Y, &d_Qp.y); fe_set_zero(&jQp.Z); jQp.Z.v[0] = 1u;
        if (off == 0) { fe_copy(&S.x, &d_Qp.x); fe_copy(&S.y, &d_Qp.y); }
        else {
            affpt offG; affine_scalar_mul_G(&offG, off);
            jpoint joff; fe_copy(&joff.X, &offG.x); fe_copy(&joff.Y, &offG.y); fe_set_zero(&joff.Z); joff.Z.v[0] = 1u;
            jpoint jr; jpoint_add(&jr, &jQp, &joff);
            fe ax, ay; jpoint_to_affine(&ax, &ay, &jr);
            fe_copy(&S.x, &ax); fe_copy(&S.y, &ay);
        }
        a = 1; w = (int64_t)off;
    }
    // canonicalize: if odd y, negate point and flip the signed state
    if (fe_is_odd(&S.y)) {
        fe ny; fe_negate(&ny, &S.y); fe_copy(&S.y, &ny);
        w = -w; a = -a;
    }
    fe_copy(&px[g].x, &S.x); fe_copy(&px[g].y, &S.y);
    dist[g] = w; sgn[g] = (int8_t)a;
}

// ----- the walk ------------------------------------------------------------
// Each thread advances its kangaroo `steps` times. On a distinguished point (x has dp_bits trailing
// zero bits) it appends a DP record {x[8], dist, type} to the report buffer. State persists in global
// memory across rounds (the host relaunches until a tame/wild DP collision is found).
__global__ void k_walk(affpt *px, uint64_t *dist, uint32_t M_t, uint32_t M_w,
                       int steps, uint32_t dp_mask,
                       uint32_t *dp_x, uint64_t *dp_dist, uint8_t *dp_type, uint32_t cap) {
    uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= M_t + M_w) return;
    uint8_t type = (g < M_t) ? 0u : 1u;
    affpt P = px[g];
    uint64_t D = dist[g];

    for (int s = 0; s < steps; s++) {
        uint32_t idx = P.x.v[0] & (K_JUMPS - 1);
        // affine add P += J[idx]; needs 1/(J.x - P.x)
        fe dx; fe_sub(&dx, &d_jump[idx].x, &P.x);
        if (fe_is_zero(&dx)) {
            aff_add_fallback(&P, &P, &d_jump[idx]);  // P == ±J[idx] (rare): verified Jacobian add
        } else {
            fe inv; fe_inv(&inv, &dx);
            affpt R; aff_add_with_inv(&R, &P, &d_jump[idx], &inv);
            P = R;
        }
        D += d_jumpscalar[idx];

        if ((P.x.v[0] & dp_mask) == 0u) {            // distinguished point -> report
            uint32_t at = atomicAdd(&d_dpcount, 1u);
            if (at < cap) {
#pragma unroll
                for (int l = 0; l < 8; l++) dp_x[at * 8 + l] = P.x.v[l];
                dp_dist[at] = D;
                dp_type[at] = type;
            }
        }
    }
    px[g] = P;
    dist[g] = D;
}

// ----- batched-inversion walk (fast path) ----------------------------------
// Each THREAD owns B kangaroos (B = WALK_B). One Fermat inverse + ~3B muls inverts all B step
// denominators (Montgomery's trick), collapsing the per-kangaroo inverse cost from ~270 muls to
// ~270/B + 2. This is the dominant throughput lever (the per-step k_walk pays a full inverse/step).
//
// Layout: thread t owns kangaroos [t*B, t*B+B). The same global px[]/dist[] arrays are used; we just
// stride by B per thread. Each kangaroo keeps its own type (tame if global index < M_t).
// All field math reuses the BIT-EXACT primitives (aff_add_with_inv / aff_add_fallback / fe_inv).
#ifndef WALK_B
#define WALK_B 32   // kangaroos per thread for batched inversion. Swept 8/16/32/64 on RTX 5090:
#endif              // throughput 0.56/0.91/1.05/0.43 G jumps/s — 32 is the occupancy sweet spot
                    // (64 spills the per-thread affpt[B]+pre[B+1] frame to local mem). See notes.
__global__ void k_walk_batch(affpt *px, uint64_t *dist, uint32_t M_t, uint32_t M_w,
                             int steps, uint32_t dp_mask,
                             uint32_t *dp_x, uint64_t *dp_dist, uint8_t *dp_type, uint32_t cap) {
    const int B = WALK_B;
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base = t * B;
    uint32_t M = M_t + M_w;
    if (base >= M) return;
    int nb = B; if (base + nb > M) nb = M - base;   // tail thread may own < B kangaroos

    affpt P[WALK_B];
    uint64_t D[WALK_B];
    uint32_t idx[WALK_B];
#pragma unroll
    for (int i = 0; i < WALK_B; i++) if (i < nb) { P[i] = px[base + i]; D[i] = dist[base + i]; }

    for (int s = 0; s < steps; s++) {
        // forward pass: jump index per kangaroo, denominator dd[i] = J[idx].x - P[i].x, prefix products
        fe dd[WALK_B];
        fe pre[WALK_B + 1];           // pre[0]=1, pre[i+1]=pre[i]*dd[i] (zeros substituted by 1)
        uint8_t zero[WALK_B];
        fe_set_zero(&pre[0]); pre[0].v[0] = 1u;
        for (int i = 0; i < nb; i++) {
            idx[i] = P[i].x.v[0] & (K_JUMPS - 1);
            fe_sub(&dd[i], &d_jump[idx[i]].x, &P[i].x);
            if (fe_is_zero(&dd[i])) { zero[i] = 1u; fe_copy(&pre[i + 1], &pre[i]); }
            else                    { zero[i] = 0u; fe_mul(&pre[i + 1], &pre[i], &dd[i]); }
        }
        fe u; fe_inv(&u, &pre[nb]);    // 1 / (product of all NONZERO denominators)
        // backward pass: inv(dd[i]) = u * pre[i]; then strip: u *= dd[i]
        for (int i = nb - 1; i >= 0; i--) {
            if (zero[i]) {
                aff_add_fallback(&P[i], &P[i], &d_jump[idx[i]]);   // P==±J (rare), u unchanged
            } else {
                fe inv_i; fe_mul(&inv_i, &u, &pre[i]);
                affpt R; aff_add_with_inv(&R, &P[i], &d_jump[idx[i]], &inv_i);
                P[i] = R;
                fe nu; fe_mul(&nu, &u, &dd[i]); fe_copy(&u, &nu);
            }
            D[i] += d_jumpscalar[idx[i]];
        }
        // DP check for all kangaroos in the batch
        for (int i = 0; i < nb; i++) {
            if ((P[i].x.v[0] & dp_mask) == 0u) {
                uint32_t at = atomicAdd(&d_dpcount, 1u);
                if (at < cap) {
#pragma unroll
                    for (int l = 0; l < 8; l++) dp_x[at * 8 + l] = P[i].x.v[l];
                    dp_dist[at] = D[i];
                    dp_type[at] = (base + i < M_t) ? 0u : 1u;
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < WALK_B; i++) if (i < nb) { px[base + i] = P[i]; dist[base + i] = D[i]; }
}

// ----- negation-map walk (batched inversion + canonicalization) ------------
// The negation map folds {P,-P} onto the canonical rep (x, EVEN-y), shrinking the walk space ~2x
// (~sqrt(2) fewer jumps). Bookkeeping (proven in kangaroo_negmap_ref.py): each kangaroo carries a
// SIGNED dlog-state dlog(C) = a*d' + w, a in {0 (tame), +-1 (wild)}, w a signed int64 (never reduced
// mod n — stays << sqrt(W) << n). A step C -> canon(C + J[idx]) with canon-sign eps in {+1,-1} maps
//   (a, w) -> (eps*a, eps*(w + s_idx)).
// Fruitless-cycle fix: a 2-cycle (the dominant hazard, ~1/(2K)/step) is detected by x == x_2-ago and
// escaped with the ALTERNATE jump idx^1 — a PURE FUNCTION of the point, so two kangaroos sharing the
// cycle pick the same alternate and MERGE (preserving the collision property). At GPU scale (32+ bit)
// the tiny-space false-firing that hurt the serial reference cannot occur.
// Distances are int64_t here; sgn[] holds a in {+1 wild, -1 wild-negated, 0 tame} packed as int8.
__global__ void k_walk_negmap(affpt *px, int64_t *dist, int8_t *sgn, uint32_t M_t, uint32_t M_w,
                             int steps, uint32_t dp_mask,
                             uint32_t *dp_x, int64_t *dp_dist, int8_t *dp_sgn, uint8_t *dp_type,
                             uint32_t cap) {
    const int B = WALK_B;
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base = t * B;
    uint32_t M = M_t + M_w;
    if (base >= M) return;
    int nb = B; if (base + nb > M) nb = M - base;

    affpt P[WALK_B];
    int64_t D[WALK_B];
    int8_t  A[WALK_B];
    uint32_t prev2[WALK_B];          // low-limb x from two steps ago (cycle detection)
    uint32_t prev1[WALK_B];
    uint32_t idx[WALK_B];
#pragma unroll
    for (int i = 0; i < WALK_B; i++) if (i < nb) {
        P[i] = px[base + i]; D[i] = dist[base + i]; A[i] = sgn[base + i];
        prev2[i] = 0xFFFFFFFFu; prev1[i] = 0xFFFFFFFFu;
    }

    for (int s = 0; s < steps; s++) {
        fe dd[WALK_B];
        fe pre[WALK_B + 1];
        uint8_t zero[WALK_B];
        fe_set_zero(&pre[0]); pre[0].v[0] = 1u;
        for (int i = 0; i < nb; i++) {
            uint32_t curx = P[i].x.v[0];
            idx[i] = curx & (K_JUMPS - 1);
            if (curx == prev2[i]) { idx[i] ^= 1u; atomicAdd(&d_escapes, 1ull); }  // 2-cycle escape
            fe_sub(&dd[i], &d_jump[idx[i]].x, &P[i].x);
            if (fe_is_zero(&dd[i])) { zero[i] = 1u; fe_copy(&pre[i + 1], &pre[i]); }
            else                    { zero[i] = 0u; fe_mul(&pre[i + 1], &pre[i], &dd[i]); }
        }
        fe u; fe_inv(&u, &pre[nb]);
        for (int i = nb - 1; i >= 0; i--) {
            affpt R;
            if (zero[i]) {
                aff_add_fallback(&R, &P[i], &d_jump[idx[i]]);
            } else {
                fe inv_i; fe_mul(&inv_i, &u, &pre[i]);
                aff_add_with_inv(&R, &P[i], &d_jump[idx[i]], &inv_i);
                fe nu; fe_mul(&nu, &u, &dd[i]); fe_copy(&u, &nu);
            }
            // canonicalize R to (x, even-y); eps = -1 if we negated
            int eps = 1;
            if (fe_is_odd(&R.y)) { fe ny; fe_negate(&ny, &R.y); fe_copy(&R.y, &ny); eps = -1; }
            // update signed dlog-state: (a,w) -> (eps*a, eps*(w + s_idx))
            int64_t s_idx = (int64_t)d_jumpscalar[idx[i]];
            D[i] = (eps < 0) ? -(D[i] + s_idx) : (D[i] + s_idx);
            A[i] = (int8_t)(eps * (int)A[i]);
            // shift cycle-detection history
            prev2[i] = prev1[i]; prev1[i] = P[i].x.v[0];
            P[i] = R;
        }
        for (int i = 0; i < nb; i++) {
            if ((P[i].x.v[0] & dp_mask) == 0u) {
                uint32_t at = atomicAdd(&d_dpcount, 1u);
                if (at < cap) {
#pragma unroll
                    for (int l = 0; l < 8; l++) dp_x[at * 8 + l] = P[i].x.v[l];
                    dp_dist[at] = D[i];
                    dp_sgn[at]  = A[i];
                    dp_type[at] = (base + i < M_t) ? 0u : 1u;
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < WALK_B; i++) if (i < nb) {
        px[base + i] = P[i]; dist[base + i] = D[i]; sgn[base + i] = A[i];
    }
}

// ----- final verification kernel -------------------------------------------
// Confirm cand*G == Q (compressed) bit-exact before the host prints the answer.
__global__ void k_verify(uint64_t cand, const uint8_t *Q_comp, int *ok) {
    if (threadIdx.x || blockIdx.x) return;
    uint32_t k[8]; u64_to_k(k, cand);
    jpoint R; scalar_mul_G(&R, k);
    fe ax, ay;
    if (!jpoint_to_affine(&ax, &ay, &R)) { *ok = 0; return; }
    uint8_t comp[33]; compressed_pubkey(comp, &ax, &ay);
    int eq = 1;
    for (int i = 0; i < 33; i++) if (comp[i] != Q_comp[i]) eq = 0;
    *ok = eq;
}

// ===========================================================================
// Host
// ===========================================================================
#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(2); } } while (0)

static uint64_t parse_u64_hex(const char *s) {
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s += 2;
    return strtoull(s, NULL, 16);
}

// n - lo as 8 little-endian u32 limbs (lo fits in u64).
static void neg_lo_limbs(uint64_t lo, uint32_t out[8]) {
    // n (group order), little-endian u32 limbs
    static const uint32_t Nl[8] = {
        0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
        0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };
    uint64_t lo_l[8] = {0}; lo_l[0] = (uint32_t)lo; lo_l[1] = (uint32_t)(lo >> 32);
    int64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        int64_t t = (int64_t)Nl[i] - (int64_t)lo_l[i] - borrow;
        if (t < 0) { t += (1LL << 32); borrow = 1; } else borrow = 0;
        out[i] = (uint32_t)t;
    }
}

static uint64_t isqrt_u64(uint64_t n) {
    if (n == 0) return 0;
    uint64_t x = (uint64_t)sqrt((double)n);
    while (x * x > n) x--;
    while ((x + 1) * (x + 1) <= n) x++;
    return x;
}

static long env_long(const char *name, long defv) {
    const char *v = getenv(name);
    return v ? strtol(v, NULL, 0) : defv;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <lo_hex> <hi_hex> <Q_compressed_hex>\n", argv[0]); return 2; }
    uint64_t lo = parse_u64_hex(argv[1]);
    uint64_t hi = parse_u64_hex(argv[2]);
    const char *qhex = argv[3];
    if (strlen(qhex) != 66) { fprintf(stderr, "Q must be 33-byte compressed hex (66 chars)\n"); return 2; }
    uint8_t Q[33];
    for (int i = 0; i < 33; i++) { unsigned b; sscanf(qhex + 2 * i, "%2x", &b); Q[i] = (uint8_t)b; }
    if (hi <= lo) { fprintf(stderr, "need hi > lo\n"); return 2; }
    uint64_t W = hi - lo;
    uint64_t sqrtW = isqrt_u64(W);

    // ---- parameters (env-overridable for tuning / scaling experiments) ----
    // Herd is PER SIDE (M_t tame + M_w wild). Two regimes:
    //   * scaling measurement: a MODEST herd with M << sqrtW keeps us in the linear-speedup
    //     regime (total jumps ~ c*sqrt(W), independent of M) so the sqrt(n) law is cleanly
    //     visible; KANG_HERD picks it.
    //   * peak throughput: a LARGE herd saturates the GPU (set KANG_HERD high on a big interval).
    long herd = env_long("KANG_HERD", 1L << 14);
    uint32_t M_t = (uint32_t)herd; if ((uint64_t)M_t > sqrtW + 1) M_t = (uint32_t)(sqrtW + 1);
    if (M_t < 64) M_t = 64;
    uint32_t M_w = M_t;
    uint32_t M = M_t + M_w;
    uint64_t mean = env_long("KANG_MEAN", 0);
    if (mean == 0) mean = sqrtW / 2; if (mean < 1) mean = 1;

    // DP bits: total stored DPs ~ (c*sqrt(W))/2^dp; keep it ~<= 2^20 so the host tables and the
    // per-round report buffer never overflow. bitlen(sqrtW)-18 gives dp=0 for tiny intervals
    // (store-everything, instant) and grows ~1 bit per +2 interval bits at scale.
    int sqrtW_bits = 0; { uint64_t t = sqrtW; while (t) { sqrtW_bits++; t >>= 1; } }
    long dp_override = env_long("KANG_DP", -1);
    int dp_bits = (dp_override >= 0) ? (int)dp_override : (sqrtW_bits > 18 ? sqrtW_bits - 18 : 0);
    if (dp_bits > 31) dp_bits = 31;
    uint32_t dp_mask = (dp_bits >= 32) ? 0xFFFFFFFFu : ((1u << dp_bits) - 1u);
    uint64_t dp_period = 1ull << dp_bits;  // expected jumps between DPs per kangaroo

    // Expected per-kangaroo steps to solve ~ c*sqrt(W)/M (+ DP overshoot 2^dp). Use c=8 (loose
    // upper bound for this herd layout) to size rounds and budget so we never false-fail.
    uint64_t est_per_kang = 8 * (sqrtW / (M ? M : 1)) + dp_period + 1;

    // steps_per_round: aim for ~16-64 rounds (fine granularity for the scaling curve) while
    // bounding the per-round DP buffer: M*steps/2^dp <= cap.
    uint64_t buf_cap_steps = ((uint64_t)KANG_CAP * dp_period) / (M ? M : 1);
    long steps_override = env_long("KANG_STEPS", -1);
    uint64_t steps_per_round_u = (steps_override > 0) ? (uint64_t)steps_override : (est_per_kang / 16 + 1);
    if (steps_per_round_u < 64) steps_per_round_u = 64;
    if (steps_per_round_u > buf_cap_steps && buf_cap_steps > 0) steps_per_round_u = buf_cap_steps;
    if (steps_per_round_u < 1) steps_per_round_u = 1;
    int steps_per_round = (int)steps_per_round_u;

    // per-kangaroo step budget: 32x the estimate (generous) so a slow-tail run still completes.
    uint64_t budget = (uint64_t)env_long("KANG_BUDGET", 0);
    if (budget == 0) budget = 32 * est_per_kang + 200000;

    fprintf(stderr, "[kangaroo] W=2^%.1f sqrtW=%llu M=%u (t=%u,w=%u) mean=%llu dp_bits=%d steps/round=%d budget=%llu\n",
            log2((double)W), (unsigned long long)sqrtW, M, M_t, M_w,
            (unsigned long long)mean, dp_bits, steps_per_round, (unsigned long long)budget);

    // ---- jump scalars (deterministic LCG, secret-independent), in [1, 2*mean] ----
    uint64_t jscal[K_JUMPS];
    uint64_t st = 0x9E3779B97F4A7C15ULL;
    uint64_t span = 2 * mean; if (span < 1) span = 1;
    for (int i = 0; i < K_JUMPS; i++) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        jscal[i] = 1 + (st % span);
    }

    // ---- device allocations ----
    uint8_t *d_Q; CUDA_OK(cudaMalloc(&d_Q, 33));
    CUDA_OK(cudaMemcpy(d_Q, Q, 33, cudaMemcpyHostToDevice));
    uint32_t neglo[8]; neg_lo_limbs(lo, neglo);
    uint32_t *d_neglo; CUDA_OK(cudaMalloc(&d_neglo, 32));
    CUDA_OK(cudaMemcpy(d_neglo, neglo, 32, cudaMemcpyHostToDevice));
    uint64_t *d_jscal; CUDA_OK(cudaMalloc(&d_jscal, K_JUMPS * 8));
    CUDA_OK(cudaMemcpy(d_jscal, jscal, K_JUMPS * 8, cudaMemcpyHostToDevice));

    affpt *d_px; CUDA_OK(cudaMalloc(&d_px, (size_t)M * sizeof(affpt)));

    uint32_t cap = KANG_CAP;
    uint32_t *d_dpx; CUDA_OK(cudaMalloc(&d_dpx, (size_t)cap * 8 * sizeof(uint32_t)));
    uint8_t *d_dptype; CUDA_OK(cudaMalloc(&d_dptype, (size_t)cap * sizeof(uint8_t)));
    int *d_ok; CUDA_OK(cudaMalloc(&d_ok, sizeof(int)));
#ifdef NEGMAP
    int64_t *d_dist; CUDA_OK(cudaMalloc(&d_dist, (size_t)M * sizeof(int64_t)));
    int8_t  *d_sgn;  CUDA_OK(cudaMalloc(&d_sgn,  (size_t)M * sizeof(int8_t)));
    int64_t *d_dpdist; CUDA_OK(cudaMalloc(&d_dpdist, (size_t)cap * sizeof(int64_t)));
    int8_t  *d_dpsgn;  CUDA_OK(cudaMalloc(&d_dpsgn,  (size_t)cap * sizeof(int8_t)));
#else
    uint64_t *d_dist; CUDA_OK(cudaMalloc(&d_dist, (size_t)M * sizeof(uint64_t)));
    uint64_t *d_dpdist; CUDA_OK(cudaMalloc(&d_dpdist, (size_t)cap * sizeof(uint64_t)));
#endif

    // ---- setup ----
    k_init_global<<<1, 1>>>(d_Q, d_neglo, d_jscal);
    CUDA_OK(cudaDeviceSynchronize());
    uint64_t stride_t = W / M_t; if (stride_t < 1) stride_t = 1;
    uint64_t stride_w = W / M_w; if (stride_w < 1) stride_w = 1;
    int threads = 256, blocks = (M + threads - 1) / threads;
#ifdef NEGMAP
    k_init_kangaroos_negmap<<<blocks, threads>>>(d_px, d_dist, d_sgn, M_t, M_w, stride_t, stride_w);
#else
    k_init_kangaroos<<<blocks, threads>>>(d_px, d_dist, M_t, M_w, stride_t, stride_w);
#endif
    CUDA_OK(cudaDeviceSynchronize());

    // ---- host DP tables (accumulate across rounds) ----
    // Tame stores (a,w)=(0,m); wild stores (a,w), a in {+-1}. Collision at same canonical x:
    //   a_t*d'+w_t = a_w*d'+w_w  => d' = (w_w-w_t)/(a_t-a_w). For the base path (no NEGMAP) tame a=0,
    //   wild a=+1 => d' = w_t - w_w = T - D, identical to the original solver.
    struct DPVal { int64_t w; int8_t a; };
    std::unordered_map<std::string, DPVal> tame_map, wild_map;
    std::vector<uint32_t> hx((size_t)cap * 8);
    std::vector<uint8_t> htype(cap);
#ifdef NEGMAP
    std::vector<int64_t> hdist(cap);
    std::vector<int8_t>  hsgn(cap);
#else
    std::vector<uint64_t> hdist(cap);
#endif

    uint64_t total_steps = 0;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    int walk_b = WALK_B;
    int blocks_batch = (int)(((uint64_t)M + (uint64_t)threads * walk_b - 1) / ((uint64_t)threads * walk_b));
#ifdef NEGMAP
    fprintf(stderr, "[kangaroo] walk=negation-map (batched-inversion B=%d)\n", walk_b);
#else
    // walk kernel selection: KANG_BATCH=1 (default) batched-inversion; 0 = per-step-inverse (A/B).
    int use_batch = (int)env_long("KANG_BATCH", 1);
    fprintf(stderr, "[kangaroo] walk=%s B=%d\n",
            use_batch ? "batched-inversion" : "per-step-inverse", use_batch ? walk_b : 1);
#endif

    uint64_t found_cand = 0; int solved = 0;
    auto try_pair = [&](int8_t a_t, int64_t w_t, int8_t a_w, int64_t w_w) -> int {
        int da = (int)a_t - (int)a_w;
        if (da == 0) return 0;
        int64_t num = w_w - w_t;
        if (num % da != 0) return 0;
        int64_t dprime = num / da;
        if (dprime < 0) return 0;
        uint64_t cand = lo + (uint64_t)dprime;
        if (cand >= lo && cand <= hi) { found_cand = cand; return 1; }
        return 0;
    };

    while (total_steps < budget && !solved) {
        uint32_t zero = 0; CUDA_OK(cudaMemcpyToSymbol(d_dpcount, &zero, sizeof(uint32_t)));
#ifdef NEGMAP
        k_walk_negmap<<<blocks_batch, threads>>>(d_px, d_dist, d_sgn, M_t, M_w, steps_per_round,
                                                 dp_mask, d_dpx, d_dpdist, d_dpsgn, d_dptype, cap);
#else
        if (use_batch)
            k_walk_batch<<<blocks_batch, threads>>>(d_px, d_dist, M_t, M_w, steps_per_round, dp_mask,
                                                    d_dpx, d_dpdist, d_dptype, cap);
        else
            k_walk<<<blocks, threads>>>(d_px, d_dist, M_t, M_w, steps_per_round, dp_mask,
                                        d_dpx, d_dpdist, d_dptype, cap);
#endif
        CUDA_OK(cudaDeviceSynchronize());
        total_steps += steps_per_round;

        uint32_t cnt = 0; CUDA_OK(cudaMemcpyFromSymbol(&cnt, d_dpcount, sizeof(uint32_t)));
        if (cnt > cap) cnt = cap;
        if (cnt == 0) continue;
        CUDA_OK(cudaMemcpy(hx.data(), d_dpx, (size_t)cnt * 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(htype.data(), d_dptype, (size_t)cnt * sizeof(uint8_t), cudaMemcpyDeviceToHost));
#ifdef NEGMAP
        CUDA_OK(cudaMemcpy(hdist.data(), d_dpdist, (size_t)cnt * sizeof(int64_t), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(hsgn.data(), d_dpsgn, (size_t)cnt * sizeof(int8_t), cudaMemcpyDeviceToHost));
#else
        CUDA_OK(cudaMemcpy(hdist.data(), d_dpdist, (size_t)cnt * sizeof(uint64_t), cudaMemcpyDeviceToHost));
#endif

        for (uint32_t i = 0; i < cnt && !solved; i++) {
            std::string key((const char *)&hx[(size_t)i * 8], 32);
#ifdef NEGMAP
            int64_t w = hdist[i]; int8_t a = hsgn[i];
#else
            int64_t w = (int64_t)hdist[i]; int8_t a = (htype[i] == 0) ? 0 : 1;  // tame a=0, wild a=+1
#endif
            if (htype[i] == 0) {  // tame
                auto it = wild_map.find(key);
                if (it != wild_map.end() && try_pair(a, w, it->second.a, it->second.w)) { solved = 1; break; }
                if (tame_map.find(key) == tame_map.end()) tame_map[key] = {w, a};
            } else {              // wild
                auto it = tame_map.find(key);
                if (it != tame_map.end() && try_pair(it->second.a, it->second.w, a, w)) { solved = 1; break; }
                if (wild_map.find(key) == wild_map.end()) wild_map[key] = {w, a};
            }
        }
    }

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    double jumps = (double)total_steps * (double)M;
    fprintf(stderr, "[kangaroo] rounds done: total_steps/kangaroo=%llu jumps=%.3e time=%.1fms jumps/s=%.3e\n",
            (unsigned long long)total_steps, jumps, ms, jumps / (ms / 1000.0));
#ifdef NEGMAP
    { unsigned long long esc = 0; cudaMemcpyFromSymbol(&esc, d_escapes, sizeof(esc));
      fprintf(stderr, "[kangaroo] negmap escapes fired: %llu (%.4f per jump)\n",
              esc, (double)esc / (jumps > 0 ? jumps : 1)); }
#endif

    fprintf(stderr, "[kangaroo] distinct DPs: tame=%zu wild=%zu\n", tame_map.size(), wild_map.size());
    if (!solved) { fprintf(stderr, "[kangaroo] FAILED: no collision within budget\n"); return 1; }

    // ---- final bit-exact verification on device: cand*G == Q ----
    int hok = 0;
    k_verify<<<1, 1>>>(found_cand, d_Q, d_ok);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(&hok, d_ok, sizeof(int), cudaMemcpyDeviceToHost));
    if (!hok) { fprintf(stderr, "[kangaroo] candidate failed cand*G==Q verification!\n"); return 1; }

    printf("d=%llx\n", (unsigned long long)found_cand);
    fprintf(stderr, "[kangaroo] SOLVED d=0x%llx (verified cand*G==Q)\n", (unsigned long long)found_cand);
    return 0;
}
