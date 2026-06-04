// secp256k1_affine.cuh — Wave 3 fast path: affine point ops + batched Montgomery inversion.
//
// WHY THIS EXISTS (ADR-0003 Track A fast path):
//   The Wave-1 baseline recomputes a full 256-step double-and-add k·G for EVERY candidate key,
//   even though consecutive keys differ by just +G. The fast path walks a CONTIGUOUS sub-range
//   with ONE affine point-addition per key. Affine addition needs a modular inverse per step
//   (the ~270-multiply Fermat inverse — the single most expensive field op), so a naive affine
//   walk would be SLOWER than Jacobian. Montgomery's trick fixes that: it inverts a whole batch
//   of N denominators with ONE inverse + 3N multiplies, so the per-key inverse cost collapses to
//   ~(270/N + 3) muls. "Affine beats Jacobian at N>20" (ADR-0003) is exactly this crossover.
//
// CORRECTNESS-CRITICAL (the most bug-prone path in Wave 3):
//   The batch denominator for lane j is d_j = (jG).x - base.x. It is ZERO exactly when
//   base == ±(jG), i.e. when the walk would double a point or land on the point at infinity.
//   A single zero kills the running product. We substitute 1 for any zero in the product chain
//   (preserving every OTHER lane's inverse), flag those lanes, and resolve them with the verified
//   Jacobian add (which correctly handles doubling and infinity). Walking from key 1 deliberately
//   drives a zero (key 2 = G+G is a doubling) so the oracle comparison exercises this path.
//
// All arithmetic reuses the BIT-EXACT field/EC primitives from secp256k1_field.cuh / _ec.cuh.
// No new field math is introduced — only the affine add and the two-pass batch inverse.
#ifndef SECP256K1_AFFINE_CUH
#define SECP256K1_AFFINE_CUH

#include "secp256k1_field.cuh"
#include "secp256k1_ec.cuh"

#ifndef BATCH
#define BATCH 256  // batch size per thread; tunable via -DBATCH=. Bigger = better inverse
#endif             // amortization, more local-memory pressure. Swept empirically (16..512) in the
                   // Wave-3 notes: throughput plateaus ~256–512; 256 is the robust default.

typedef struct { fe x, y; } affpt;   // affine point (the walk never stores infinity here)

// Affine add R = P + Q given a PRECOMPUTED inverse of (Q.x - P.x). Assumes P != ±Q
// (the zero-denominator case is routed to aff_add_fallback instead).
//   λ  = (Q.y - P.y) / (Q.x - P.x)
//   X3 = λ² - P.x - Q.x
//   Y3 = λ·(P.x - X3) - P.y
__device__ __forceinline__ void aff_add_with_inv(affpt *r, const affpt *p,
                                                  const affpt *q, const fe *inv_dx) {
    fe num, lam, t;
    fe_sub(&num, &q->y, &p->y);   // Q.y - P.y
    fe_mul(&lam, &num, inv_dx);   // λ
    fe_sqr(&t, &lam);             // λ²
    fe_sub(&t, &t, &p->x);        // λ² - P.x
    fe_sub(&r->x, &t, &q->x);     // - Q.x   => X3
    fe_sub(&t, &p->x, &r->x);     // P.x - X3
    fe_mul(&lam, &lam, &t);       // λ·(P.x - X3)
    fe_sub(&r->y, &lam, &p->y);   // - P.y   => Y3
}

// Robust fallback for a zero denominator (P == ±Q): defer to the VERIFIED Jacobian add,
// which resolves doubling (P == Q) and the point at infinity (P == -Q) correctly. Rare —
// only at keyspace boundaries — so its per-call cost (incl. a Fermat inverse) is irrelevant.
// For infinity (an invalid key k ≡ 0 mod n, never a real puzzle key) we emit (0,0) as a
// sentinel; such a key is never a valid target.
__device__ void aff_add_fallback(affpt *r, const affpt *p, const affpt *q) {
    jpoint jp, jq, jr;
    fe_copy(&jp.X, &p->x); fe_copy(&jp.Y, &p->y); fe_set_zero(&jp.Z); jp.Z.v[0] = 1u;
    fe_copy(&jq.X, &q->x); fe_copy(&jq.Y, &q->y); fe_set_zero(&jq.Z); jq.Z.v[0] = 1u;
    jpoint_add(&jr, &jp, &jq);
    fe ax, ay;
    if (jpoint_to_affine(&ax, &ay, &jr)) { fe_copy(&r->x, &ax); fe_copy(&r->y, &ay); }
    else { fe_set_zero(&r->x); fe_set_zero(&r->y); }
}

// Forward pass of Montgomery's trick: build prefix products of the batch denominators
//   d_i = table[i+1].x - base.x   (lane i corresponds to multiple (i+1)·G)
//   prod[0] = 1,  prod[i+1] = prod[i] * d_i   (ZEROS substituted by 1, flagged in `zero[]`)
// After this, prod[N] = product of all NONZERO denominators (always != 0 mod p, so its
// Fermat inverse is well-defined — fe_inv(0) is never called).
__device__ __forceinline__ void batch_build_prod(fe prod[BATCH + 1], uint8_t zero[BATCH],
                                                  const affpt *base, const affpt *table, int N) {
    fe_set_zero(&prod[0]); prod[0].v[0] = 1u;
    for (int i = 0; i < N; i++) {
        fe d;
        fe_sub(&d, &table[i + 1].x, &base->x);
        if (fe_is_zero(&d)) { zero[i] = 1u; fe_copy(&prod[i + 1], &prod[i]); }
        else                { zero[i] = 0u; fe_mul(&prod[i + 1], &prod[i], &d); }
    }
}

// One batched step: given an affine `base` at key K, fill results[i] with the affine point for
// key K+(i+1) (multiple i+1 = base + (i+1)·G), for i = 0..BATCH-1, using ONE Fermat inverse for
// the whole batch (Montgomery's trick). results[BATCH-1] is the advance point (key K+BATCH).
//
// The backward pass is FUSED with the affine add (no separate inverse array → less local memory):
//   u = 1/(product of nonzero d's); walking high→low, inv(d_i)=u·prod[i], then u·=d_i to strip it.
// Zero-denominator lanes (base == ±(i+1)·G) skip the product factor (it was 1) and use the verified
// Jacobian fallback — this is the single most bug-prone path in Wave 3.
__device__ void affine_step(affpt *results, const affpt *base, const affpt *table) {
    fe prod[BATCH + 1];
    uint8_t zero[BATCH];
    batch_build_prod(prod, zero, base, table, BATCH);

    fe u;
    fe_inv(&u, &prod[BATCH]);          // 1 / (product of all NONZERO denominators) — never inv(0)
    for (int i = BATCH - 1; i >= 0; i--) {
        if (zero[i]) {
            aff_add_fallback(&results[i], base, &table[i + 1]);   // doubling / infinity, u unchanged
        } else {
            fe inv_i;
            fe_mul(&inv_i, &u, &prod[i]);                          // 1/d_i
            aff_add_with_inv(&results[i], base, &table[i + 1], &inv_i);
            fe d;
            fe_sub(&d, &table[i + 1].x, &base->x);                 // recompute nonzero d_i
            fe_mul(&u, &u, &d);                                    // strip this factor from u
        }
    }
}

// Build the shared multiples table: table[j] = j·G (affine) for j = 1..BATCH.
// One-time, single-thread; cost amortized over the entire benchmark.
__global__ void kfast_build_table(affpt *table) {
    if (threadIdx.x || blockIdx.x) return;
    jpoint acc, G;
#pragma unroll
    for (int i = 0; i < 8; i++) acc.X.v[i] = GX_LIMBS[i];
#pragma unroll
    for (int i = 0; i < 8; i++) acc.Y.v[i] = GY_LIMBS[i];
    fe_set_zero(&acc.Z); acc.Z.v[0] = 1u;
    G = acc;
    for (int j = 1; j <= BATCH; j++) {
        fe ax, ay;
        jpoint_to_affine(&ax, &ay, &acc);
        fe_copy(&table[j].x, &ax); fe_copy(&table[j].y, &ay);
        if (j < BATCH) { jpoint t; jpoint_add(&t, &acc, &G); acc = t; }
    }
}

#endif // SECP256K1_AFFINE_CUH
