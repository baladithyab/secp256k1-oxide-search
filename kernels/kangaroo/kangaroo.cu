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
    long herd = env_long("KANG_HERD", 1L << 16);
    uint32_t M_t = (uint32_t)herd; if ((uint64_t)M_t > sqrtW + 1) M_t = (uint32_t)(sqrtW + 1);
    if (M_t < 64) M_t = 64;
    uint32_t M_w = M_t;
    uint32_t M = M_t + M_w;
    uint64_t mean = env_long("KANG_MEAN", 0);
    if (mean == 0) mean = sqrtW / 2; if (mean < 1) mean = 1;
    int steps_per_round = (int)env_long("KANG_STEPS", 1024);
    // DP bits: keep (M * 2^dp)/sqrtW small (~1/8). dp = floor(log2(sqrtW/(8*M))), clamped >= 0.
    long dp_override = env_long("KANG_DP", -1);
    int dp_bits;
    if (dp_override >= 0) dp_bits = (int)dp_override;
    else { double r = (double)sqrtW / (8.0 * (double)M); dp_bits = (r >= 1.0) ? (int)floor(log2(r)) : 0; }
    if (dp_bits > 31) dp_bits = 31;
    uint32_t dp_mask = (dp_bits >= 32) ? 0xFFFFFFFFu : ((1u << dp_bits) - 1u);

    // per-kangaroo step budget: generous multiple of expected 2*sqrtW/M_t
    uint64_t budget = (uint64_t)env_long("KANG_BUDGET", 0);
    if (budget == 0) budget = 64 * (sqrtW / (M_t ? M_t : 1) + 1) + 200000;

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
    uint64_t *d_dist; CUDA_OK(cudaMalloc(&d_dist, (size_t)M * sizeof(uint64_t)));

    uint32_t cap = KANG_CAP;
    uint32_t *d_dpx; CUDA_OK(cudaMalloc(&d_dpx, (size_t)cap * 8 * sizeof(uint32_t)));
    uint64_t *d_dpdist; CUDA_OK(cudaMalloc(&d_dpdist, (size_t)cap * sizeof(uint64_t)));
    uint8_t *d_dptype; CUDA_OK(cudaMalloc(&d_dptype, (size_t)cap * sizeof(uint8_t)));
    int *d_ok; CUDA_OK(cudaMalloc(&d_ok, sizeof(int)));

    // ---- setup ----
    k_init_global<<<1, 1>>>(d_Q, d_neglo, d_jscal);
    CUDA_OK(cudaDeviceSynchronize());
    uint64_t stride_t = W / M_t; if (stride_t < 1) stride_t = 1;
    uint64_t stride_w = W / M_w; if (stride_w < 1) stride_w = 1;
    int threads = 256, blocks = (M + threads - 1) / threads;
    k_init_kangaroos<<<blocks, threads>>>(d_px, d_dist, M_t, M_w, stride_t, stride_w);
    CUDA_OK(cudaDeviceSynchronize());

    // ---- host DP tables (accumulate across rounds) ----
    std::unordered_map<std::string, uint64_t> tame_map, wild_map;
    std::vector<uint32_t> hx((size_t)cap * 8);
    std::vector<uint64_t> hdist(cap);
    std::vector<uint8_t> htype(cap);

    uint64_t total_steps = 0;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    uint64_t found_cand = 0; int solved = 0;
    while (total_steps < budget && !solved) {
        uint32_t zero = 0; CUDA_OK(cudaMemcpyToSymbol(d_dpcount, &zero, sizeof(uint32_t)));
        k_walk<<<blocks, threads>>>(d_px, d_dist, M_t, M_w, steps_per_round, dp_mask,
                                    d_dpx, d_dpdist, d_dptype, cap);
        CUDA_OK(cudaDeviceSynchronize());
        total_steps += steps_per_round;

        uint32_t cnt = 0; CUDA_OK(cudaMemcpyFromSymbol(&cnt, d_dpcount, sizeof(uint32_t)));
        if (cnt > cap) cnt = cap;
        if (cnt == 0) continue;
        CUDA_OK(cudaMemcpy(hx.data(), d_dpx, (size_t)cnt * 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(hdist.data(), d_dpdist, (size_t)cnt * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(htype.data(), d_dptype, (size_t)cnt * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < cnt && !solved; i++) {
            std::string key((const char *)&hx[(size_t)i * 8], 32);
            uint64_t dval = hdist[i];
            if (htype[i] == 0) {  // tame: T
                auto it = wild_map.find(key);
                if (it != wild_map.end()) {
                    uint64_t Tt = dval, Dw = it->second;
                    if (Tt >= Dw) {
                        uint64_t dprime = Tt - Dw, cand = lo + dprime;
                        if (cand >= lo && cand <= hi) { found_cand = cand; solved = 1; break; }
                    }
                }
                if (tame_map.find(key) == tame_map.end()) tame_map[key] = dval;
            } else {              // wild: D
                auto it = tame_map.find(key);
                if (it != tame_map.end()) {
                    uint64_t Tt = it->second, Dw = dval;
                    if (Tt >= Dw) {
                        uint64_t dprime = Tt - Dw, cand = lo + dprime;
                        if (cand >= lo && cand <= hi) { found_cand = cand; solved = 1; break; }
                    }
                }
                if (wild_map.find(key) == wild_map.end()) wild_map[key] = dval;
            }
        }
    }

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    double jumps = (double)total_steps * (double)M;
    fprintf(stderr, "[kangaroo] rounds done: total_steps/kangaroo=%llu jumps=%.3e time=%.1fms jumps/s=%.3e\n",
            (unsigned long long)total_steps, jumps, ms, jumps / (ms / 1000.0));

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
