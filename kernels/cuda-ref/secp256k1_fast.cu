// secp256k1_fast.cu — Wave 3 fast-path scanner: affine sequential walk + batched Montgomery inversion.
//
// A/B PARTNER of the Wave-1 baseline (secp256k1_ref.cu), which is kept intact for honest comparison.
// Same per-key pipeline downstream (affine x,y -> compressed SEC -> sha256 -> ripemd160 -> hash160),
// but the EC stage is replaced: instead of a full 256-step k·G per key, each thread does ONE Jacobian
// scalar-mult to seed its start point, then walks a contiguous range with ONE affine add per key,
// amortizing the (single most expensive) modular inverse across a batch via Montgomery's trick.
//
// Modes:
//   ./secp256k1_fast verify [walkfile]   walk consecutive keys, assert each hash160 bit-exact vs oracle
//   ./secp256k1_fast bench  [keys]       throughput (median keys/s >=10 iters) -> results CSV
//   ./secp256k1_fast scan                seed target = vector[0].hash160; walk a window, assert FOUND
//
// The verify path deliberately walks from key 1, which drives the batched-inversion ZERO-denominator
// (doubling) path at key 2 — so the oracle comparison itself certifies the most bug-prone code.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

#include "secp256k1_field.cuh"
#include "secp256k1_ec.cuh"
#include "secp256k1_affine.cuh"
#include "hash.cuh"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1);} } while(0)

// Per-key action specialized at compile time so the inner loop has no mode branch.
enum WalkMode { MODE_EMIT = 0, MODE_BENCH = 1, MODE_SCAN = 2 };

// NEG_DEFAULT selects the negation-map (Opt C) variant at COMPILE time. When NEGMAP is undefined
// (the Opt A binary `secp256k1_fast`), NEG_DEFAULT==false and every `if (NEG)` below is dead-code-
// eliminated, so the Opt A code path stays byte-identical for honest A/B. When built with -DNEGMAP
// (the `secp256k1_negmap` binary), NEG_DEFAULT==true and each affine point hashes BOTH parities.
#ifdef NEGMAP
constexpr bool NEG_DEFAULT = true;
constexpr int  KEYS_PER_POINT = 2;   // Opt C: each affine point covers k AND n-k (2 addresses)
#else
constexpr bool NEG_DEFAULT = false;
constexpr int  KEYS_PER_POINT = 1;   // Opt A: each affine point covers exactly 1 key
#endif

// Hash an affine point's compressed pubkey -> hash160 (20B). Shared by every mode (Opt A path).
__device__ __forceinline__ void affine_hash160(uint8_t h160[20], const affpt *P) {
    uint8_t pub[33];
    compressed_pubkey(pub, &P->x, &P->y);
    hash160_33(h160, pub);
}

// NEGATION-MAP hash (Opt C): from ONE affine point P=(x,y) for key k, produce BOTH hash160s:
//   h_k    = hash160(prefix_k  || x)   prefix_k from parity of y           -> address of key k
//   h_negk = hash160(prefix_k^1|| x)   the flipped-parity encoding         -> address of key n-k
// Point -P = (x, p-y) has the SAME x and the OPPOSITE parity (p is odd), so the mirror key n-k's
// compressed pubkey is byte-identical except prefix ^= 0x01 (0x02<->0x03). The 32 x-bytes — hence
// the ENTIRE sha256 message except byte 0 — are shared; we still run 2 full hash160 (sha256 is
// sequential, the differing byte is in word 0, so the blocks cannot be shared — see negmap notes).
__device__ __forceinline__ void affine_hash160_negmap(uint8_t h_k[20], uint8_t h_negk[20],
                                                       const affpt *P) {
    uint8_t pub[33];
    compressed_pubkey(pub, &P->x, &P->y);   // pub[0] = parity of y (the real key k's encoding)
    hash160_33(h_k, pub);
    pub[0] ^= 0x01u;                         // flip 0x02<->0x03 : the mirror key n-k's encoding
    hash160_33(h_negk, pub);
}

// Per-point action for ONE hash160 (the body of every mode). Specialized at compile time so the
// inner loop carries no mode branch. `side` distinguishes a neg-map k-match (0) from an n-k match
// (1) for MODE_SCAN; ignored by EMIT/BENCH. For NEG=false this is invoked once per point with the
// exact same effect as the original Opt A inner block.
template <WalkMode MODE>
__device__ __forceinline__ void emit_or_match(const uint8_t h160[20], long out_idx,
                                              uint64_t keyVal, int side,
                                              uint8_t *out, uint8_t *sink, const uint8_t *target20,
                                              uint32_t *found_key, int *found_flag, int *found_side) {
    if (MODE == MODE_EMIT) {
#pragma unroll
        for (int b = 0; b < 20; b++) out[out_idx * 20 + b] = h160[b];
    } else if (MODE == MODE_BENCH) {
        uint8_t acc = 0;
#pragma unroll
        for (int b = 0; b < 20; b++) acc ^= h160[b];
        sink[out_idx & 1023] = acc;
    } else { // MODE_SCAN
        int m = 1;
#pragma unroll
        for (int b = 0; b < 20; b++) if (h160[b] != target20[b]) m = 0;
        if (m && atomicExch(found_flag, 1) == 0) {
            found_key[0] = (uint32_t)keyVal; found_key[1] = (uint32_t)(keyVal >> 32);
#pragma unroll
            for (int i = 2; i < 8; i++) found_key[i] = 0;  // small-key window
            if (found_side) *found_side = side;
        }
    }
}

// The single walk routine used by ALL modes (verify == bench == scan crypto, byte-identical).
// Thread covers keys [startKey .. startKey + steps*BATCH]: hashes the seed point, then each step
// advances by +BATCH·G via one batched inverse, hashing the BATCH new keys along the way.
//   MODE_EMIT : write each hash160 to out[(localKeyIndex)*20]   (verify; low volume)
//   MODE_BENCH: xor each hash160 into a tiny sink (throughput; keeps work live)
//   MODE_SCAN : memcmp each hash160 to target20; on match record key + set found flag
//
// NEG (Opt C, negation map): when true, each affine point P=(x,y) for key k yields TWO candidate
// keys — k (parity-correct 02/03||x) and n-k (the flipped-parity encoding, = point -P). We hash
// BOTH and process both. This HALVES the EC/walk work per address covered (one affine-add now
// covers 2 keys) but does NOT reduce hash work (still 2 hash160/point). Output/index layout for
// NEG: point at local index ki emits its k-hash at slot 2*ki and its n-k-hash at slot 2*ki+1.
// When NEG=false every `if (NEG)` branch is dead-code-eliminated, so the Opt A path is unchanged.
template <WalkMode MODE, bool NEG>
__device__ void walk_thread(const affpt *table_s, affpt base, const uint32_t startKey_lo[2],
                            int steps,
                            uint8_t *out,            // MODE_EMIT
                            uint8_t *sink,           // MODE_BENCH
                            const uint8_t *target20, // MODE_SCAN
                            uint32_t *found_key, int *found_flag, int *found_side,
                            long localKeyBase) {
    uint8_t h160[20], h160b[20];

    // key counter (only the low 64 bits move within a thread's range; high limbs static here)
    uint64_t key64 = ((uint64_t)startKey_lo[1] << 32) | startKey_lo[0];

    // hash the seed (key = startKey)
    {
        long ki = localKeyBase;  // index 0 = seed
        if (NEG) {
            affine_hash160_negmap(h160, h160b, &base);
            // for an n-k match the recorded low64 is the seed's k (host computes n-k); side=1
            emit_or_match<MODE>(h160,  2 * ki + 0, key64, 0, out, sink, target20, found_key, found_flag, found_side);
            emit_or_match<MODE>(h160b, 2 * ki + 1, key64, 1, out, sink, target20, found_key, found_flag, found_side);
        } else {
            affine_hash160(h160, &base);
            emit_or_match<MODE>(h160, ki, key64, 0, out, sink, target20, found_key, found_flag, found_side);
        }
    }

    // ---- the batched walk ----
    fe prod[BATCH + 1];
    uint8_t zero[BATCH];
    for (int s = 0; s < steps; s++) {
        // forward pass: prefix products of nonzero denominators d_i = table[i+1].x - base.x
        fe_set_zero(&prod[0]); prod[0].v[0] = 1u;
#pragma unroll 1
        for (int i = 0; i < BATCH; i++) {
            fe d; fe_sub(&d, &table_s[i + 1].x, &base.x);
            if (fe_is_zero(&d)) { zero[i] = 1u; fe_copy(&prod[i + 1], &prod[i]); }
            else                { zero[i] = 0u; fe_mul(&prod[i + 1], &prod[i], &d); }
        }
        fe u; fe_inv(&u, &prod[BATCH]);   // ONE inverse for the whole batch (never inv(0))

        affpt advance;  // will hold key = startKey + (s+1)*BATCH  (= base + BATCH·G)
        // backward pass: fuse inverse-extract + affine add + hash160, no results[] array
#pragma unroll 1
        for (int i = BATCH - 1; i >= 0; i--) {
            affpt R;
            if (zero[i]) {
                aff_add_fallback(&R, &base, &table_s[i + 1]);  // doubling / infinity (rare)
            } else {
                fe inv_i; fe_mul(&inv_i, &u, &prod[i]);        // 1/d_i
                aff_add_with_inv(&R, &base, &table_s[i + 1], &inv_i);
                fe d; fe_sub(&d, &table_s[i + 1].x, &base.x);
                fe_mul(&u, &u, &d);                            // strip this factor from u
            }
            if (i == BATCH - 1) advance = R;

            // R is key = startKey + s*BATCH + (i+1)
            long ki = localKeyBase + (long)s * BATCH + (i + 1);
            uint64_t thisKey = key64 + (uint64_t)s * BATCH + (uint64_t)(i + 1);
            if (NEG) {
                affine_hash160_negmap(h160, h160b, &R);
                emit_or_match<MODE>(h160,  2 * ki + 0, thisKey, 0, out, sink, target20, found_key, found_flag, found_side);
                emit_or_match<MODE>(h160b, 2 * ki + 1, thisKey, 1, out, sink, target20, found_key, found_flag, found_side);
            } else {
                affine_hash160(h160, &R);
                emit_or_match<MODE>(h160, ki, thisKey, 0, out, sink, target20, found_key, found_flag, found_side);
            }
        }
        base = advance;  // next batch starts at key += BATCH
    }
}

// Load the global multiples table into shared memory once per block (read N times/step/thread).
__device__ __forceinline__ void load_table_shared(affpt *table_s, const affpt *table_g) {
    for (int i = threadIdx.x; i <= BATCH; i += blockDim.x) table_s[i] = table_g[i];
    __syncthreads();
}

// ---- SEED-HASH kernel: hash N independent seed points through the fast-path affine pipeline ----
// Proves the fast kernel reproduces the canonical (scattered) vectors' hash160 bit-exact. Exercises
// the SAME affine_hash160 the walk uses; seeds come from the verified Jacobian scalar_mul_G.
__global__ void kfast_seedhash(const affpt *seeds, uint8_t *out, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint8_t h160[20];
    affine_hash160(h160, &seeds[idx]);
#pragma unroll
    for (int b = 0; b < 20; b++) out[idx * 20 + b] = h160[b];
}

// ---- VERIFY kernel: single thread walks `keys` consecutive keys from `base0`, emits hash160s ----
// In the NEGMAP build (NEG_DEFAULT==true) each point emits TWO hash160 (k at slot 2*ki, n-k at 2*ki+1).
__global__ void kfast_verify(const affpt *table_g, affpt base0,
                             uint32_t start_lo0, uint32_t start_lo1,
                             int steps, uint8_t *out) {
    extern __shared__ affpt table_s[];
    load_table_shared(table_s, table_g);
    if (threadIdx.x || blockIdx.x) return;
    uint32_t lo[2] = {start_lo0, start_lo1};
    walk_thread<MODE_EMIT, NEG_DEFAULT>(table_s, base0, lo, steps, out, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, 0);
}

// ---- BENCH kernel: each thread walks `steps` batches from its own contiguous start ----
__global__ void kfast_bench(const affpt *table_g, const affpt *bases,
                            int steps, uint8_t *sink) {
    extern __shared__ affpt table_s[];
    load_table_shared(table_s, table_g);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    affpt base = bases[tid];
    uint32_t lo[2] = {0u, 0u};  // key value irrelevant for bench
    walk_thread<MODE_BENCH, NEG_DEFAULT>(table_s, base, lo, steps, nullptr, sink, nullptr,
                                         nullptr, nullptr, nullptr, 0);
}

// ---- SCAN kernel: each thread walks its window, compares to target ----
// found_side: 0 if the target matched the key-k address, 1 if it matched the n-k (mirror) address.
__global__ void kfast_scan(const affpt *table_g, const affpt *bases,
                           const uint32_t *base_keys_lo,  // 2 limbs per thread
                           int steps, const uint8_t *target20,
                           uint32_t *found_key, int *found_flag, int *found_side) {
    extern __shared__ affpt table_s[];
    load_table_shared(table_s, table_g);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    affpt base = bases[tid];
    uint32_t lo[2] = {base_keys_lo[tid * 2 + 0], base_keys_lo[tid * 2 + 1]};
    walk_thread<MODE_SCAN, NEG_DEFAULT>(table_s, base, lo, steps, nullptr, nullptr, target20,
                                        found_key, found_flag, found_side, 0);
}

// ======================= Host: scalar-mult seed via the VERIFIED Jacobian path =======================
// We compute start points (startKey·G, affine) on the GPU using the Wave-1 scalar_mul_G, then hand
// the affine seed(s) to the walk. This reuses verified code for the one-time per-thread init.
__global__ void kseed_affine(const uint32_t *keys, affpt *out, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint32_t k[8];
#pragma unroll
    for (int i = 0; i < 8; i++) k[i] = keys[idx * 8 + i];
    jpoint P; scalar_mul_G(&P, k);
    fe x, y; jpoint_to_affine(&x, &y, &P);
    fe_copy(&out[idx].x, &x); fe_copy(&out[idx].y, &y);
}

// ======================= Host helpers (shared shape with secp256k1_ref.cu) =======================
struct Vector { std::string priv_hex, pub_hex, h160_hex; };

static std::string read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::string s; s.resize(n);
    if (fread(&s[0], 1, n, f) != (size_t)n) { fprintf(stderr, "read error\n"); exit(1); }
    fclose(f); return s;
}
static std::vector<std::string> extract_values(const std::string &json, const std::string &key) {
    std::vector<std::string> out; std::string needle = "\"" + key + "\""; size_t pos = 0;
    while ((pos = json.find(needle, pos)) != std::string::npos) {
        size_t colon = json.find(':', pos + needle.size()); if (colon == std::string::npos) break;
        size_t q1 = json.find('"', colon); if (q1 == std::string::npos) break;
        size_t q2 = json.find('"', q1 + 1); if (q2 == std::string::npos) break;
        out.push_back(json.substr(q1 + 1, q2 - q1 - 1)); pos = q2 + 1;
    }
    return out;
}
static std::vector<Vector> load_vectors(const char *path) {
    std::string json = read_file(path);
    auto privs = extract_values(json, "priv_hex");
    auto pubs  = extract_values(json, "pub_compressed_hex");
    auto h160s = extract_values(json, "hash160_hex");
    std::vector<Vector> v; size_t n = privs.size();
    for (size_t i = 0; i < n; i++) {
        Vector vec; vec.priv_hex = privs[i];
        vec.pub_hex = (i < pubs.size()) ? pubs[i] : "";
        vec.h160_hex = (i < h160s.size()) ? h160s[i] : "";
        v.push_back(vec);
    }
    return v;
}
static void hex_to_limbs(const std::string &hex, uint32_t out[8]) {
    std::string h = hex; while (h.size() < 64) h = "0" + h;
    for (int i = 0; i < 8; i++) {
        std::string grp = h.substr((7 - i) * 8, 8);
        out[i] = (uint32_t)strtoul(grp.c_str(), nullptr, 16);
    }
}
static void hex20_to_bytes(const std::string &hex, uint8_t out[20]) {
    for (int i = 0; i < 20; i++) out[i] = (uint8_t)strtoul(hex.substr(i*2,2).c_str(), nullptr, 16);
}
static std::string bytes_to_hex(const uint8_t *b, int n) {
    static const char *hexd = "0123456789abcdef"; std::string s; s.resize(n*2);
    for (int i = 0; i < n; i++) { s[i*2]=hexd[b[i]>>4]; s[i*2+1]=hexd[b[i]&0xf]; }
    return s;
}

// Compute affine seeds for an array of 8-limb keys on the GPU (verified Jacobian path).
static void seed_affine(const std::vector<uint32_t> &keys_limbs, int count, std::vector<affpt> &out) {
    uint32_t *d_keys; affpt *d_out;
    CUDA_CHECK(cudaMalloc(&d_keys, count * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, count * sizeof(affpt)));
    CUDA_CHECK(cudaMemcpy(d_keys, keys_limbs.data(), count * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    int th = 64, bl = (count + th - 1) / th;
    kseed_affine<<<bl, th>>>(d_keys, d_out, count);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    out.resize(count);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, count * sizeof(affpt), cudaMemcpyDeviceToHost));
    cudaFree(d_keys); cudaFree(d_out);
}

static affpt *build_table_device() {
    affpt *d_table;
    CUDA_CHECK(cudaMalloc(&d_table, (BATCH + 1) * sizeof(affpt)));
    kfast_build_table<<<1, 1>>>(d_table);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    return d_table;
}

// ======================= Modes =======================

// Are the vectors a consecutive run priv[i] == priv[0]+i (low-limb only, no high-limb carry)?
// Canonical test_vectors.json is scattered -> false; oracle `walk` output -> true.
static bool vectors_consecutive(const std::vector<Vector> &vecs) {
    if (vecs.size() < 2) return false;
    uint32_t k0[8]; hex_to_limbs(vecs[0].priv_hex, k0);
    for (size_t i = 1; i < vecs.size(); i++) {
        uint32_t ki[8]; hex_to_limbs(vecs[i].priv_hex, ki);
        // expected = k0 + i  (add to low limbs, propagate carry)
        uint32_t e[8]; for (int j = 0; j < 8; j++) e[j] = k0[j];
        uint64_t add = (uint64_t)i, c = 0;
        uint64_t s = (uint64_t)e[0] + (uint32_t)(add & 0xffffffffu); e[0]=(uint32_t)s; c=s>>32;
        s = (uint64_t)e[1] + (uint32_t)(add>>32) + c; e[1]=(uint32_t)s; c=s>>32;
        for (int j = 2; j < 8; j++) { s=(uint64_t)e[j]+c; e[j]=(uint32_t)s; c=s>>32; }
        for (int j = 0; j < 8; j++) if (e[j] != ki[j]) return false;
    }
    return true;
}

// Scattered-vector path: seed each key independently (verified Jacobian) and hash through the SAME
// fast-path affine pipeline. Proves the canonical 8 vectors stay bit-exact under the fast kernel.
static int mode_verify_scattered(const std::vector<Vector> &vecs) {
    int n = (int)vecs.size();
    std::vector<uint32_t> keys(n * 8);
    for (int i = 0; i < n; i++) hex_to_limbs(vecs[i].priv_hex, &keys[i * 8]);
    std::vector<affpt> seeds; seed_affine(keys, n, seeds);

    affpt *d_seeds; uint8_t *d_out;
    CUDA_CHECK(cudaMalloc(&d_seeds, n * sizeof(affpt)));
    CUDA_CHECK(cudaMalloc(&d_out, n * 20));
    CUDA_CHECK(cudaMemcpy(d_seeds, seeds.data(), n * sizeof(affpt), cudaMemcpyHostToDevice));
    int th = 64, bl = (n + th - 1) / th;
    kfast_seedhash<<<bl, th>>>(d_seeds, d_out, n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> h_out(n * 20);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * 20, cudaMemcpyDeviceToHost));
    cudaFree(d_seeds); cudaFree(d_out);

    int ok = 1;
    for (int i = 0; i < n; i++) {
        std::string got = bytes_to_hex(&h_out[i * 20], 20);
        if (got != vecs[i].h160_hex) {
            ok = 0;
            printf("  vector[%d] FAIL got=%s expected=%s (priv=%s)\n",
                   i, got.c_str(), vecs[i].h160_hex.c_str(), vecs[i].priv_hex.c_str());
        } else {
            printf("  vector[%d] PASS hash160=%s\n", i, got.c_str());
        }
    }
    if (ok) printf("ALL %d VECTORS PASS\n", n);
    else    printf("SOME VECTORS FAILED\n");
    return ok ? 0 : 1;
}

static int mode_verify(const std::vector<Vector> &vecs) {
    int nkeys = (int)vecs.size();
    if (nkeys < 1) { fprintf(stderr, "no vectors\n"); return 1; }

    // Scattered (canonical) vectors -> seed-and-hash each; consecutive -> exercise the +G walk.
    if (!vectors_consecutive(vecs)) {
        printf("(scattered vectors: seed-and-hash each through the fast affine pipeline)\n");
        return mode_verify_scattered(vecs);
    }
    printf("(consecutive vectors: exercising the +G affine walk + batched inversion)\n");

    uint32_t start[8]; hex_to_limbs(vecs[0].priv_hex, start);
    std::vector<uint32_t> seedk(start, start + 8);
    std::vector<affpt> seed; seed_affine(seedk, 1, seed);

    // steps so that 1 + steps*BATCH >= nkeys  (we then compare the first nkeys outputs)
    int steps = (nkeys - 1 + BATCH - 1) / BATCH;
    long cap = 1 + (long)steps * BATCH;
    affpt *d_table = build_table_device();
    uint8_t *d_out; CUDA_CHECK(cudaMalloc(&d_out, cap * 20));
    size_t shmem = (BATCH + 1) * sizeof(affpt);
    kfast_verify<<<1, 64, shmem>>>(d_table, seed[0], start[0], start[1], steps, d_out);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> h_out(cap * 20);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, cap * 20, cudaMemcpyDeviceToHost));
    cudaFree(d_out); cudaFree(d_table);

    int ok = 1;
    for (int i = 0; i < nkeys; i++) {
        std::string got = bytes_to_hex(&h_out[(long)i * 20], 20);
        if (got != vecs[i].h160_hex) {
            ok = 0;
            printf("  walk[%d] FAIL got=%s expected=%s (priv=%s)\n",
                   i, got.c_str(), vecs[i].h160_hex.c_str(), vecs[i].priv_hex.c_str());
            if (i > 6) break;  // don't spam
        } else if (i < 3 || i == nkeys - 1) {
            printf("  walk[%d] PASS hash160=%s\n", i, got.c_str());
        }
    }
    if (ok) printf("ALL %d WALK VECTORS PASS\n", nkeys);
    else    printf("SOME WALK VECTORS FAILED\n");
    return ok ? 0 : 1;
}

// ---- NEG-MAP verify: walk consecutive keys from START; assert BOTH parity hashes bit-exact ----
// Reads a negmap oracle file (kat_oracle.py negmap START COUNT) with fields k_hex/h160_k_hex/
// h160_negk_hex. The NEGMAP kfast_verify emits 2 slots per point: 2*i = key-k hash, 2*i+1 = n-k
// hash. We compare both against the oracle, so this certifies the parity-byte flip AND the n-k
// address derivation in one shot. Only meaningful in the -DNEGMAP binary (KEYS_PER_POINT==2).
static int mode_negverify(const char *path) {
    if (KEYS_PER_POINT != 2) {
        printf("negverify requires the -DNEGMAP build (this binary covers 1 key/point)\n");
        return 2;
    }
    std::string json = read_file(path);
    auto ks    = extract_values(json, "k_hex");
    auto hk    = extract_values(json, "h160_k_hex");
    auto hnk   = extract_values(json, "h160_negk_hex");
    int nkeys = (int)ks.size();
    if (nkeys < 1 || hk.size() != (size_t)nkeys || hnk.size() != (size_t)nkeys) {
        fprintf(stderr, "bad negmap file %s\n", path); return 1;
    }
    printf("(neg-map verify: %d consecutive keys from k=%s, hashing BOTH parities/point)\n",
           nkeys, ks[0].c_str());

    uint32_t start[8]; hex_to_limbs(ks[0], start);
    std::vector<uint32_t> seedk(start, start + 8);
    std::vector<affpt> seed; seed_affine(seedk, 1, seed);

    int steps = (nkeys - 1 + BATCH - 1) / BATCH;
    long pts = 1 + (long)steps * BATCH;
    long cap = pts * 2;  // 2 hash160 slots per affine point in the NEGMAP build
    affpt *d_table = build_table_device();
    uint8_t *d_out; CUDA_CHECK(cudaMalloc(&d_out, cap * 20));
    size_t shmem = (BATCH + 1) * sizeof(affpt);
    kfast_verify<<<1, 64, shmem>>>(d_table, seed[0], start[0], start[1], steps, d_out);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> h_out(cap * 20);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, cap * 20, cudaMemcpyDeviceToHost));
    cudaFree(d_out); cudaFree(d_table);

    int ok = 1;
    for (int i = 0; i < nkeys; i++) {
        std::string gk  = bytes_to_hex(&h_out[(long)(2 * i + 0) * 20], 20);  // key-k hash
        std::string gnk = bytes_to_hex(&h_out[(long)(2 * i + 1) * 20], 20);  // n-k mirror hash
        bool pk = (gk == hk[i]), pnk = (gnk == hnk[i]);
        if (!pk || !pnk) {
            ok = 0;
            printf("  neg[%d] FAIL  k:%s(%s)  n-k:%s(%s)\n", i,
                   gk.c_str(),  pk  ? "ok" : ("!=" + hk[i]).c_str(),
                   gnk.c_str(), pnk ? "ok" : ("!=" + hnk[i]).c_str());
            if (i > 6) break;
        } else if (i < 3 || i == nkeys - 1) {
            printf("  neg[%d] PASS  k-hash=%s  n-k-hash=%s\n", i, gk.c_str(), gnk.c_str());
        }
    }
    if (ok) printf("ALL %d NEG-MAP PAIRS PASS (both parities bit-exact)\n", nkeys);
    else    printf("SOME NEG-MAP PAIRS FAILED\n");
    return ok ? 0 : 1;
}

static int mode_scan(const std::vector<Vector> &vecs) {
    // target = vecs[0].hash160; one thread walks [priv-? .. ] covering priv. Use startKey = priv-? .
    // We seed a single thread at base = (priv - 5)·G and walk a small window so priv is inside.
    uint32_t priv[8]; hex_to_limbs(vecs[0].priv_hex, priv);
    const uint64_t BACK = 5;
    uint32_t startk[8]; for (int i = 0; i < 8; i++) startk[i] = priv[i];
    { int64_t borrow = BACK; for (int i = 0; i < 8 && borrow; i++) {
        int64_t t = (int64_t)startk[i] - (borrow & 0xffffffff); int64_t hib = borrow >> 32;
        if (t < 0) { t += (1ll<<32); hib += 1; } startk[i] = (uint32_t)t; borrow = hib; } }

    uint8_t target20[20]; hex20_to_bytes(vecs[0].h160_hex, target20);
    std::vector<uint32_t> seedk(startk, startk + 8);
    std::vector<affpt> seed; seed_affine(seedk, 1, seed);

    int steps = 2;  // 1 + 2*BATCH keys covers priv (BACK=5 < BATCH)
    affpt *d_table = build_table_device();
    affpt *d_bases; CUDA_CHECK(cudaMalloc(&d_bases, sizeof(affpt)));
    CUDA_CHECK(cudaMemcpy(d_bases, &seed[0], sizeof(affpt), cudaMemcpyHostToDevice));
    uint32_t lo[2] = {startk[0], startk[1]};
    uint32_t *d_lo; CUDA_CHECK(cudaMalloc(&d_lo, 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_lo, lo, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    uint8_t *d_target; CUDA_CHECK(cudaMalloc(&d_target, 20));
    CUDA_CHECK(cudaMemcpy(d_target, target20, 20, cudaMemcpyHostToDevice));
    uint32_t *d_fk; int *d_ff; int *d_fs;
    CUDA_CHECK(cudaMalloc(&d_fk, 8 * sizeof(uint32_t))); CUDA_CHECK(cudaMalloc(&d_ff, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fs, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ff, 0, sizeof(int))); CUDA_CHECK(cudaMemset(d_fk, 0, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_fs, 0, sizeof(int)));

    size_t shmem = (BATCH + 1) * sizeof(affpt);
    kfast_scan<<<1, 64, shmem>>>(d_table, d_bases, d_lo, steps, d_target, d_fk, d_ff, d_fs);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    int ff = 0, fs = 0; uint32_t fk[8];
    CUDA_CHECK(cudaMemcpy(&ff, d_ff, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&fs, d_fs, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fk, d_fk, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_table); cudaFree(d_bases); cudaFree(d_lo); cudaFree(d_target);
    cudaFree(d_fk); cudaFree(d_ff); cudaFree(d_fs);

    uint64_t want = ((uint64_t)priv[1] << 32) | priv[0];
    uint64_t got = ((uint64_t)fk[1] << 32) | fk[0];
    // side==0: matched the key-k address (the recorded low64 IS the key). side==1: matched the n-k
    // (mirror) address; the recorded low64 is k, and the actual found key is n-k (host knows N).
    if (ff && got == want) {
        printf("SCAN: FOUND key_lo=%016llx matches vector[0].priv low64 EXACTLY (side=%s)\n",
               (unsigned long long)got, fs == 0 ? "k" : "n-k");
        return 0;
    }
    printf("SCAN: %s (got_lo=%016llx want_lo=%016llx side=%d)\n",
           ff ? "FOUND-BUT-MISMATCH" : "NOT FOUND",
           (unsigned long long)got, (unsigned long long)want, fs);
    return 1;
}

static int mode_bench(const std::vector<Vector> &vecs, uint64_t target_keys, int iters) {
    // Choose a thread grid; each thread walks `steps` batches => keys/thread = 1 + steps*BATCH.
    int threads = 256;
    int blocks = 1024;
    int nthreads = threads * blocks;
    // steps to reach ~target_keys total
    long perThread = (long)((target_keys + nthreads - 1) / nthreads);
    int steps = (int)((perThread - 1 + BATCH - 1) / BATCH); if (steps < 1) steps = 1;
    uint64_t pointsPerThread = 1ull + (uint64_t)steps * BATCH;       // affine points (EC adds) / thread
    uint64_t totalPoints = pointsPerThread * (uint64_t)nthreads;
    // HONEST A/B metric: addresses (candidate private keys) COVERED per second. Opt A covers 1 per
    // affine point; Opt C (neg-map) covers KEYS_PER_POINT=2 (k and n-k) per point. So the throughput
    // comparison is addresses/s = points/s * KEYS_PER_POINT — this is what makes a neg-map win (or
    // its absence) visible: if hashing fully bottlenecks, points/s halves and addresses/s is flat.
    uint64_t totalKeys = totalPoints * (uint64_t)KEYS_PER_POINT;

    // seed each thread at a distinct contiguous start (= base + tid*pointsPerThread). Use vecs[0].priv
    // as the global base; exact key values are irrelevant for throughput, only the work is.
    uint32_t base[8]; hex_to_limbs(vecs[0].priv_hex, base);
    std::vector<uint32_t> keys(nthreads * 8);
    for (int t = 0; t < nthreads; t++) {
        uint32_t k[8]; for (int i = 0; i < 8; i++) k[i] = base[i];
        uint64_t add = (uint64_t)t * pointsPerThread;
        uint64_t s = (uint64_t)k[0] + (uint32_t)(add & 0xffffffffu); k[0]=(uint32_t)s; uint64_t c=s>>32;
        s = (uint64_t)k[1] + (uint32_t)(add>>32) + c; k[1]=(uint32_t)s; c=s>>32;
        for (int i = 2; i < 8; i++) { s=(uint64_t)k[i]+c; k[i]=(uint32_t)s; c=s>>32; }
        for (int i = 0; i < 8; i++) keys[t*8+i] = k[i];
    }
    std::vector<affpt> seeds; seed_affine(keys, nthreads, seeds);

    affpt *d_table = build_table_device();
    affpt *d_bases; CUDA_CHECK(cudaMalloc(&d_bases, nthreads * sizeof(affpt)));
    CUDA_CHECK(cudaMemcpy(d_bases, seeds.data(), nthreads * sizeof(affpt), cudaMemcpyHostToDevice));
    uint8_t *d_sink; CUDA_CHECK(cudaMalloc(&d_sink, 1024));
    size_t shmem = (BATCH + 1) * sizeof(affpt);

    // warm-up
    kfast_bench<<<blocks, threads, shmem>>>(d_table, d_bases, steps, d_sink);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> kps;
    for (int it = 0; it < iters; it++) {
        cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        kfast_bench<<<blocks, threads, shmem>>>(d_table, d_bases, steps, d_sink);
        CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        double rate = (double)totalKeys / (ms / 1000.0);
        kps.push_back(rate);
        printf("  bench iter %2d: %.3f ms -> %.0f keys/s\n", it, ms, rate);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    cudaFree(d_table); cudaFree(d_bases); cudaFree(d_sink);

    std::sort(kps.begin(), kps.end());
    double median = kps[kps.size()/2];
    if (kps.size()%2==0) median = 0.5*(kps[kps.size()/2-1]+kps[kps.size()/2]);
    double pts_per_s = median / (double)KEYS_PER_POINT;
    printf("FAST BENCH: median %.0f keys/s over %d iters (BATCH=%d, steps=%d, %llu keys/iter, "
           "KEYS_PER_POINT=%d -> %.0f affine-points/s)\n",
           median, (int)kps.size(), BATCH, steps, (unsigned long long)totalKeys,
           KEYS_PER_POINT, pts_per_s);
    return 0;
}

int main(int argc, char **argv) {
    const char *vec_path = "kernels/cuda-ref/test_vectors.json";
    std::string mode = (argc > 1) ? argv[1] : "verify";
    if (const char *p = getenv("VEC_PATH")) vec_path = p;
    // verify may take an explicit walk file as argv[2]
    if (mode == "verify" && argc > 2) vec_path = argv[2];

    // negverify reads a negmap-format oracle file directly (k_hex / h160_k_hex / h160_negk_hex),
    // not the standard priv/pub/h160 Vector schema, so handle it before the generic loader.
    if (mode == "negverify") {
        const char *np = (argc > 2) ? argv[2] : vec_path;
        printf("BATCH=%d KEYS_PER_POINT=%d  negmap file=%s\n", BATCH, KEYS_PER_POINT, np);
        return mode_negverify(np);
    }

    std::vector<Vector> vecs = load_vectors(vec_path);
    if (vecs.empty()) { fprintf(stderr, "no vectors loaded from %s\n", vec_path); return 1; }
    printf("Loaded %zu vectors from %s (BATCH=%d, KEYS_PER_POINT=%d)\n",
           vecs.size(), vec_path, BATCH, KEYS_PER_POINT);

    int rc = 0;
    if (mode == "verify")      rc = mode_verify(vecs);
    else if (mode == "scan")   rc = mode_scan(vecs);
    else if (mode == "bench") {
        uint64_t target = (argc > 2) ? strtoull(argv[2], nullptr, 10) : (1ull << 24);
        rc = mode_bench(vecs, target, 12);
    } else { fprintf(stderr, "unknown mode %s\n", mode.c_str()); rc = 2; }
    return rc;
}
