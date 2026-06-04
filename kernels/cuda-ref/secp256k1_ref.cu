// secp256k1_ref.cu — Wave 1 single-GPU brute-force secp256k1 scanner + correctness harness.
//
// Pipeline per private key k (matches scripts/kat_oracle.py BIT-EXACT):
//   P = k*G (Jacobian double-and-add) -> affine (x,y) -> compressed SEC pubkey (33B)
//   -> sha256 -> ripemd160 (hash160, 20B) -> memcmp vs target.
//
// Modes:
//   ./secp256k1_ref verify            run all test vectors, assert hash160 bit-exact -> "ALL N VECTORS PASS"
//   ./secp256k1_ref scan              scan a small range containing vector[0].priv for vector[0].hash160
//   ./secp256k1_ref bench [n]         throughput benchmark (median keys/s over >=10 iters) -> CSV
//   ./secp256k1_ref all               verify + scan + bench (default)
//
// No external deps; tiny hand-rolled JSON parser for test_vectors.json.

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
#include "hash.cuh"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1);} } while(0)

// ======================= Device kernels =======================

// Compute hash160 for an array of private keys (each 8 little-endian uint32 limbs).
__global__ void kernel_hash160_batch(const uint32_t *privs, uint8_t *out_h160, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint32_t k[8];
#pragma unroll
    for (int i = 0; i < 8; i++) k[i] = privs[idx * 8 + i];

    jpoint P;
    scalar_mul_G(&P, k);
    fe x, y;
    jpoint_to_affine(&x, &y, &P);
    uint8_t pub[33];
    compressed_pubkey(pub, &x, &y);
    uint8_t h160[20];
    hash160_33(h160, pub);
#pragma unroll
    for (int i = 0; i < 20; i++) out_h160[idx * 20 + i] = h160[i];
}

// Scan kernel: for keys base..base+count-1 (base is 8 limbs), compute hash160 and compare to target.
// On match, write the matching key (8 limbs) to found_key and set found_flag=1.
__global__ void kernel_scan(const uint32_t *base, uint64_t count,
                            const uint8_t *target20,
                            uint32_t *found_key, int *found_flag) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    // k = base + idx  (add idx as a 64-bit increment to the low limbs, propagate carry)
    uint32_t k[8];
#pragma unroll
    for (int i = 0; i < 8; i++) k[i] = base[i];
    uint64_t add = idx;
    uint64_t s = (uint64_t)k[0] + (uint32_t)(add & 0xffffffffu);
    k[0] = (uint32_t)s;
    uint64_t carry = s >> 32;
    s = (uint64_t)k[1] + (uint32_t)(add >> 32) + carry;
    k[1] = (uint32_t)s;
    carry = s >> 32;
#pragma unroll
    for (int i = 2; i < 8; i++) {
        s = (uint64_t)k[i] + carry;
        k[i] = (uint32_t)s;
        carry = s >> 32;
    }

    jpoint P;
    scalar_mul_G(&P, k);
    fe x, y;
    jpoint_to_affine(&x, &y, &P);
    uint8_t pub[33];
    compressed_pubkey(pub, &x, &y);
    uint8_t h160[20];
    hash160_33(h160, pub);

    int match = 1;
#pragma unroll
    for (int i = 0; i < 20; i++) if (h160[i] != target20[i]) { match = 0; }
    if (match) {
        if (atomicExch(found_flag, 1) == 0) {
#pragma unroll
            for (int i = 0; i < 8; i++) found_key[i] = k[i];
        }
    }
}

// Bench kernel: same full pipeline, no early-out, writes a tiny xor checksum so the
// compiler cannot eliminate the work. Each thread does `per_thread` keys starting at base+tid*per_thread.
__global__ void kernel_bench(const uint32_t *base, uint64_t total, uint8_t *sink) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    uint32_t k[8];
#pragma unroll
    for (int i = 0; i < 8; i++) k[i] = base[i];
    uint64_t add = idx;
    uint64_t s = (uint64_t)k[0] + (uint32_t)(add & 0xffffffffu);
    k[0] = (uint32_t)s; uint64_t carry = s >> 32;
    s = (uint64_t)k[1] + (uint32_t)(add >> 32) + carry; k[1] = (uint32_t)s; carry = s >> 32;
#pragma unroll
    for (int i = 2; i < 8; i++) { s = (uint64_t)k[i] + carry; k[i] = (uint32_t)s; carry = s >> 32; }

    jpoint P;
    scalar_mul_G(&P, k);
    fe x, y;
    jpoint_to_affine(&x, &y, &P);
    uint8_t pub[33];
    compressed_pubkey(pub, &x, &y);
    uint8_t h160[20];
    hash160_33(h160, pub);
    uint8_t acc = 0;
#pragma unroll
    for (int i = 0; i < 20; i++) acc ^= h160[i];
    sink[idx & 1023] = acc;  // scatter into a small buffer to keep results live
}

// ======================= Host helpers =======================

struct Vector {
    std::string priv_hex, pub_hex, h160_hex;
};

static std::string read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::string s; s.resize(n);
    if (fread(&s[0], 1, n, f) != (size_t)n) { fprintf(stderr, "read error\n"); exit(1); }
    fclose(f);
    return s;
}

// Extract all string values for a given key name in order of appearance.
static std::vector<std::string> extract_values(const std::string &json, const std::string &key) {
    std::vector<std::string> out;
    std::string needle = "\"" + key + "\"";
    size_t pos = 0;
    while ((pos = json.find(needle, pos)) != std::string::npos) {
        size_t colon = json.find(':', pos + needle.size());
        if (colon == std::string::npos) break;
        size_t q1 = json.find('"', colon);
        if (q1 == std::string::npos) break;
        size_t q2 = json.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        out.push_back(json.substr(q1 + 1, q2 - q1 - 1));
        pos = q2 + 1;
    }
    return out;
}

static std::vector<Vector> load_vectors(const char *path) {
    std::string json = read_file(path);
    auto privs = extract_values(json, "priv_hex");
    auto pubs  = extract_values(json, "pub_compressed_hex");
    auto h160s = extract_values(json, "hash160_hex");
    std::vector<Vector> v;
    size_t n = privs.size();
    for (size_t i = 0; i < n; i++) {
        Vector vec;
        vec.priv_hex = privs[i];
        vec.pub_hex  = (i < pubs.size()) ? pubs[i] : "";
        vec.h160_hex = (i < h160s.size()) ? h160s[i] : "";
        v.push_back(vec);
    }
    return v;
}

// hex string (64 chars, big-endian) -> 8 little-endian uint32 limbs.
static void hex_to_limbs(const std::string &hex, uint32_t out[8]) {
    // pad to 64 chars
    std::string h = hex;
    while (h.size() < 64) h = "0" + h;
    for (int i = 0; i < 8; i++) {
        // limb i (little-endian) corresponds to the (7-i)-th 8-hex-digit group from the left
        std::string grp = h.substr((7 - i) * 8, 8);
        out[i] = (uint32_t)strtoul(grp.c_str(), nullptr, 16);
    }
}

static void hex20_to_bytes(const std::string &hex, uint8_t out[20]) {
    for (int i = 0; i < 20; i++) {
        std::string b = hex.substr(i * 2, 2);
        out[i] = (uint8_t)strtoul(b.c_str(), nullptr, 16);
    }
}

static std::string bytes_to_hex(const uint8_t *b, int n) {
    static const char *hexd = "0123456789abcdef";
    std::string s; s.resize(n * 2);
    for (int i = 0; i < n; i++) { s[i*2] = hexd[b[i] >> 4]; s[i*2+1] = hexd[b[i] & 0xf]; }
    return s;
}

static std::string limbs_to_hex(const uint32_t k[8]) {
    uint8_t b[32];
    for (int i = 0; i < 8; i++) {
        uint32_t limb = k[7 - i];
        b[i*4+0] = (uint8_t)(limb >> 24);
        b[i*4+1] = (uint8_t)(limb >> 16);
        b[i*4+2] = (uint8_t)(limb >> 8);
        b[i*4+3] = (uint8_t)(limb);
    }
    return bytes_to_hex(b, 32);
}

// ======================= Modes =======================

static int mode_verify(const std::vector<Vector> &vecs) {
    int n = (int)vecs.size();
    std::vector<uint32_t> h_privs(n * 8);
    for (int i = 0; i < n; i++) hex_to_limbs(vecs[i].priv_hex, &h_privs[i * 8]);

    uint32_t *d_privs; uint8_t *d_h160;
    CUDA_CHECK(cudaMalloc(&d_privs, n * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_h160, n * 20));
    CUDA_CHECK(cudaMemcpy(d_privs, h_privs.data(), n * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    int threads = 64;
    int blocks = (n + threads - 1) / threads;
    kernel_hash160_batch<<<blocks, threads>>>(d_privs, d_h160, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> h_h160(n * 20);
    CUDA_CHECK(cudaMemcpy(h_h160.data(), d_h160, n * 20, cudaMemcpyDeviceToHost));

    int ok = 1;
    for (int i = 0; i < n; i++) {
        std::string got = bytes_to_hex(&h_h160[i * 20], 20);
        if (got != vecs[i].h160_hex) {
            ok = 0;
            printf("  vector[%d] FAIL  got=%s expected=%s (priv=%s)\n",
                   i, got.c_str(), vecs[i].h160_hex.c_str(), vecs[i].priv_hex.c_str());
        } else {
            printf("  vector[%d] PASS  hash160=%s\n", i, got.c_str());
        }
    }
    cudaFree(d_privs); cudaFree(d_h160);
    if (ok) printf("ALL %d VECTORS PASS\n", n);
    else    printf("SOME VECTORS FAILED\n");
    return ok ? 0 : 1;
}

static int mode_scan(const std::vector<Vector> &vecs) {
    // target = vector[0].hash160; scan [priv-1000, priv+1000)
    uint32_t target_priv[8];
    hex_to_limbs(vecs[0].priv_hex, target_priv);
    uint8_t target20[20];
    hex20_to_bytes(vecs[0].h160_hex, target20);

    const uint64_t WINDOW = 1000;
    // base = priv - 1000  (subtract from low limbs; priv >> 1000 so no underflow past limb1 here,
    // but handle borrow generally up the limbs)
    uint32_t base[8];
    for (int i = 0; i < 8; i++) base[i] = target_priv[i];
    {
        int64_t borrow = WINDOW;
        for (int i = 0; i < 8 && borrow; i++) {
            int64_t t = (int64_t)base[i] - (borrow & 0xffffffff);
            int64_t hib = borrow >> 32;
            if (t < 0) { t += (1ll << 32); hib += 1; }
            base[i] = (uint32_t)t;
            borrow = hib;
        }
    }
    uint64_t count = 2 * WINDOW + 1;  // inclusive window around priv

    uint32_t *d_base, *d_found_key; uint8_t *d_target; int *d_found_flag;
    CUDA_CHECK(cudaMalloc(&d_base, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_found_key, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_target, 20));
    CUDA_CHECK(cudaMalloc(&d_found_flag, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_base, base, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, target20, 20, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_found_flag, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_found_key, 0, 8 * sizeof(uint32_t)));

    int threads = 128;
    uint64_t blocks = (count + threads - 1) / threads;
    kernel_scan<<<(unsigned)blocks, threads>>>(d_base, count, d_target, d_found_key, d_found_flag);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int found_flag = 0;
    uint32_t found_key[8];
    CUDA_CHECK(cudaMemcpy(&found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(found_key, d_found_key, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaFree(d_base); cudaFree(d_found_key); cudaFree(d_target); cudaFree(d_found_flag);

    if (found_flag) {
        std::string fk = limbs_to_hex(found_key);
        printf("SCAN: FOUND key=%s (target hash160=%s, window=+/-%llu)\n",
               fk.c_str(), vecs[0].h160_hex.c_str(), (unsigned long long)WINDOW);
        if (fk == vecs[0].priv_hex) {
            printf("SCAN: key matches vector[0].priv EXACTLY\n");
            return 0;
        } else {
            printf("SCAN: FAIL — found key != vector[0].priv (expected %s)\n", vecs[0].priv_hex.c_str());
            return 1;
        }
    } else {
        printf("SCAN: NOT FOUND (FAIL)\n");
        return 1;
    }
}

static int mode_bench(const std::vector<Vector> &vecs, uint64_t total, int iters) {
    // base = vector[0].priv (any starting point works for throughput measurement)
    uint32_t base[8];
    hex_to_limbs(vecs[0].priv_hex, base);

    uint32_t *d_base; uint8_t *d_sink;
    CUDA_CHECK(cudaMalloc(&d_base, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sink, 1024));
    CUDA_CHECK(cudaMemcpy(d_base, base, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    int threads = 256;
    uint64_t blocks = (total + threads - 1) / threads;

    // warm-up
    kernel_bench<<<(unsigned)blocks, threads>>>(d_base, total, d_sink);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> kps;
    for (int it = 0; it < iters; it++) {
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        kernel_bench<<<(unsigned)blocks, threads>>>(d_base, total, d_sink);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        double secs = ms / 1000.0;
        double rate = (double)total / secs;
        kps.push_back(rate);
        printf("  bench iter %2d: %.3f ms  -> %.0f keys/s\n", it, ms, rate);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    cudaFree(d_base); cudaFree(d_sink);

    std::sort(kps.begin(), kps.end());
    double median = kps[kps.size() / 2];
    if (kps.size() % 2 == 0) median = 0.5 * (kps[kps.size()/2 - 1] + kps[kps.size()/2]);
    printf("BENCH: median %.0f keys/s over %d iters (%llu keys/iter)\n",
           median, (int)kps.size(), (unsigned long long)total);

    // write CSV
    system("mkdir -p results");
    FILE *csv = fopen("results/wave1-cuda-keys-per-sec.csv", "w");
    if (csv) {
        fprintf(csv, "metric,value,unit\n");
        fprintf(csv, "median_keys_per_sec,%.0f,keys/s\n", median);
        fprintf(csv, "iterations,%d,count\n", (int)kps.size());
        fprintf(csv, "keys_per_iter,%llu,count\n", (unsigned long long)total);
        fprintf(csv, "min_keys_per_sec,%.0f,keys/s\n", kps.front());
        fprintf(csv, "max_keys_per_sec,%.0f,keys/s\n", kps.back());
        fclose(csv);
        // append nvidia-smi thermal context line
        system("printf 'gpu_thermal_context,' >> results/wave1-cuda-keys-per-sec.csv; "
               "nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader "
               "| head -1 | sed 's/^/\"/;s/$/\"/' >> results/wave1-cuda-keys-per-sec.csv");
        printf("BENCH: wrote results/wave1-cuda-keys-per-sec.csv\n");
    } else {
        fprintf(stderr, "could not write CSV\n");
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *vec_path = "kernels/cuda-ref/test_vectors.json";
    std::string mode = (argc > 1) ? argv[1] : "all";

    // allow vector path override via env for flexibility
    if (const char *p = getenv("VEC_PATH")) vec_path = p;

    std::vector<Vector> vecs = load_vectors(vec_path);
    if (vecs.empty()) { fprintf(stderr, "no vectors loaded from %s\n", vec_path); return 1; }
    printf("Loaded %zu test vectors from %s\n", vecs.size(), vec_path);

    int rc = 0;
    if (mode == "verify") {
        rc = mode_verify(vecs);
    } else if (mode == "scan") {
        rc = mode_scan(vecs);
    } else if (mode == "bench") {
        uint64_t total = (argc > 2) ? strtoull(argv[2], nullptr, 10) : (1ull << 20);
        rc = mode_bench(vecs, total, 12);
    } else { // all
        rc |= mode_verify(vecs);
        rc |= mode_scan(vecs);
        uint64_t total = (argc > 2) ? strtoull(argv[2], nullptr, 10) : (1ull << 20);
        rc |= mode_bench(vecs, total, 12);
    }
    return rc;
}
