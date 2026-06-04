// hash.cuh — on-GPU SHA-256 (big-endian) and RIPEMD-160 (little-endian) for hash160.
//
// hash160(pubkey33) = RIPEMD160(SHA256(pubkey33)).
// SHA-256: big-endian message schedule + big-endian 64-bit length; digest in standard order.
// RIPEMD-160: little-endian word loading + little-endian 64-bit length; output words little-endian.
#ifndef HASH_CUH
#define HASH_CUH

#include <stdint.h>

// ===================== SHA-256 =====================

__device__ __constant__ uint32_t SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

// SHA-256 of exactly a 33-byte message (the compressed pubkey). One block of 64 bytes is enough
// (33 + 1 padding byte + length still fits in one 512-bit block since 33 < 56).
__device__ void sha256_33(uint8_t out[32], const uint8_t msg[33]) {
    uint32_t h[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };
    uint8_t block[64];
#pragma unroll
    for (int i = 0; i < 33; i++) block[i] = msg[i];
    block[33] = 0x80;
#pragma unroll
    for (int i = 34; i < 64; i++) block[i] = 0;
    // message length in bits = 33*8 = 264 = 0x108, big-endian in last 8 bytes
    uint64_t bitlen = 33ull * 8ull;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        block[56 + i] = (uint8_t)(bitlen >> (56 - 8 * i));
    }

    uint32_t w[64];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i*4] << 24) | ((uint32_t)block[i*4+1] << 16)
             | ((uint32_t)block[i*4+2] << 8) | ((uint32_t)block[i*4+3]);
    }
#pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i-15],7) ^ rotr32(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2],17) ^ rotr32(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
#pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = hh + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        hh=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    // output big-endian
#pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i*4+0] = (uint8_t)(h[i] >> 24);
        out[i*4+1] = (uint8_t)(h[i] >> 16);
        out[i*4+2] = (uint8_t)(h[i] >> 8);
        out[i*4+3] = (uint8_t)(h[i]);
    }
}

// ===================== RIPEMD-160 =====================

__device__ __forceinline__ uint32_t rol32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

// f functions
__device__ __forceinline__ uint32_t rmd_f(int j, uint32_t x, uint32_t y, uint32_t z) {
    if (j < 16) return x ^ y ^ z;
    if (j < 32) return (x & y) | (~x & z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

// RIPEMD-160 message-schedule / rotation / constant tables (file-scope constant memory).
__device__ __constant__ uint8_t RMD_rL[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};
__device__ __constant__ uint8_t RMD_rR[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};
__device__ __constant__ uint8_t RMD_sL[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};
__device__ __constant__ uint8_t RMD_sR[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};
__device__ __constant__ uint32_t RMD_KL[5] = {0x00000000,0x5a827999,0x6ed9eba1,0x8f1bbcdc,0xa953fd4e};
__device__ __constant__ uint32_t RMD_KR[5] = {0x50a28be6,0x5c4dd124,0x6d703ef3,0x7a6d76e9,0x00000000};

// RIPEMD-160 of exactly a 32-byte message (the SHA-256 digest). One 64-byte block suffices.
__device__ void ripemd160_32(uint8_t out[20], const uint8_t msg[32]) {
    // load 16 little-endian words; message is 32 bytes, pad to 64.
    uint32_t X[16];
    uint8_t block[64];
#pragma unroll
    for (int i = 0; i < 32; i++) block[i] = msg[i];
    block[32] = 0x80;
#pragma unroll
    for (int i = 33; i < 56; i++) block[i] = 0;
    uint64_t bitlen = 32ull * 8ull;  // 256 bits
#pragma unroll
    for (int i = 0; i < 8; i++) block[56 + i] = (uint8_t)(bitlen >> (8 * i)); // little-endian length
#pragma unroll
    for (int i = 0; i < 16; i++) {
        X[i] = ((uint32_t)block[i*4]) | ((uint32_t)block[i*4+1] << 8)
             | ((uint32_t)block[i*4+2] << 16) | ((uint32_t)block[i*4+3] << 24);
    }

    uint32_t h0=0x67452301, h1=0xefcdab89, h2=0x98badcfe, h3=0x10325476, h4=0xc3d2e1f0;
    uint32_t al=h0,bl=h1,cl=h2,dl=h3,el=h4;
    uint32_t ar=h0,br=h1,cr=h2,dr=h3,er=h4;

#pragma unroll
    for (int j = 0; j < 80; j++) {
        int round = j / 16;
        uint32_t t = rol32(al + rmd_f(j, bl, cl, dl) + X[RMD_rL[j]] + RMD_KL[round], RMD_sL[j]) + el;
        al = el; el = dl; dl = rol32(cl, 10); cl = bl; bl = t;
        int jr = 79 - j; // right-line boolean function uses f(79-j)
        uint32_t tr = rol32(ar + rmd_f(jr, br, cr, dr) + X[RMD_rR[j]] + RMD_KR[round], RMD_sR[j]) + er;
        ar = er; er = dr; dr = rol32(cr, 10); cr = br; br = tr;
    }
    uint32_t tmp = h1 + cl + dr;
    h1 = h2 + dl + er;
    h2 = h3 + el + ar;
    h3 = h4 + al + br;
    h4 = h0 + bl + cr;
    h0 = tmp;

    // output 5 words little-endian
    uint32_t hs[5] = {h0,h1,h2,h3,h4};
#pragma unroll
    for (int i = 0; i < 5; i++) {
        out[i*4+0] = (uint8_t)(hs[i]);
        out[i*4+1] = (uint8_t)(hs[i] >> 8);
        out[i*4+2] = (uint8_t)(hs[i] >> 16);
        out[i*4+3] = (uint8_t)(hs[i] >> 24);
    }
}

// hash160 = ripemd160(sha256(pubkey33))
__device__ void hash160_33(uint8_t out[20], const uint8_t pubkey[33]) {
    uint8_t sha[32];
    sha256_33(sha, pubkey);
    ripemd160_32(out, sha);
}

#endif // HASH_CUH
