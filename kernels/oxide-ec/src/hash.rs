//! SHA-256 (big-endian) + RIPEMD-160 (little-endian) -> hash160, ported BIT-EXACT from
//! kernels/cuda-ref/hash.cuh. Integer-only. Constant tables come from #[device] fns returning
//! local array literals (cuda-oxide 0.1.0 cannot lower module-level const arrays).

#![allow(clippy::needless_range_loop)]

use cuda_device::device;

#[device]
fn rotr32(x: u32, n: u32) -> u32 {
    (x >> n) | (x << (32 - n))
}

#[device]
fn sha256_k() -> [u32; 64] {
    [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ]
}

// SHA-256 of exactly a 33-byte message -> 32-byte digest.
#[device]
pub fn sha256_33(msg: &[u8; 33]) -> [u8; 32] {
    let k = sha256_k();
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut block: [u8; 64] = [0u8; 64];
    for i in 0..33 {
        block[i] = msg[i];
    }
    block[33] = 0x80;
    // length in bits = 264 = 0x108, big-endian in last 8 bytes
    let bitlen: u64 = 33 * 8;
    for i in 0..8 {
        block[56 + i] = (bitlen >> (56 - 8 * i)) as u8;
    }
    let mut w: [u32; 64] = [0u32; 64];
    for i in 0..16 {
        w[i] = ((block[i * 4] as u32) << 24)
            | ((block[i * 4 + 1] as u32) << 16)
            | ((block[i * 4 + 2] as u32) << 8)
            | (block[i * 4 + 3] as u32);
    }
    for i in 16..64 {
        let s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        let s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let mut a = h[0];
    let mut b = h[1];
    let mut c = h[2];
    let mut d = h[3];
    let mut e = h[4];
    let mut f = h[5];
    let mut g = h[6];
    let mut hh = h[7];
    for i in 0..64 {
        let s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(k[i])
            .wrapping_add(w[i]);
        let s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
    let mut out: [u8; 32] = [0u8; 32];
    for i in 0..8 {
        out[i * 4] = (h[i] >> 24) as u8;
        out[i * 4 + 1] = (h[i] >> 16) as u8;
        out[i * 4 + 2] = (h[i] >> 8) as u8;
        out[i * 4 + 3] = h[i] as u8;
    }
    out
}

// ===================== RIPEMD-160 =====================

#[device]
fn rol32(x: u32, n: u32) -> u32 {
    (x << n) | (x >> (32 - n))
}

#[device]
fn rmd_f(j: i32, x: u32, y: u32, z: u32) -> u32 {
    if j < 16 {
        x ^ y ^ z
    } else if j < 32 {
        (x & y) | (!x & z)
    } else if j < 48 {
        (x | !y) ^ z
    } else if j < 64 {
        (x & z) | (y & !z)
    } else {
        x ^ (y | !z)
    }
}

#[device]
fn rmd_rl() -> [u8; 80] {
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
        3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
        1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
        4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13,
    ]
}

#[device]
fn rmd_rr() -> [u8; 80] {
    [
        5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
        6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
        15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
        8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
        12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11,
    ]
}

#[device]
fn rmd_sl() -> [u8; 80] {
    [
        11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
        7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
        11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
        11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
        9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6,
    ]
}

#[device]
fn rmd_sr() -> [u8; 80] {
    [
        8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
        9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
        9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
        15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
        8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11,
    ]
}

#[device]
fn rmd_kl() -> [u32; 5] {
    [0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e]
}

#[device]
fn rmd_kr() -> [u32; 5] {
    [0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000]
}

// RIPEMD-160 of exactly a 32-byte message -> 20-byte digest.
#[device]
pub fn ripemd160_32(msg: &[u8; 32]) -> [u8; 20] {
    let rl = rmd_rl();
    let rr = rmd_rr();
    let sl = rmd_sl();
    let sr = rmd_sr();
    let kl = rmd_kl();
    let kr = rmd_kr();

    let mut block: [u8; 64] = [0u8; 64];
    for i in 0..32 {
        block[i] = msg[i];
    }
    block[32] = 0x80;
    let bitlen: u64 = 32 * 8; // 256
    for i in 0..8 {
        block[56 + i] = (bitlen >> (8 * i)) as u8; // little-endian length
    }
    let mut x: [u32; 16] = [0u32; 16];
    for i in 0..16 {
        x[i] = (block[i * 4] as u32)
            | ((block[i * 4 + 1] as u32) << 8)
            | ((block[i * 4 + 2] as u32) << 16)
            | ((block[i * 4 + 3] as u32) << 24);
    }

    let h0: u32 = 0x67452301;
    let h1: u32 = 0xefcdab89;
    let h2: u32 = 0x98badcfe;
    let h3: u32 = 0x10325476;
    let h4: u32 = 0xc3d2e1f0;
    let mut al = h0;
    let mut bl = h1;
    let mut cl = h2;
    let mut dl = h3;
    let mut el = h4;
    let mut ar = h0;
    let mut br = h1;
    let mut cr = h2;
    let mut dr = h3;
    let mut er = h4;

    for j in 0..80 {
        let round = (j / 16) as usize;
        let t = rol32(
            al.wrapping_add(rmd_f(j as i32, bl, cl, dl))
                .wrapping_add(x[rl[j] as usize])
                .wrapping_add(kl[round]),
            sl[j] as u32,
        )
        .wrapping_add(el);
        al = el;
        el = dl;
        dl = rol32(cl, 10);
        cl = bl;
        bl = t;
        let jr = 79 - (j as i32);
        let tr = rol32(
            ar.wrapping_add(rmd_f(jr, br, cr, dr))
                .wrapping_add(x[rr[j] as usize])
                .wrapping_add(kr[round]),
            sr[j] as u32,
        )
        .wrapping_add(er);
        ar = er;
        er = dr;
        dr = rol32(cr, 10);
        cr = br;
        br = tr;
    }
    let tmp = h1.wrapping_add(cl).wrapping_add(dr);
    let nh1 = h2.wrapping_add(dl).wrapping_add(er);
    let nh2 = h3.wrapping_add(el).wrapping_add(ar);
    let nh3 = h4.wrapping_add(al).wrapping_add(br);
    let nh4 = h0.wrapping_add(bl).wrapping_add(cr);
    let nh0 = tmp;

    let hs: [u32; 5] = [nh0, nh1, nh2, nh3, nh4];
    let mut out: [u8; 20] = [0u8; 20];
    for i in 0..5 {
        out[i * 4] = hs[i] as u8;
        out[i * 4 + 1] = (hs[i] >> 8) as u8;
        out[i * 4 + 2] = (hs[i] >> 16) as u8;
        out[i * 4 + 3] = (hs[i] >> 24) as u8;
    }
    out
}

// hash160 = ripemd160(sha256(pubkey33))
#[device]
pub fn hash160_33(pubkey: &[u8; 33]) -> [u8; 20] {
    let sha = sha256_33(pubkey);
    ripemd160_32(&sha)
}
