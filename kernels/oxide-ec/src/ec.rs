//! secp256k1 Jacobian point ops + scalar mult, ported BIT-EXACT from kernels/cuda-ref/secp256k1_ec.cuh.
//!
//! Jacobian point represented as three Fe ([X,Y,Z]) bundled into [[u32;8];3] = [Fe;3].
//! Index 0=X, 1=Y, 2=Z. Point at infinity: Z = 0.
//! All const generator coords come from #[device] fns (local literals) — cuda-oxide 0.1.0
//! cannot lower module-level const [u32;8] arrays.

#![allow(clippy::needless_range_loop)]

use crate::field::*;
use cuda_device::device;

pub type Jpoint = [Fe; 3]; // [X, Y, Z]

// Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
#[device]
pub fn gx_limbs() -> Fe {
    [
        0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
        0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E,
    ]
}

// Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
#[device]
pub fn gy_limbs() -> Fe {
    [
        0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
        0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77,
    ]
}

#[device]
pub fn jpoint_is_infinity(p: &Jpoint) -> bool {
    fe_is_zero(&p[2])
}

#[device]
pub fn jpoint_set_infinity() -> Jpoint {
    // canonical (1:1:0). Build each inner Fe locally — cuda-oxide 0.1.0 cannot lower a
    // 2-level array-element assignment (r[i][j] = ...).
    let mut x: Fe = [0u32; 8];
    x[0] = 1;
    let mut y: Fe = [0u32; 8];
    y[0] = 1;
    let z: Fe = [0u32; 8];
    [x, y, z]
}

// 2P in Jacobian, a = 0.
#[device]
pub fn jpoint_double(p: &Jpoint) -> Jpoint {
    if fe_is_zero(&p[1]) || jpoint_is_infinity(p) {
        return jpoint_set_infinity();
    }
    let px = p[0];
    let py = p[1];
    let pz = p[2];
    // A = X^2
    let a = fe_sqr(&px);
    // B = Y^2
    let b = fe_sqr(&py);
    // C = B^2
    let c = fe_sqr(&b);
    // D = 2*((X+B)^2 - A - C)
    let mut t1 = fe_add(&px, &b);
    t1 = fe_sqr(&t1);
    t1 = fe_sub(&t1, &a);
    t1 = fe_sub(&t1, &c);
    let d = fe_add(&t1, &t1);
    // E = 3*A
    let mut e = fe_add(&a, &a);
    e = fe_add(&e, &a);
    // F = E^2
    let ff = fe_sqr(&e);
    // X3 = F - 2*D
    let t2d = fe_add(&d, &d);
    let x3 = fe_sub(&ff, &t2d);
    // Y3 = E*(D - X3) - 8*C
    let dmx = fe_sub(&d, &x3);
    let mut y3 = fe_mul(&e, &dmx);
    let mut c8 = fe_add(&c, &c); // 2C
    c8 = fe_add(&c8, &c8); // 4C
    c8 = fe_add(&c8, &c8); // 8C
    y3 = fe_sub(&y3, &c8);
    // Z3 = 2*Y*Z
    let mut z3 = fe_mul(&py, &pz);
    z3 = fe_add(&z3, &z3);
    [x3, y3, z3]
}

// R = P + Q (general add).
#[device]
pub fn jpoint_add(p: &Jpoint, q: &Jpoint) -> Jpoint {
    if jpoint_is_infinity(p) {
        return *q;
    }
    if jpoint_is_infinity(q) {
        return *p;
    }
    let z1z1 = fe_sqr(&p[2]);
    let z2z2 = fe_sqr(&q[2]);
    let u1 = fe_mul(&p[0], &z2z2);
    let u2 = fe_mul(&q[0], &z1z1);
    let t1 = fe_mul(&q[2], &z2z2); // Z2^3
    let s1 = fe_mul(&p[1], &t1);
    let t2 = fe_mul(&p[2], &z1z1); // Z1^3
    let s2 = fe_mul(&q[1], &t2);

    if fe_equal(&u1, &u2) {
        if !fe_equal(&s1, &s2) {
            return jpoint_set_infinity(); // P == -Q
        } else {
            return jpoint_double(p); // P == Q
        }
    }
    let h = fe_sub(&u2, &u1);
    let mut ii = fe_add(&h, &h); // 2H
    ii = fe_sqr(&ii); // I = (2H)^2
    let j = fe_mul(&h, &ii);
    let mut r_ = fe_sub(&s2, &s1);
    r_ = fe_add(&r_, &r_); // r = 2*(S2-S1)
    let v = fe_mul(&u1, &ii);
    // X3 = r^2 - J - 2V
    let mut x3 = fe_sqr(&r_);
    x3 = fe_sub(&x3, &j);
    let v2 = fe_add(&v, &v);
    x3 = fe_sub(&x3, &v2);
    // Y3 = r*(V - X3) - 2*S1*J
    let vmx = fe_sub(&v, &x3);
    let mut y3 = fe_mul(&r_, &vmx);
    let mut s1j = fe_mul(&s1, &j);
    s1j = fe_add(&s1j, &s1j);
    y3 = fe_sub(&y3, &s1j);
    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    let mut z3 = fe_add(&p[2], &q[2]);
    z3 = fe_sqr(&z3);
    z3 = fe_sub(&z3, &z1z1);
    z3 = fe_sub(&z3, &z2z2);
    z3 = fe_mul(&z3, &h);
    [x3, y3, z3]
}

// R = k*G, double-and-add MSB->LSB.
#[device]
pub fn scalar_mul_g(k: &Fe) -> Jpoint {
    let base: Jpoint = {
        let mut z: Fe = [0u32; 8];
        z[0] = 1;
        [gx_limbs(), gy_limbs(), z]
    };
    let mut acc = jpoint_set_infinity();
    let mut limb: i32 = 7;
    while limb >= 0 {
        let bits = k[limb as usize];
        let mut b: i32 = 31;
        while b >= 0 {
            acc = jpoint_double(&acc);
            if ((bits >> b) & 1) == 1 {
                acc = jpoint_add(&acc, &base);
            }
            b -= 1;
        }
        limb -= 1;
    }
    acc
}

// Jacobian -> affine (x,y). x=X/Z^2, y=Y/Z^3. Returns ([x,y], ok).
#[device]
pub fn jpoint_to_affine(p: &Jpoint) -> [Fe; 2] {
    if jpoint_is_infinity(p) {
        return [[0u32; 8], [0u32; 8]];
    }
    let zinv = fe_inv(&p[2]);
    let zinv2 = fe_sqr(&zinv);
    let zinv3 = fe_mul(&zinv2, &zinv);
    let x = fe_mul(&p[0], &zinv2);
    let y = fe_mul(&p[1], &zinv3);
    [x, y]
}

// Compressed pubkey (33 bytes): prefix (0x02/0x03 by y parity) + X big-endian.
// Returned as [u32; 9] where byte j is ((out[j>>2] >> (8*(j&3))) & 0xff) packed little-endian
// within each u32; the host/consumer reads 33 bytes. We instead build a [u8;33]-equivalent in
// [u32;33] for simplicity of indexed writes (one byte per u32 slot is wasteful but trivially
// indexable). To keep it tight we pack into bytes via a [u8; 36] local then return bytes.
#[device]
pub fn compressed_pubkey(x: &Fe, y: &Fe) -> [u8; 33] {
    let mut out: [u8; 33] = [0u8; 33];
    out[0] = if fe_is_odd(y) { 0x03 } else { 0x02 };
    // X big-endian: limb[7] most significant
    for i in 0..8 {
        let limb = x[7 - i];
        out[1 + i * 4] = (limb >> 24) as u8;
        out[1 + i * 4 + 1] = (limb >> 16) as u8;
        out[1 + i * 4 + 2] = (limb >> 8) as u8;
        out[1 + i * 4 + 3] = limb as u8;
    }
    out
}
