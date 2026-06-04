//! secp256k1 256-bit field arithmetic mod p, ported BIT-EXACT from kernels/cuda-ref/secp256k1_field.cuh.
//!
//! Limb layout: [u32; 8], little-endian (limb[0] = bits 0..31).
//! p = 2^256 - 2^32 - 977. Reduction folds high 256 bits via 2^256 = 2^32 + 977.
//!
//! Carry-propagation strategy (a): widen u32 limbs to u64, carry = sum >> 32. No inline asm.
//! 32x32->64 partial products use (a as u64) * (b as u64) which the codegen must lower to a
//! 32-bit multiply with full 64-bit result (PTX mul.wide.u32 / IMAD).

#![allow(clippy::needless_range_loop)]

use cuda_device::device;

pub type Fe = [u32; 8];

// p = 0xFFFFFFFF...FFFFFFFE FFFFFC2F.
// NOTE: cuda-oxide 0.1.0 codegen cannot lower a module-level `const [u32; 8]` constant
// (translate_constant: "Unsupported constant type", ptr_to_array dispatch miss). So p is
// produced by a #[device] fn that builds a LOCAL array literal (the supported store path).
#[device]
pub fn p_limbs() -> Fe {
    [
        0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    ]
}

#[device]
pub fn fe_is_zero(a: &Fe) -> bool {
    let mut x: u32 = 0;
    for i in 0..8 {
        x |= a[i];
    }
    x == 0
}

#[device]
pub fn fe_is_odd(a: &Fe) -> bool {
    (a[0] & 1) == 1
}

#[device]
pub fn fe_equal(a: &Fe, b: &Fe) -> bool {
    let mut d: u32 = 0;
    for i in 0..8 {
        d |= a[i] ^ b[i];
    }
    d == 0
}

// Returns true if a >= p.
#[device]
pub fn fe_ge_p(a: &Fe) -> bool {
    let p = p_limbs();
    // compare from most significant limb down
    let mut i: i32 = 7;
    while i >= 0 {
        let ai = a[i as usize];
        let pi = p[i as usize];
        if ai > pi {
            return true;
        }
        if ai < pi {
            return false;
        }
        i -= 1;
    }
    true // equal => a >= p
}

// r = a - p  (assumes a >= p)
#[device]
pub fn fe_sub_p(a: &Fe) -> Fe {
    let p = p_limbs();
    let mut r: Fe = [0u32; 8];
    let mut borrow: u64 = 0;
    for i in 0..8 {
        // (a[i] - p[i] - borrow) mod 2^64; borrow if it underflowed
        let t = (a[i] as u64).wrapping_sub(p[i] as u64).wrapping_sub(borrow);
        r[i] = t as u32;
        borrow = (t >> 63) & 1; // top bit set => underflow
    }
    r
}

#[device]
pub fn fe_reduce_once(a: &Fe) -> Fe {
    if fe_ge_p(a) {
        fe_sub_p(a)
    } else {
        *a
    }
}

#[device]
pub fn fe_add(a: &Fe, b: &Fe) -> Fe {
    let mut t: Fe = [0u32; 8];
    let mut carry: u64 = 0;
    for i in 0..8 {
        let s = (a[i] as u64) + (b[i] as u64) + carry;
        t[i] = s as u32;
        carry = s >> 32;
    }
    if carry != 0 {
        // fold 2^256 = 2^32 + 977
        let mut c = (t[0] as u64) + 977;
        t[0] = c as u32;
        c >>= 32;
        c += (t[1] as u64) + 1;
        t[1] = c as u32;
        c >>= 32;
        for i in 2..8 {
            c += t[i] as u64;
            t[i] = c as u32;
            c >>= 32;
        }
        if c != 0 {
            let mut c2 = (t[0] as u64) + 977;
            t[0] = c2 as u32;
            c2 >>= 32;
            c2 += (t[1] as u64) + 1;
            t[1] = c2 as u32;
            c2 >>= 32;
            for i in 2..8 {
                c2 += t[i] as u64;
                t[i] = c2 as u32;
                c2 >>= 32;
            }
        }
    }
    fe_reduce_once(&t)
}

#[device]
pub fn fe_sub(a: &Fe, b: &Fe) -> Fe {
    let mut t: Fe = [0u32; 8];
    let mut borrow: i64 = 0;
    for i in 0..8 {
        let s = (a[i] as i64) - (b[i] as i64) - borrow;
        t[i] = s as u32;
        borrow = if s < 0 { 1 } else { 0 };
    }
    if borrow != 0 {
        // add p back
        let p = p_limbs();
        let mut carry: u64 = 0;
        for i in 0..8 {
            let s = (t[i] as u64) + (p[i] as u64) + carry;
            t[i] = s as u32;
            carry = s >> 32;
        }
    }
    t
}

// Fold a 512-bit value (16 x u32 little-endian) into 256-bit result mod p.
#[device]
pub fn fe_fold512(w: &[u32; 16]) -> Fe {
    // acc[0..8] = lo limbs as u64, acc[8] = overflow word
    let mut acc: [u64; 9] = [0u64; 9];
    for i in 0..8 {
        acc[i] = w[i] as u64;
    }
    acc[8] = 0;

    // add hi*977 (hi limbs are w[8..15])
    let mut carry: u64 = 0;
    for i in 0..8 {
        let prod = (w[8 + i] as u64) * 977 + acc[i] + carry;
        acc[i] = (prod as u32) as u64;
        carry = prod >> 32;
    }
    acc[8] += carry;

    // add hi*2^32 (hi shifted up by one limb): add w[8+i] into acc[i+1]
    carry = 0;
    for i in 0..7 {
        let s = acc[i + 1] + (w[8 + i] as u64) + carry;
        acc[i + 1] = (s as u32) as u64;
        carry = s >> 32;
    }
    // i = 7: full-precision overflow word
    let extra = acc[8] + (w[15] as u64) + carry;

    let mut t: Fe = [0u32; 8];
    for i in 0..8 {
        t[i] = acc[i] as u32;
    }

    let e = extra;
    if e != 0 {
        // add e*977
        let mut c = (t[0] as u64) + e * 977;
        t[0] = c as u32;
        c >>= 32;
        for i in 1..8 {
            c += t[i] as u64;
            t[i] = c as u32;
            c >>= 32;
        }
        // add e*2^32 (i.e. add e into limb 1), absorbing leftover c
        let mut c2 = (t[1] as u64) + e + c;
        t[1] = c2 as u32;
        c2 >>= 32;
        for i in 2..8 {
            c2 += t[i] as u64;
            t[i] = c2 as u32;
            c2 >>= 32;
        }
        if c2 != 0 {
            let mut c3 = (t[0] as u64) + c2 * 977;
            t[0] = c3 as u32;
            c3 >>= 32;
            c3 += (t[1] as u64) + c2; // + c2*2^32
            t[1] = c3 as u32;
            c3 >>= 32;
            for i in 2..8 {
                c3 += t[i] as u64;
                t[i] = c3 as u32;
                c3 >>= 32;
            }
        }
    }
    // conditional subtract loop (at most a couple iters)
    let mut rep = 0;
    while rep < 3 {
        if !fe_ge_p(&t) {
            break;
        }
        t = fe_reduce_once(&t);
        rep += 1;
    }
    t
}

#[device]
pub fn fe_mul(a: &Fe, b: &Fe) -> Fe {
    let mut prod: [u32; 16] = [0u32; 16];
    for i in 0..8 {
        let mut carry: u64 = 0;
        for j in 0..8 {
            let t = (a[i] as u64) * (b[j] as u64) + (prod[i + j] as u64) + carry;
            prod[i + j] = t as u32;
            carry = t >> 32;
        }
        prod[i + 8] = (prod[i + 8] as u64 + carry) as u32; // prod[i+8] was 0
    }
    fe_fold512(&prod)
}

#[device]
pub fn fe_sqr(a: &Fe) -> Fe {
    fe_mul(a, a)
}

// modular inverse via Fermat: a^(p-2) mod p, square-and-multiply LSB->MSB.
#[device]
pub fn fe_inv(a: &Fe) -> Fe {
    // p-2 limbs (little-endian): same as p but limb0 = 0xFFFFFC2D
    let e: Fe = [
        0xFFFFFC2D, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    ];
    let mut result: Fe = [0u32; 8];
    result[0] = 1;
    let mut base: Fe = *a;
    for limb in 0..8 {
        let mut bits = e[limb];
        for _b in 0..32 {
            if (bits & 1) == 1 {
                result = fe_mul(&result, &base);
            }
            base = fe_sqr(&base);
            bits >>= 1;
        }
    }
    result
}
