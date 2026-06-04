// secp256k1_ec.cuh — Jacobian point ops + scalar multiplication for secp256k1.
//
// Curve: y^2 = x^3 + 7 over F_p, a = 0, b = 7.
// Jacobian coordinates (X,Y,Z) represent affine (X/Z^2, Y/Z^3). Point at infinity: Z = 0.
// Reference scalar mult is plain double-and-add, MSB->LSB over the 256-bit scalar.
#ifndef SECP256K1_EC_CUH
#define SECP256K1_EC_CUH

#include "secp256k1_field.cuh"

typedef struct {
    fe X, Y, Z;
} jpoint;

// Generator G (affine), big-int values split to little-endian uint32 limbs.
// Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
// Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
__device__ __constant__ uint32_t GX_LIMBS[8] = {
    0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
    0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu
};
__device__ __constant__ uint32_t GY_LIMBS[8] = {
    0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
    0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u
};

__device__ __forceinline__ int jpoint_is_infinity(const jpoint *P) {
    return fe_is_zero(&P->Z);
}

__device__ __forceinline__ void jpoint_set_infinity(jpoint *P) {
    fe_set_zero(&P->X);
    fe_set_zero(&P->Y);  // Y arbitrary
    fe_set_zero(&P->Z);  // Z = 0 marks infinity
    P->X.v[0] = 1u;      // canonical (1:1:0)
    P->Y.v[0] = 1u;
}

// Point doubling: 2P in Jacobian, a=0 (secp256k1).
// Uses the standard a=0 doubling formulas.
__device__ void jpoint_double(jpoint *R, const jpoint *P) {
    if (fe_is_zero(&P->Y) || jpoint_is_infinity(P)) {
        jpoint_set_infinity(R);
        return;
    }
    fe A, B, C, D, X3, Y3, Z3, t1, t2;
    // A = X^2
    fe_sqr(&A, &P->X);
    // B = Y^2
    fe_sqr(&B, &P->Y);
    // C = B^2
    fe_sqr(&C, &B);
    // D = 2*((X+B)^2 - A - C)
    fe_add(&t1, &P->X, &B);     // X+B
    fe_sqr(&t1, &t1);           // (X+B)^2
    fe_sub(&t1, &t1, &A);       // -A
    fe_sub(&t1, &t1, &C);       // -C
    fe_add(&D, &t1, &t1);       // *2
    // E = 3*A
    fe_add(&t2, &A, &A);        // 2A
    fe_add(&t2, &t2, &A);       // 3A  (E)
    // F = E^2
    fe fF;
    fe_sqr(&fF, &t2);
    // X3 = F - 2*D
    fe_add(&t1, &D, &D);        // 2D
    fe_sub(&X3, &fF, &t1);      // F - 2D
    // Y3 = E*(D - X3) - 8*C
    fe_sub(&t1, &D, &X3);       // D - X3
    fe_mul(&Y3, &t2, &t1);      // E*(D-X3)
    fe_add(&t1, &C, &C);        // 2C
    fe_add(&t1, &t1, &t1);      // 4C
    fe_add(&t1, &t1, &t1);      // 8C
    fe_sub(&Y3, &Y3, &t1);      // -8C
    // Z3 = 2*Y*Z
    fe_mul(&Z3, &P->Y, &P->Z);
    fe_add(&Z3, &Z3, &Z3);
    fe_copy(&R->X, &X3);
    fe_copy(&R->Y, &Y3);
    fe_copy(&R->Z, &Z3);
}

// Point addition R = P + Q (both Jacobian). General add (handles P==Q and infinities).
__device__ void jpoint_add(jpoint *R, const jpoint *P, const jpoint *Q) {
    if (jpoint_is_infinity(P)) { fe_copy(&R->X,&Q->X); fe_copy(&R->Y,&Q->Y); fe_copy(&R->Z,&Q->Z); return; }
    if (jpoint_is_infinity(Q)) { fe_copy(&R->X,&P->X); fe_copy(&R->Y,&P->Y); fe_copy(&R->Z,&P->Z); return; }

    fe Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, r_, V, t1, t2;
    fe_sqr(&Z1Z1, &P->Z);            // Z1^2
    fe_sqr(&Z2Z2, &Q->Z);            // Z2^2
    fe_mul(&U1, &P->X, &Z2Z2);       // U1 = X1*Z2^2
    fe_mul(&U2, &Q->X, &Z1Z1);       // U2 = X2*Z1^2
    fe_mul(&t1, &Q->Z, &Z2Z2);       // Z2^3
    fe_mul(&S1, &P->Y, &t1);         // S1 = Y1*Z2^3
    fe_mul(&t2, &P->Z, &Z1Z1);       // Z1^3
    fe_mul(&S2, &Q->Y, &t2);         // S2 = Y2*Z1^3

    if (fe_equal(&U1, &U2)) {
        if (!fe_equal(&S1, &S2)) {
            jpoint_set_infinity(R);  // P == -Q
            return;
        } else {
            jpoint_double(R, P);     // P == Q
            return;
        }
    }
    fe_sub(&H, &U2, &U1);            // H = U2 - U1
    fe_add(&I, &H, &H);             // 2H
    fe_sqr(&I, &I);                 // I = (2H)^2
    fe_mul(&J, &H, &I);             // J = H*I
    fe_sub(&r_, &S2, &S1);         // r = S2 - S1
    fe_add(&r_, &r_, &r_);         // r = 2*(S2-S1)
    fe_mul(&V, &U1, &I);           // V = U1*I
    // X3 = r^2 - J - 2V
    fe X3, Y3, Z3;
    fe_sqr(&X3, &r_);
    fe_sub(&X3, &X3, &J);
    fe_add(&t1, &V, &V);
    fe_sub(&X3, &X3, &t1);
    // Y3 = r*(V - X3) - 2*S1*J
    fe_sub(&t1, &V, &X3);
    fe_mul(&Y3, &r_, &t1);
    fe_mul(&t2, &S1, &J);
    fe_add(&t2, &t2, &t2);
    fe_sub(&Y3, &Y3, &t2);
    // Z3 = 2*Z1*Z2*H  ... computed as ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    fe_add(&t1, &P->Z, &Q->Z);
    fe_sqr(&t1, &t1);
    fe_sub(&t1, &t1, &Z1Z1);
    fe_sub(&t1, &t1, &Z2Z2);
    fe_mul(&Z3, &t1, &H);
    fe_copy(&R->X, &X3);
    fe_copy(&R->Y, &Y3);
    fe_copy(&R->Z, &Z3);
}

// Scalar multiplication: R = k*G, k given as 8 little-endian uint32 limbs.
// Double-and-add, MSB -> LSB.
__device__ void scalar_mul_G(jpoint *R, const uint32_t k[8]) {
    jpoint base;
    // set base = G (affine -> Jacobian with Z=1)
#pragma unroll
    for (int i = 0; i < 8; i++) base.X.v[i] = GX_LIMBS[i];
#pragma unroll
    for (int i = 0; i < 8; i++) base.Y.v[i] = GY_LIMBS[i];
    fe_set_zero(&base.Z); base.Z.v[0] = 1u;

    jpoint acc;
    jpoint_set_infinity(&acc);

    // iterate bits MSB -> LSB
    for (int limb = 7; limb >= 0; limb--) {
        uint32_t bits = k[limb];
        for (int b = 31; b >= 0; b--) {
            jpoint d;
            jpoint_double(&d, &acc);
            fe_copy(&acc.X, &d.X); fe_copy(&acc.Y, &d.Y); fe_copy(&acc.Z, &d.Z);
            if ((bits >> b) & 1u) {
                jpoint s;
                jpoint_add(&s, &acc, &base);
                fe_copy(&acc.X, &s.X); fe_copy(&acc.Y, &s.Y); fe_copy(&acc.Z, &s.Z);
            }
        }
    }
    fe_copy(&R->X, &acc.X);
    fe_copy(&R->Y, &acc.Y);
    fe_copy(&R->Z, &acc.Z);
}

// Convert Jacobian point to affine (x,y). x = X/Z^2, y = Y/Z^3 mod p.
// Returns 1 on success, 0 if P is the point at infinity (Z=0). For the point at
// infinity there is no affine representation; we set (x,y)=(0,0) and flag it rather
// than silently feeding fe_inv(0)=0 into the SEC serialization. Not reachable for
// valid k in [1,n-1] (the scan/verify inputs), but the reference must guard it.
__device__ int jpoint_to_affine(fe *x, fe *y, const jpoint *P) {
    if (jpoint_is_infinity(P)) {
        fe_set_zero(x);
        fe_set_zero(y);
        return 0;
    }
    fe zinv, zinv2, zinv3;
    fe_inv(&zinv, &P->Z);       // 1/Z
    fe_sqr(&zinv2, &zinv);      // 1/Z^2
    fe_mul(&zinv3, &zinv2, &zinv); // 1/Z^3
    fe_mul(x, &P->X, &zinv2);
    fe_mul(y, &P->Y, &zinv3);
    return 1;
}

// Serialize compressed pubkey (33 bytes): prefix (0x02/0x03) + X big-endian.
// x,y must be affine.
__device__ void compressed_pubkey(uint8_t out[33], const fe *x, const fe *y) {
    out[0] = fe_is_odd(y) ? 0x03 : 0x02;
    // X big-endian: limb[7] is most-significant; byte order MSB first.
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t limb = x->v[7 - i];
        out[1 + i * 4 + 0] = (uint8_t)(limb >> 24);
        out[1 + i * 4 + 1] = (uint8_t)(limb >> 16);
        out[1 + i * 4 + 2] = (uint8_t)(limb >> 8);
        out[1 + i * 4 + 3] = (uint8_t)(limb);
    }
}

#endif // SECP256K1_EC_CUH
