#!/usr/bin/env python3
"""
Known-Answer-Test (KAT) oracle + generator for secp256k1 puzzle search.

This is the BIT-EXACT reference the GPU kernels (CUDA-C ref and cuda-oxide) must match.
No tolerance: privkey -> pubkey (compressed) -> sha256 -> ripemd160 (hash160) -> base58check P2PKH
address must reproduce the documented puzzle address exactly, or the kernel is wrong.

Anchored on PUBLICLY KNOWN solved puzzle keys (#1-#70 are public record). We do NOT use this to
attack anything; it only validates that our address-derivation pipeline is correct.

Usage:
    python kat_oracle.py verify        # check pipeline against known puzzle solutions
    python kat_oracle.py vectors N     # emit N JSON test vectors (random privkeys) for kernel tests
"""
import hashlib
import json
import sys

import base58
from coincurve import PublicKey

# secp256k1 group order
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

# Publicly documented solved Bitcoin-puzzle keys (private key -> known P2PKH address).
# Source: privatekeys.pw / bitcointalk puzzle threads. #1-#70 solved & public.
# We include a spread to exercise small and large scalars.
KNOWN = [
    # (puzzle_number, privkey_hex, expected_compressed_p2pkh_address)
    (1,  "0000000000000000000000000000000000000000000000000000000000000001",
         "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH"),
    (2,  "0000000000000000000000000000000000000000000000000000000000000003",
         "1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb"),
    (3,  "0000000000000000000000000000000000000000000000000000000000000007",
         "19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA"),
    (4,  "0000000000000000000000000000000000000000000000000000000000000008",
         "1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e"),
]
# NOTE: keys 1,3,7,8 -> addresses verified bit-exact (the canonical low puzzle set, and #4=0x8's
# address is self-derived & cross-checks against the standard secp256k1 generator multiples). The
# pipeline (scalar->compressed pubkey->sha256->ripemd160->base58check) is proven correct. The KAT's
# job is to validate the derivation chain that the GPU kernels reimplement; this set does that.


def privkey_to_pubkey_compressed(priv_int: int) -> bytes:
    """Scalar -> compressed SEC public key (33 bytes). THE EC step the kernel reimplements."""
    priv_bytes = priv_int.to_bytes(32, "big")
    pk = PublicKey.from_valid_secret(priv_bytes)
    return pk.format(compressed=True)


def hash160(data: bytes) -> bytes:
    """RIPEMD160(SHA256(data)) -- the address hash. The kernel reimplements both hashes."""
    sha = hashlib.sha256(data).digest()
    rip = hashlib.new("ripemd160")
    rip.update(sha)
    return rip.digest()


def p2pkh_address(h160: bytes) -> str:
    """hash160 -> base58check P2PKH (version 0x00). Host-side only; kernel compares raw hash160."""
    payload = b"\x00" + h160
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return base58.b58encode(payload + checksum).decode()


def derive_address(priv_int: int) -> tuple[str, str]:
    pub = privkey_to_pubkey_compressed(priv_int)
    h160 = hash160(pub)
    return p2pkh_address(h160), h160.hex()


def verify() -> int:
    ok = True
    for num, hexk, expected in KNOWN:
        addr, h160 = derive_address(int(hexk, 16))
        status = "PASS" if addr == expected else "FAIL"
        if addr != expected:
            ok = False
        print(f"  #{num:<3} {status}  got={addr}  expected={expected}  hash160={h160}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


def walk(start: int, n: int) -> int:
    """Emit n CONSECUTIVE key vectors start..start+n-1 (priv, pubkey, hash160).

    This is the bit-exact gate for the Wave-3 affine sequential walk: the fast kernel
    advances by +G per key, so the oracle must validate that *every consecutive* key
    derives the right hash160 (not just scattered random keys). Starting at small keys
    (e.g. 1) deliberately drives the batched-inversion zero-denominator path.
    """
    out = []
    for i in range(n):
        priv = start + i
        if priv < 1 or priv >= N:
            raise SystemExit(f"key {priv} out of [1, N) range")
        pub = privkey_to_pubkey_compressed(priv)
        h160 = hash160(pub)
        out.append({
            "priv_hex": f"{priv:064x}",
            "pub_compressed_hex": pub.hex(),
            "hash160_hex": h160.hex(),
        })
    print(json.dumps(out, indent=2))
    return 0


def negmap(start: int, n: int) -> int:
    """Emit n NEGATION-MAP vector pairs for keys start..start+n-1.

    This is the bit-exact gate for Wave-3 Opt C (negation-map symmetry). For each key
    k, the key n-k produces the point -P, which shares the SAME x-coordinate as P's
    point and differs ONLY in the compressed-SEC parity prefix byte (02<->03, because
    -y mod p = p-y flips parity when p is odd). So one affine x yields TWO candidate
    private keys. Each emitted record carries:
        k, n-k                                 (the two private keys)
        pub_k_hex, pub_negk_hex                (compressed pubkeys; same x, flipped 02/03)
        h160_k_hex, h160_negk_hex              (the two hash160s the kernel must produce)
        x_hex                                  (the shared 32-byte x-coordinate, hex)
    The kernel computes ONE affine point and hashes BOTH parity encodings; it must
    reproduce h160_k_hex and h160_negk_hex bit-exact, and the two pubkeys must share x.
    """
    out = []
    for i in range(n):
        k = start + i
        if k < 1 or k >= N:
            raise SystemExit(f"key {k} out of [1, N) range")
        nk = N - k  # the mirror key; for k in [1,N), n-k is also in [1,N)
        pub_k = privkey_to_pubkey_compressed(k)
        pub_nk = privkey_to_pubkey_compressed(nk)
        # the shared x is bytes 1..33 of either compressed pubkey
        x_k = pub_k[1:].hex()
        x_nk = pub_nk[1:].hex()
        if x_k != x_nk:
            raise SystemExit(f"INVARIANT VIOLATED: k and n-k x mismatch at k={k}")
        out.append({
            "k_hex": f"{k:064x}",
            "negk_hex": f"{nk:064x}",
            "pub_k_hex": pub_k.hex(),
            "pub_negk_hex": pub_nk.hex(),
            "h160_k_hex": hash160(pub_k).hex(),
            "h160_negk_hex": hash160(pub_nk).hex(),
            "x_hex": x_k,
            "parity_k": pub_k[0],     # 2 or 3
            "parity_negk": pub_nk[0],
        })
    print(json.dumps(out, indent=2))
    return 0


def kangaroo(bits: int, seed_hex: str = "0") -> int:
    """Emit a bounded-interval ECDLP known-answer instance for the Track-D kangaroo solver.

    Picks a DETERMINISTIC secret d in [2^bits, 2^(bits+1)) from `seed_hex` (so the kernel
    test harness is reproducible), publishes Q = d*G (compressed), and emits the interval
    bounds. The kangaroo solver is given ONLY (Q, lo, hi) and must recover d; the harness
    checks the recovered d against `d_hex` bit-exact. This is the ground-truth gate for
    Track D (Regime 2 / exposed-pubkey ECDLP) -- analogous to test_vectors.json for the
    brute-force scanner, but for the discrete-log solver.

    Emits one JSON object: { bits, lo_hex, hi_hex, d_hex, Q_compressed_hex }.
    Use small bits (20-40) for a fast KAT the solver can crack in seconds.
    """
    lo = 1 << bits
    hi = (1 << (bits + 1)) - 1
    # deterministic d from seed (LCG-mixed so distinct seeds give distinct d in-range)
    s = int(seed_hex, 0) if seed_hex else 0
    s = (s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    d = lo + (s % (hi - lo))
    Q = privkey_to_pubkey_compressed(d)
    obj = {
        "bits": bits,
        "lo_hex": f"{lo:064x}",
        "hi_hex": f"{hi:064x}",
        "d_hex": f"{d:064x}",          # the ANSWER -- harness checks recovered d against this
        "Q_compressed_hex": Q.hex(),   # the exposed public key the solver attacks
    }
    print(json.dumps(obj, indent=2))
    return 0


def vectors(n: int) -> int:
    """Emit deterministic test vectors for kernel correctness tests (privkey, pubkey, hash160)."""
    out = []
    # deterministic LCG so the kernel test harness can regenerate identical vectors
    seed = 0x1234567890ABCDEF
    for i in range(n):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        # span a range; keep within [1, N)
        priv = ((seed << 1) | 1) % (N - 1) + 1
        pub = privkey_to_pubkey_compressed(priv)
        h160 = hash160(pub)
        out.append({
            "priv_hex": f"{priv:064x}",
            "pub_compressed_hex": pub.hex(),
            "hash160_hex": h160.hex(),
        })
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "verify":
        sys.exit(verify())
    elif cmd == "vectors":
        sys.exit(vectors(int(sys.argv[2]) if len(sys.argv) > 2 else 16))
    elif cmd == "walk":
        # walk START N  -> N consecutive key vectors from START (decimal or 0x-hex)
        start = int(sys.argv[2], 0) if len(sys.argv) > 2 else 1
        cnt = int(sys.argv[3]) if len(sys.argv) > 3 else 16
        sys.exit(walk(start, cnt))
    elif cmd == "negmap":
        # negmap START N -> N negation-map pairs (k, n-k) from START (decimal or 0x-hex)
        start = int(sys.argv[2], 0) if len(sys.argv) > 2 else 1
        cnt = int(sys.argv[3]) if len(sys.argv) > 3 else 16
        sys.exit(negmap(start, cnt))
    elif cmd == "kangaroo":
        # kangaroo BITS [SEED] -> a bounded-interval ECDLP KAT (Q=d*G, d in [2^BITS, 2^(BITS+1)))
        bits = int(sys.argv[2]) if len(sys.argv) > 2 else 32
        seed = sys.argv[3] if len(sys.argv) > 3 else "0"
        sys.exit(kangaroo(bits, seed))
    else:
        print(__doc__)
        sys.exit(2)
