# Wave 2 — cuda-oxide secp256k1 port: research notes

**Outcome: PASS_KERNEL.** The full secp256k1 scanner (EC scalar mult -> affine -> compressed
SEC -> SHA-256 -> RIPEMD-160 = hash160) was ported to cuda-oxide 0.1.0 in pure Rust, compiled to
PTX, JIT-loaded, and run on an RTX 5090 (sm_120). It is **bit-exact** vs the Python oracle on all
8 canonical vectors and 32 fresh deterministic vectors (40 total). No inline asm, no float in the
EC/hash path (verified at the SASS level: zero FADD/FMUL/FFMA in any kernel).

The headline is the **throughput gap**: oxide ~287 K keys/s vs same-session cuda-ref ~33.1 M
keys/s — about **115x slower** — and the root cause is fully diagnosed (local-memory array spilling,
not a carry-chain or correctness problem).

Files (all under the allowed paths):
- `kernels/oxide-ec/src/field.rs` — 256-bit field arith mod p (add/sub/mul/sqr/inv).
- `kernels/oxide-ec/src/ec.rs` — Jacobian double/add, scalar mult, affine, compressed SEC.
- `kernels/oxide-ec/src/hash.rs` — SHA-256 (BE) + RIPEMD-160 (LE) -> hash160.
- `kernels/oxide-ec/src/main.rs` — `#[cuda_module]` kernels + host harness (fieldproof/verify/bench).
- `kernels/oxide-ec/Cargo.toml` — deps pinned to cuda-oxide rev `2a03dfd` (see API friction #1).
- `results/wave2-oxide-keys-per-sec.csv`, `results/wave1-cuda-keys-per-sec.csv` (same-session rebench).

---

## 1. `cargo oxide doctor` output

`doctor` must be run from inside a project (it errors "Could not find cuda-oxide workspace" at the
repo root). Inside `kernels/oxide-ec/` with the required env exported **plus** `CUDA_OXIDE_BACKEND`
(see friction #1):

```
cargo-oxide environment check
==============================
Rust nightly toolchain... ✓ rustc 1.96.0-nightly (55e86c996 2026-04-02)
rust-toolchain.toml... ✓ present
Codegen backend... ✓ .../cuda-oxide-.../2a03dfd/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so
CUDA toolkit (nvcc)... ✓ Cuda compilation tools, release 13.2, V13.2.78
libNVVM (libnvvm.so)... ✓ libNVVM 2.0
nvJitLink (libnvJitLink.so)... ✓ nvJitLink 13.2
libdevice (libdevice.10.bc)... ✓ /usr/local/cuda/nvvm/libdevice/libdevice.10.bc
llc (LLVM)... ✓ Ubuntu LLVM version 21.1.8 (llc-21)
clang / libclang resource dir... ✓ /usr/lib/llvm-21/lib/clang/21
cuda-gdb (optional)... ✓ ...
✅ Environment looks good!
```

Every line is `✓`.

## 2. Scaffolding

`cargo oxide new oxide-ec` (run inside `kernels/`) scaffolds a standalone Cargo project
(`src/main.rs`, `Cargo.toml`, `rust-toolchain.toml` pinning `nightly-2026-04-03`). The generated
`vecadd` template did NOT compile as-shipped (`error[E0382]: borrow of moved value: idx` —
`ThreadIndex` is `!Copy`; the fix is `let i = idx.get();` before `c.get_mut(idx)`). The
`#[cuda_module] mod kernels { ... }` + `kernels::from_module(ctx.load_module_from_file("<name>.ptx"))`
typed-launch pattern (from the `numeric_stress` / `helper_fn` examples) is the one that works.

Build/run from inside the standalone project with `cargo oxide build --arch sm_120` /
`cargo oxide run`. The emitted PTX is `oxide_ec.ptx` (underscored crate name), not `oxide-ec.ptx`.

## 3. API frictions (each with the exact error)

### Friction #1 (BLOCKER, then resolved): stale cached codegen backend — a "shadow backend"
A minimal `vecadd` standalone project failed PTX verification deep inside cuda-oxide's own
`index_1d`:

```
error: [rustc_codegen_cuda] Device codegen failed: PTX generation failed:
Verification failed for 'cuda_device::__internal::index_1d':
  ... thread.rs:214:9: 219:10 ... MirConstructStructOp result must be a struct type
  op186v1_res0 = mir.construct_struct (...) : <(builtin.integer ui64,
     mir.struct <PhantomData,[],[],[],[],0>, mir.struct <PhantomData,...>, ...) -> (builtin.integer ui64)>
```

`ThreadIndex { raw: usize, _kernel: PhantomData, _space: PhantomData, _not_send_sync: PhantomData }`
is ABI-collapsed to a bare `usize` (ui64), but the MIR still constructs a 4-field struct — the
verifier rejects the mismatch. Crucially, the **in-repo `array_index` example built and ran fine on
the same machine**, which isolated the cause: `cargo-oxide`'s backend-discovery order
(`backend.rs::find_or_build_backend`) is:
1. `CUDA_OXIDE_BACKEND` env override, 2. local repo build, 3. **cached
`~/.cargo/cuda-oxide/librustc_codegen_cuda.so`**, 4. git auto-fetch.

In-repo builds rebuild the backend from source (path 2, fresh + correct). A standalone project has
no local repo, so it silently falls back to the **stale cached `.so` (dated May 8, 273,950,816
bytes)** which carries the `MirConstructStructOp` regression. The freshly-built backend
(`279,683,344` bytes) does not. This is the same class as the documented "libNVVM shadow bug":
a stale shim on the discovery path produces a wrong result with no obvious signal.

**Resolution:** export
`CUDA_OXIDE_BACKEND=<repo>/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so`
pointing at a freshly-built backend (the in-repo `cargo oxide build <example>` builds it for you).
With that override, the standalone `vecadd` builds and the full secp256k1 port builds. The
`Cargo.toml` is additionally pinned to rev `2a03dfd` (the in-repo source whose backend we built);
note that the deps commit and the backend commit must agree.

### Friction #2: module-level `const [u32; 8]` arrays do not lower
First field-arith build failed:

```
Translation failed: oxide_ec::field::...fe_ge_p: invalid input program.
Unsupported construct: Unsupported constant type in translate_constant.
  Rust type : Array(Uint(U32), 8)   pliron type: MirArrayType { element_ty: Ptr<Box<dyn Type>>, size: 8 }
  const repr : Allocated(... bytes: [47,252,255,255,254,255,...])   <- this is P_LIMBS (0xFFFFFC2F...)
  The type dispatch (ZST -> ptr_to_array -> struct -> enum -> float -> pointer -> integer)
  did not match this constant. A new handler may need to be added.
```

`const P_LIMBS: [u32;8] = [...]` used inside a `#[device]` fn hits an unimplemented constant-array
handler. **Workaround:** emit every constant array from a `#[device] fn p_limbs() -> [u32;8]`
returning a **local array literal** (the supported `store`/`insertvalue` path), not a `const`. Same
fix applied to the curve generator (`gx_limbs`/`gy_limbs`) and all six RIPEMD-160 tables
(`rmd_rl/rr/sl/sr/kl/kr`) and the SHA-256 `K` table. Local array literals lower fine.

### Friction #3: 2-level array-element assignment not implemented
Building the EC layer failed:

```
Translation failed: oxide_ec::ec::...jpoint_set_infinity: src/ec.rs:41:5: 41:16
invalid input program.  Unsupported construct: 2-level projection
ConstantIndex{offset:0,...} -> ConstantIndex{offset:0,...} not yet implemented for assignment
```

A Jacobian point is `[[u32;8];3]`; `r[0][0] = 1` is a nested (2-level) array-element write, which
the codegen cannot lower. **Workaround:** build each inner `[u32;8]` as its own local
(`let mut x=[0u32;8]; x[0]=1;`) and assemble the outer array from those locals
(`[x, y, z]`). Single-level writes (`out[1+i*4] = ...`) and nested *reads* (`fe_sqr(&p[2])`) are
both fine — only nested *writes* are unsupported.

### Friction #4 (ergonomic): `cargo oxide run -- <arg>` collides with the example-name positional
`cargo oxide run` treats the first positional as the example/binary name, so `... run -- fieldproof_in.txt`
tried to build an example called `fieldproof_in.txt` and then loaded the wrong PTX. Run the host
binary directly after building: `./target/release/oxide-ec <mode> <args>` with
`LD_LIBRARY_PATH=/usr/local/cuda/lib64:$(rustc --print sysroot)/lib`.

### Friction #5 (cosmetic): PTX `.target sm_80`, SASS `sm_120`
The emitted PTX text always carries `.target sm_80` / `.version 7.0` regardless of `--arch sm_120`
(`--arch` drives `CUDA_OXIDE_TARGET`, which governs the cubin/JIT arch, not the PTX text header).
This is a forward-compatible virtual arch; `ptxas -arch=sm_120` and the runtime JIT both produce
native sm_120 SASS (`cuobjdump --dump-sass` reports `code for sm_120`). Confirmed sm_120 native; no
tcgen05 / sm_100a-only constructs are used.

## 4. Carry-propagation strategy — what worked

Strategy **(a)** from the task — 8x u32 limbs, carry via u64 widening
(`let s = (a[i] as u64) + (b[i] as u64) + carry; limb = s as u32; carry = s >> 32;`) and
32x32->64 partial products as `(a[i] as u64) * (b[j] as u64)` — **compiles and is bit-exact.**
This was the first strategy tried and it succeeded, so (b) `carrying_add` and (c) explicit `u128`
intermediates were not needed for correctness. (The `primitive_stress` example confirms `u128`
`wrapping_mul`/shifts/`>>64` also lower, so (c) is available as an alternative; `u64` widening was
sufficient and is what the cuda-ref reference itself uses, keeping the port a line-by-line mirror.)

Field-arith proof (`fieldproof` kernel, 16 cases incl. edge values 1, p-1, p-2): **all 16
modmul + modinv results bit-exact** vs `pow(a, p-2, p)` / `(a*b)%p` in Python. This was proven
*before* building the full pipeline, per the task's Step-2 gate.

## 5. PTX / SASS instruction mix (sm_120, both)

| Metric (per *bench* kernel)        | cuda-oxide       | cuda-ref (CUDA-C) |
|------------------------------------|------------------|-------------------|
| SASS target                        | `code for sm_120`| `code for sm_120` |
| Float ops (FADD/FMUL/FFMA)         | 0                | 0                 |
| Registers                          | 168              | 118               |
| **Stack frame**                    | **2504 bytes**   | **96 bytes**      |
| **Local-mem ops (LDL/STL)**        | **1675**         | **167**           |
| IMAD/IADD3 (this kernel's slice)   | 87*              | (heavily inlined) |

(*oxide IMAD/IADD3 counts are lower partly because much of the work is shuffled through
load/store-local rather than arithmetic.)

Neither side has inline asm (cuda-oxide 0.1.0 has none; the cuda-ref reference also avoids
`madc.hi/lo.cc` and uses plain u64-widened carries), so the carry chain is *not* the differentiator.

## 6. Throughput gap — the HEADLINE result (same session, RTX 5090, sm_120)

| Kernel    | Median keys/s | Timing            | Thermal (after)            |
|-----------|---------------|-------------------|-----------------------------|
| cuda-ref  | **33,139,489**| cudaEvent (kernel)| 65 C, 220 W, 100% util      |
| cuda-oxide| **286,907**   | host wall-clock   | 65 C, 319 W, 79% util       |

Gap ≈ **115x** (cuda-ref / oxide). To rule out launch-overhead bias (oxide times host wall-clock
including per-launch overhead; cuda-ref times kernel-only via cudaEvents), oxide was re-benched at
8 M keys/iter (~29 s per launch, launch overhead utterly negligible): it held **287,470 keys/s** —
identical to the 1 M-key run. The gap is therefore **genuine kernel throughput**, not measurement
artifact. (Both numbers are 1,048,576 keys/iter, 12 iters, median; CSVs in `results/`.)

### Root cause of the gap
The SASS resource usage is unambiguous: oxide's pipeline kernel has a **2504-byte stack frame and
1675 local-memory load/stores**, versus cuda-ref's **96-byte frame and 167 LDL/STL**. cuda-oxide
0.1.0 keeps fixed-size arrays (`[u32;8]` field elements, `[u32;16]` products, `[u8;64]` hash
blocks) in **local (stack) memory** — the `alloca + gep + load/store` lowering the `array_index`
example documents — and does not scalarize them into registers across the `#[device]`-function call
boundaries (each helper takes `&[u32;8]` and returns `[u32;8]` by value through memory). nvcc, by
contrast, fully inlines and scalarizes the same arrays into registers, so its inner loops are
register-to-register IMAD/IADD3 with almost no memory traffic. ~10x the local-memory traffic per
key, multiplied across thousands of field operations per scalar multiply, accounts for the ~100x+
slowdown. This is a **codegen-maturity gap (array scalarization / cross-function inlining), not a
language-expressibility or correctness gap.**

## 7. Honest conclusion: oxide vs CUDA-C for this kernel class

- **Expressibility: YES, with workarounds.** cuda-oxide 0.1.0 *can* express bit-exact 256-bit
  modular arithmetic, full EC scalar multiplication, and SHA-256/RIPEMD-160, entirely in safe-ish
  Rust with no inline asm. u64-widened carry chains lower correctly; the result is bit-exact on 40
  vectors. Three real frictions (no `const` arrays, no nested-array writes, stale-backend trap) all
  have clean workarounds. The carry-chain concern in the brief was a non-issue here — the bottleneck
  is array residency, not add-with-carry.
- **Performance: NOT competitive yet.** At ~115x slower than hand-written CUDA-C for the same
  algorithm on the same GPU, oxide 0.1.0 is unsuitable for a production brute-force scanner. The
  deficit is concrete and addressable (array scalarization + aggressive inlining of `#[device]`
  helpers), so it is a maturity gap rather than a fundamental ceiling — but as of 0.1.0 the CUDA-C
  reference is the load-bearing implementation and oxide is a correctness-validating proof of
  portability, not a speed contender.
- **Practical recommendation:** keep cuda-ref as the production kernel. cuda-oxide is valuable as a
  second, independently-written, memory-safe Rust implementation that cross-checks the C kernel
  bit-for-bit (a strong correctness signal), and as a forward bet on the toolchain maturing.

## Reproduce

```bash
export CUDA_HOME=/usr/local/cuda
export LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so
export PATH=/usr/lib/llvm-21/bin:$PATH:$HOME/.cargo/bin
export CUDA_OXIDE_BACKEND=$(ls -d ~/.cargo/git/checkouts/cuda-oxide-*/2a03dfd)/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so
cd kernels/oxide-ec
cargo oxide build --arch sm_120
LD=/usr/local/cuda/lib64:$(rustc --print sysroot)/lib
LD_LIBRARY_PATH=$LD ./target/release/oxide-ec verify ../cuda-ref/test_vectors.json   # 8 vectors, bit-exact
LD_LIBRARY_PATH=$LD ./target/release/oxide-ec fieldproof fieldproof_in.txt           # 16 modmul+modinv
LD_LIBRARY_PATH=$LD ./target/release/oxide-ec bench 1048576 12                        # throughput
```
