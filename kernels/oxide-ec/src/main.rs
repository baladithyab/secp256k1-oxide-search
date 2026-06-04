//! Wave 2 — secp256k1 scanner ported to cuda-oxide (Rust -> PTX, sm_120).
//!
//! Pipeline per private key k (mirrors kernels/cuda-ref/, bit-exact vs scripts/kat_oracle.py):
//!   P = k*G (Jacobian double-and-add) -> affine -> compressed SEC (33B)
//!   -> sha256 -> ripemd160 = hash160 (20B).
//!
//! Modes (argv[1]):
//!   fieldproof <in.txt>   prove modmul+modinv bit-exact (Step 2 harness)
//!   verify <vectors.json> compute hash160 for each priv, print hex (host diffs vs oracle)
//!   bench <total> <iters> throughput; writes results CSV

mod ec;
mod field;
mod hash;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;
use std::time::Instant;

#[cuda_module]
mod kernels {
    use super::*;
    use crate::ec::{compressed_pubkey, jpoint_to_affine, scalar_mul_g};
    use crate::field::{fe_inv, fe_mul, Fe};
    use crate::hash::hash160_33;

    /// Field-arith proof: per case i, read a,b (16 u32) -> [a*b mod p (8), a^-1 mod p (8)].
    #[kernel]
    pub fn fieldproof(inp: &[u32], mut out: DisjointSlice<u32>, count: u32) {
        let i = thread::index_1d().get();
        if (i as u32) >= count {
            return;
        }
        let mut a: Fe = [0u32; 8];
        let mut b: Fe = [0u32; 8];
        for j in 0..8 {
            a[j] = inp[i * 16 + j];
            b[j] = inp[i * 16 + 8 + j];
        }
        let m = fe_mul(&a, &b);
        let inv = fe_inv(&a);
        // SAFETY: thread i owns disjoint output stripe [i*16 .. i*16+16).
        for j in 0..8 {
            unsafe {
                *out.get_unchecked_mut(i * 16 + j) = m[j];
            }
        }
        for j in 0..8 {
            unsafe {
                *out.get_unchecked_mut(i * 16 + 8 + j) = inv[j];
            }
        }
    }

    /// Full pipeline: per key i, read priv (8 u32 LE) -> hash160 (20 bytes), packed as 5 u32 LE.
    /// Output layout: out[i*5 + w] holds bytes (4w..4w+4) of the 20-byte hash160, little-endian
    /// within the u32 (byte b at out[i*5 + b/4] >> (8*(b%4))).
    #[kernel]
    pub fn hash160_batch(privs: &[u32], mut out: DisjointSlice<u32>, count: u32) {
        let i = thread::index_1d().get();
        if (i as u32) >= count {
            return;
        }
        let mut k: Fe = [0u32; 8];
        for j in 0..8 {
            k[j] = privs[i * 8 + j];
        }
        let p = scalar_mul_g(&k);
        let xy = jpoint_to_affine(&p);
        let pub33 = compressed_pubkey(&xy[0], &xy[1]);
        let h = hash160_33(&pub33);
        // pack 20 bytes -> 5 u32 little-endian
        for w in 0..5 {
            let word = (h[w * 4] as u32)
                | ((h[w * 4 + 1] as u32) << 8)
                | ((h[w * 4 + 2] as u32) << 16)
                | ((h[w * 4 + 3] as u32) << 24);
            // SAFETY: thread i owns disjoint stripe [i*5 .. i*5+5).
            unsafe {
                *out.get_unchecked_mut(i * 5 + w) = word;
            }
        }
    }

    /// Bench: same full pipeline, scatter a checksum so work isn't elided.
    /// key = base + i (low-limb increment with carry). Writes h160 byte-xor into sink[i & 1023].
    #[kernel]
    pub fn bench(base: &[u32], total: u32, mut sink: DisjointSlice<u32>) {
        let i = thread::index_1d().get();
        if (i as u32) >= total {
            return;
        }
        let mut k: Fe = [0u32; 8];
        for j in 0..8 {
            k[j] = base[j];
        }
        // k += i (32-bit add into limb0 with carry up)
        let s0 = (k[0] as u64) + (i as u64);
        k[0] = s0 as u32;
        let mut carry = s0 >> 32;
        let mut j = 1;
        while j < 8 && carry != 0 {
            let s = (k[j] as u64) + carry;
            k[j] = s as u32;
            carry = s >> 32;
            j += 1;
        }
        let p = scalar_mul_g(&k);
        let xy = jpoint_to_affine(&p);
        let pub33 = compressed_pubkey(&xy[0], &xy[1]);
        let h = hash160_33(&pub33);
        let mut acc: u32 = 0;
        for b in 0..20 {
            acc ^= h[b] as u32;
        }
        // SAFETY: scatter into a small 1024-slot buffer; benign races acceptable for a liveness sink.
        unsafe {
            *sink.get_unchecked_mut(i & 1023) = acc;
        }
    }
}

fn hex_to_limbs(hex: &str) -> [u32; 8] {
    let mut h = hex.to_string();
    while h.len() < 64 {
        h = format!("0{}", h);
    }
    let mut out = [0u32; 8];
    for i in 0..8 {
        let grp = &h[(7 - i) * 8..(7 - i) * 8 + 8];
        out[i] = u32::from_str_radix(grp, 16).unwrap();
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module_from_file("oxide_ec.ptx")?;
    let module = kernels::from_module(module).expect("typed module init");

    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("verify");

    match mode {
        "fieldproof" => {
            let inp_path = args.get(2).map(|s| s.as_str()).unwrap_or("fieldproof_in.txt");
            let text = std::fs::read_to_string(inp_path)?;
            let mut lines = text.lines();
            let count: usize = lines.next().unwrap().trim().parse()?;
            let mut inp: Vec<u32> = Vec::with_capacity(count * 16);
            for _ in 0..count {
                for tok in lines.next().unwrap().split_whitespace() {
                    inp.push(u32::from_str_radix(tok.trim_start_matches("0x"), 16)?);
                }
            }
            let inp_dev = DeviceBuffer::from_host(&stream, &inp)?;
            let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, count * 16)?;
            module.fieldproof(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(count as u32),
                &inp_dev,
                &mut out_dev,
                count as u32,
            )?;
            let out = out_dev.to_host_vec(&stream)?;
            for c in 0..count {
                let mut s = String::new();
                for j in 0..16 {
                    s.push_str(&format!("{:08x} ", out[c * 16 + j]));
                }
                println!("{}", s.trim());
            }
        }
        "verify" => {
            let vec_path = args
                .get(2)
                .map(|s| s.as_str())
                .unwrap_or("kernels/cuda-ref/test_vectors.json");
            let json = std::fs::read_to_string(vec_path)?;
            let privs = extract_values(&json, "priv_hex");
            let n = privs.len();
            let mut h_privs: Vec<u32> = Vec::with_capacity(n * 8);
            for p in &privs {
                h_privs.extend_from_slice(&hex_to_limbs(p));
            }
            let privs_dev = DeviceBuffer::from_host(&stream, &h_privs)?;
            let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, n * 5)?;
            module.hash160_batch(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(n as u32),
                &privs_dev,
                &mut out_dev,
                n as u32,
            )?;
            let out = out_dev.to_host_vec(&stream)?;
            for i in 0..n {
                // reconstruct 20 bytes from 5 LE words
                let mut s = String::new();
                for w in 0..5 {
                    let word = out[i * 5 + w];
                    for b in 0..4 {
                        s.push_str(&format!("{:02x}", (word >> (8 * b)) & 0xff));
                    }
                }
                println!("{}", s);
            }
        }
        "bench" => {
            let total: u32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1 << 20);
            let iters: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(12);
            // base = vector[0].priv-ish; any base works for throughput
            let base = hex_to_limbs("000000000000000000000000000000000000000000000001b8af6534e1be8aa6");
            let base_dev = DeviceBuffer::from_host(&stream, &base)?;
            let mut sink = DeviceBuffer::<u32>::zeroed(&stream, 1024)?;
            let cfg = LaunchConfig::for_num_elems(total);
            // warm-up
            module.bench((stream).as_ref(), cfg, &base_dev, total, &mut sink)?;
            stream.synchronize()?;
            let mut rates: Vec<f64> = Vec::new();
            for _it in 0..iters {
                let t0 = Instant::now();
                module.bench((stream).as_ref(), cfg, &base_dev, total, &mut sink)?;
                stream.synchronize()?;
                let secs = t0.elapsed().as_secs_f64();
                let rate = (total as f64) / secs;
                rates.push(rate);
                eprintln!("  bench iter {:2}: {:.3} ms -> {:.0} keys/s", _it, secs * 1e3, rate);
            }
            rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = rates[rates.len() / 2];
            println!("BENCH: median {:.0} keys/s over {} iters ({} keys/iter)", median, iters, total);
            println!("MEDIAN_KEYS_PER_SEC={:.0}", median);
            println!("MIN_KEYS_PER_SEC={:.0}", rates[0]);
            println!("MAX_KEYS_PER_SEC={:.0}", rates[rates.len() - 1]);
        }
        other => {
            eprintln!("unknown mode: {}", other);
            std::process::exit(2);
        }
    }
    Ok(())
}

// Minimal JSON string-value extractor (same approach as cuda-ref host).
fn extract_values(json: &str, key: &str) -> Vec<String> {
    let needle = format!("\"{}\"", key);
    let mut out = Vec::new();
    let bytes = json.as_bytes();
    let mut pos = 0usize;
    while let Some(rel) = json[pos..].find(&needle) {
        let kstart = pos + rel + needle.len();
        if let Some(colon) = json[kstart..].find(':') {
            let after = kstart + colon + 1;
            if let Some(q1rel) = json[after..].find('"') {
                let q1 = after + q1rel + 1;
                if let Some(q2rel) = json[q1..].find('"') {
                    let q2 = q1 + q2rel;
                    out.push(json[q1..q2].to_string());
                    pos = q2 + 1;
                    let _ = bytes;
                    continue;
                }
            }
        }
        break;
    }
    out
}
