# Montgomery Modular Multiplication

## Overview

Computes `r = a * b * R^{-1} mod n` where `R = 2^K`, using only Boolean
operations (AND, XOR, OR, NOT).

This benchmark is a **tight-radix Montgomery instance** at a
Kyber-sized modulus: `K = 12`, `q = 3329` (canonical Kyber uses
`R = 2^16`; ML-DSA / Dilithium a 32-bit radix over `q = 8380417`).

Montgomery multiplication itself is a reduction primitive that
underlies, in their canonical (non-tight-radix) forms:

- **Post-quantum cryptography**: NTT in CRYSTALS-Kyber (ML-KEM) and
  CRYSTALS-Dilithium (ML-DSA), the NIST PQC standards
- **RSA**: Modular exponentiation via repeated Montgomery multiplication;
  **ECC**: finite-field modular arithmetic inside scalar multiplication
- **Homomorphic encryption**: Ring arithmetic in BFV/CKKS schemes

## Algorithm

Montgomery multiplication (1985) avoids expensive trial division by
replacing `mod n` with a right-shift by K bits. The algorithm:

```
Input:  a, b in [0, n), modulus n (odd), n' = -n^{-1} mod 2^K
Output: r = a * b * R^{-1} mod n, where R = 2^K

Step 1: t = a * b                        (2K-bit product)
Step 2: m = (t mod R) * n' mod R         (reduction factor)
Step 3: u = m * n                        (2K-bit product)
Step 4: s = t + u                        (2K+1-bit sum)
Step 5: r = s >> K                       (right shift = take high bits)
Step 6: if r >= n: r = r - n             (conditional correction)
```

The key insight: `t + m*n` is always divisible by `R = 2^K` (by construction
of `m`), so the right-shift in Step 5 is exact. This replaces division by `n`
with division by a power of 2 (a free shift).

## Implementation

The `.bs` program decomposes all arithmetic into gate-level operations:

- **Multiply** (Steps 1, 2, 3): Schoolbook bit-serial multiply using AND for
  partial products and full-adder chains (XOR/AND/OR) for accumulation.
  Step 2 uses a triangular loop (`for i in 0..K-j`) to skip positions >= K
  (discarded by mod 2^K), saving ~half the multiply ops.
- **Addition** (Step 4): 2K-bit ripple-carry adder with overflow tracking.
  The sum `t + u` can be up to 2K+1 bits; the overflow bit is captured.
- **Conditional correction** (Step 6): Two's complement subtraction
  (`r + ~n + 1`) followed by a per-instance bitwise mux. The overflow
  bit from Step 4 is OR'd into the subtract decision.

## Parameters

| Param | Meaning | Analyzed values |
|-------|---------|-----------------|
| K | Bit-width of modulus | 12 (Kyber modulus q=3329), 16, 32 |
| PK | Product width (2*K) | 24, 32, 64 |
| n | Modulus (odd prime) | 3329, 65521, 4294967291 |
| n' | `-n^{-1} mod 2^K` | Precomputed per (n, K) |

Test data is provided at K=12 (unit + tiered) and K=16 (unit). K=32 is
included for op-count analysis only.

## Operation Count

`15K^2 + 23K + 1` ops per Montgomery multiplication (exact).
Flat critical-path depth on the unrolled dependency DAG is `10K + 9`.

| K | n | Application | Ops | Depth |
|---|---|-------------|-----|-------|
| 12 | 3329 | Kyber modulus (R=2^12) | 2,437 | 129 |
| 16 | 65521 | General-purpose | 4,209 | 169 |
| 32 | 4294967291 | Wide arithmetic | 16,097 | 329 |

## Source

- Montgomery, P. "Modular multiplication without trial division."
  *Mathematics of Computation*, 44(170):519-521, 1985.
- Avanzi, R. et al. "CRYSTALS-Kyber: Algorithm Specifications and
  Supporting Documentation." NIST PQC Round 3, version 3.02, August 2021.
- NIST. FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism
  Standard (ML-KEM, based on CRYSTALS-Kyber), August 2024.
- NIST. FIPS 204: Module-Lattice-Based Digital Signature Standard
  (ML-DSA, based on CRYSTALS-Dilithium), August 2024.
- Zhang, J., Imani, M., Sadredini, E. "BP-NTT: Fast and Compact in-SRAM
  Number Theoretic Transform with Bit-Parallel Modular Multiplication."
  DAC 2023. (Implements this kernel in SRAM PIM.)

## Tier Data

| Tier | Items | Size |
|------|-------|------|
| small | 6.67M | 30 MB |
| medium | 66.7M | 300 MB |
| large | 666.7M | 3.0 GB |

Tier `.bsdata` files are generated on demand (`generate_tests.py --tier <t>`).

Tier inputs are random K-bit stress operands.

## Connection to Three Communities

This domain bridges all three bitstream computing communities:
- **Cryptography**: Montgomery multiplication underlies many
  modular-arithmetic public-key systems and lattice-based PQC kernels
  (e.g., ML-KEM / ML-DSA NTTs); hash-based PQ schemes such as SLH-DSA
  do not use it
- **PIM**: BP-NTT (DAC 2023) implements bit-parallel modular multiply in
  SRAM; CHOPPER/PIMsynth could compile this .bs program to DRAM
- **Text processing**: The full-adder chains are the same carry-propagation
  primitive used in Parabix's Advance/ScanThru operations
