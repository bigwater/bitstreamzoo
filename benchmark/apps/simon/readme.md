# SIMON 32/64 Block Cipher — Bitstream Benchmark

## Overview

This benchmark implements the SIMON 32/64 lightweight block cipher using the bitstream
DSL. Each bit position is a different plaintext being encrypted with the same key. The
cipher encrypts multiple plaintexts simultaneously in a single execution.

> Beaulieu, R., Shors, D., Smith, J., Treatman-Clark, S., Weeks, B., and Wingers, L.
> "The SIMON and SPECK Families of Lightweight Block Ciphers." *IACR ePrint* 2013/404,
> 2013.

**Why SIMON for bitstreams?** SIMON was designed by the NSA specifically for
resource-constrained hardware. Its round function uses only AND, XOR, and circular
rotations — making it an ideal fit for bitsliced evaluation. In bitsliced mode,
rotations become free array-index permutations, and the remaining AND/XOR operations
process all plaintexts simultaneously.

**Bitsliced cryptography** (Biham, 1997) transposes the cipher state into bit-planes.
Bit k of each state word corresponds to a different plaintext. All operations are
bitwise, so N plaintexts encrypt simultaneously. SIMON's minimal gate count (16 AND +
48 XOR per round) makes it particularly efficient in this model.

SIMON 32/64 uses a 32-bit block (two 16-bit halves) with a 64-bit key and 32 rounds.
The total operation count is 32 rounds x 64 ops/round = 2048 bitwise operations.

---

## Dataset

All data is **synthetic**. Plaintexts and keys are generated programmatically:

- **SIMON paper test vector**: Key=`1918 1110 0908 0100`, PT=`6565 6877`,
  CT=`c69b e9bb` — verified against the published specification (Beaulieu et al., 2013,
  Appendix B, "SIMON Test Vectors")
- **Zero key/plaintext**: All-zero key and plaintext
- **All-ones key/plaintext**: All-ones key and plaintext
- **Multi-block batch**: 8 random plaintexts with a random key
- **Random batches**: 50 x 16-block and 20 x 64-block batches, fully verified against
  scalar reference
- **Large-scale**: 1M random plaintexts (1M-bit wide streams), spot-checked against
  reference

The paper test vector provides external validation independent of our reference
implementation.

## Datasets

Three tiers of benchmark data (tracked by `tests.json`; the `.bsdata`
files are generated on demand with `generate_tests.py --tier <t>`, not
committed):

| Tier | File | bitlength | Stream Data |
|------|------|-----------|-------------|
| small | simon_small.bsdata | 6.25M blocks | ~50 MB |
| medium | simon_medium.bsdata | 62.5M blocks | ~500 MB |
| large | simon_large.bsdata | 625M blocks | ~5 GB |

Each test feeds 544 streams to the program: 32 plaintext streams (`plainL[16]` +
`plainR[16]`) plus 512 expanded round-key streams (32 rounds x 16 bits).

**Source:** Synthetic random plaintext streams, fixed paper key. Tier expected
output is produced by the Python interpreter and spot-checked at 500 random
positions against the scalar reference (see provenance in `tests.json`).
**Generation:** `python datasets/tests/generate_tests.py --tier {small,medium,large}`

---

## Tutorial

### How SIMON works

SIMON is a balanced Feistel cipher. The 32-bit block is split into two 16-bit halves
(L, R). Each round applies:

```
new_L = R ^ f(L) ^ round_key[r]
new_R = L
```

where the round function is:

```
f(x) = (x<<<1 & x<<<8) ^ x<<<2
```

The `<<<` notation denotes left circular rotation (within 16 bits).

### Key schedule

SIMON 32/64 expands a 64-bit key (four 16-bit words) into 32 round keys using:

```
tmp = k[i-1] >>> 3           // right rotate by 3
tmp = tmp ^ k[i-3]
tmp = tmp ^ (tmp >>> 1)      // XOR with right rotate by 1
k[i] = k[i-4] ^ tmp ^ 0xFFFC ^ z[(i-4) % 62]
```

where `z` is a fixed sequence from the specification and `0xFFFC = 2^n - 4`.

### How bitstreams parallelize this

In the bitstream model, each of the 16 state bits is a stream. Bit position k of
stream `L[b]` holds bit b of plaintext k. When we compute `f[i] = L[15] & L[8]`, this
ANDs the corresponding rotated bits for ALL K plaintexts simultaneously.

**Rotations are free.** In the standard algorithm, `x<<<1` shifts and wraps bits within
a 16-bit word. In bitsliced mode, rotation by d means: bit i reads from
`L[(i - d) mod 16]`. This is just an array index computation — no bitwise operation
needed. The only actual operations are AND and XOR.

Per-round cost:
- 16 AND operations (for `rot1 & rot8`)
- 48 XOR operations (16 for `^ rot2`, 16 for `R ^ f`, 16 for `^ round_key`)
- Total: **64 bitwise ops per round**, 32 rounds = **2048 ops total**

### Using the program from Python

```python
from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter
from benchmark.apps.simon.src.run import (
    simon_key_schedule, bitslice_16bit, encode_round_keys, decode_16bit
)

with open("benchmark/apps/simon/src/simon.bs") as f:
    program = parse(f.read())

# Paper test vector
key = [0x0100, 0x0908, 0x1110, 0x1918]
round_keys = simon_key_schedule(key)

plainL = bitslice_16bit([0x6565])
plainR = bitslice_16bit([0x6877])
rk_arrays = encode_round_keys(round_keys)

interp = Interpreter()
result = interp.run(program, inputs={}, params={},
                    input_arrays={"plainL": plainL, "plainR": plainR,
                                  "round_key": rk_arrays})

cipherL = decode_16bit(result, "cipherL", 1)
cipherR = decode_16bit(result, "cipherR", 1)
print(f"Ciphertext: {cipherL[0]:04x} {cipherR[0]:04x}")  # c69b e9bb
```

---

## .bs Program Details

The complete program is in `simon.bs` (132 lines). It has four sections:

### 1. Declarations

```
input stream plainL[16]          // left half of plaintext (16 bit-planes)
input stream plainR[16]          // right half of plaintext (16 bit-planes)
input stream round_key[512]      // 32 round keys x 16 bits = 512 broadcast streams

output stream cipherL[16]        // left half of ciphertext
output stream cipherR[16]        // right half of ciphertext
```

### 2. Initialization

Copies plaintext into working arrays `L[16]` and `R[16]`.

### 3. Round function (32 rounds)

The round function `f(L) = (L<<<1 & L<<<8) ^ L<<<2` is unrolled for all 16 bit
positions to handle wrap-around explicitly:

```
// i=0: rot1=L[15], rot8=L[8], rot2=L[14]
stream _and_0 = L[15] & L[8]
f[0] = _and_0 ^ L[14]

// i=1: rot1=L[0], rot8=L[9], rot2=L[15]
stream _and_1 = L[0] & L[9]
f[1] = _and_1 ^ L[15]
// ... (16 positions total)
```

Then the Feistel step:
```
for i in 0..16 {
    stream _xor_rf = R[i] ^ f[i]
    newL[i] = _xor_rf ^ round_key[r * 16 + i]
}
```

### 4. Output

Copies final L and R to output arrays.

### Operation count

- **Per round**: 16 AND + 48 XOR = 64 bitwise operations
- **32 rounds**: 2048 total bitwise operations
- Array copies for rotation and Feistel swap are free

### DSL features used

| Feature | Used? | Notes |
|---------|-------|-------|
| AND (`&`) | Yes | 16 per round (round function) |
| XOR (`^`) | Yes | 48 per round (round function + Feistel + key addition) |
| Stream arrays | Yes | `L[16]`, `R[16]`, `round_key[512]`, `f[16]` |
| For loops | Yes | Outer loop over 32 rounds, inner over 16 bits |
| Nested for | Yes | Round loop contains bit loop |
| ONES | No | (round keys are external inputs) |
| OR | No | |
| NOT | No | |
| SHIFT | No | (rotations are array index permutations) |
| ADD (+) | No | |
| Popcount | No | |
| Parameters | No | (no runtime parameters — 32 rounds is fixed) |

---

## Test Suite (10 entries)

Unit tests (exhaustively verified against the `simon_encrypt()` scalar reference
in `run.py`):

| Test | Blocks | What It Checks |
|------|--------|----------------|
| SIMON paper test vector | 1 | Published ciphertext from Beaulieu et al. 2013 |
| Zero key and plaintext | 1 | All-zero inputs |
| All-ones key and plaintext | 1 | All-ones inputs |
| Random 8-block batch | 8 | Multiple plaintexts simultaneously |
| Random 16-block x50 | 800 (50 cases × 16) | 50 random keys, 16 blocks each |
| Random 64-block x20 | 1,280 (20 cases × 64) | 20 random keys, 64 blocks each |

Generated and tier tests (spot-checked at random positions rather than
verified exhaustively). The tier `.bsdata` carry expected output produced by
the Python interpreter; the "Large 1M blocks" test stores no expected output
and is generated on the fly, with the backend result spot-checked against the
`simon_encrypt()` scalar reference:

| Test | Blocks | What It Checks |
|------|--------|----------------|
| Large 1M blocks | 1,000,000 | 1M-bit wide streams, on-the-fly vs scalar reference |
| Small 6.25M blocks | 6,250,000 | tier-small bsdata, spot-checked at 500 positions |
| Medium 62.5M blocks | 62,500,000 | tier-medium bsdata, spot-checked at 500 positions |
| Large 625M blocks | 625,000,000 | tier-large bsdata, spot-checked at 500 positions |

The paper test vector additionally validates against the published specification.

---

## Running

```bash
# Standalone runner (detailed output)
python3 benchmark/apps/simon/src/run.py

# Via unified framework
python3 benchmark/run_all.py simon

# Run all domains together
python3 benchmark/run_all.py
```

### Relevant source files

| File | Purpose |
|------|---------|
| `benchmark/apps/simon/src/simon.bs` | SIMON 32/64 bitsliced cipher (132 lines, 2048 ops) |
| `benchmark/apps/simon/src/run.py` | Standalone test harness with scalar reference + key schedule |
| `benchmark/apps/simon/datasets/tests/generate_tests.py` | Generates precomputed test data |

---

## References

- Beaulieu, R., Shors, D., Smith, J., Treatman-Clark, S., Weeks, B., and Wingers, L.
  "The SIMON and SPECK Families of Lightweight Block Ciphers." *IACR ePrint* 2013/404,
  2013. The original SIMON specification.

- Biham, E. "A Fast New DES Implementation in Software." *FSE 1997*. Introduced
  bitslicing for symmetric cryptography.

- Beaulieu et al. "SIMON and SPECK: Block Ciphers for the Internet of Things."
  *NIST Lightweight Cryptography Workshop*, 2015. Discussion of IoT applications.
