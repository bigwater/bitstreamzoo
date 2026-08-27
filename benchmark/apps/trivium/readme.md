# Trivium Stream Cipher — Bitstream Benchmark

## Overview

This benchmark implements the Trivium stream cipher using the bitstream DSL. Each bit
position is a different (key, IV) pair -- the cipher encrypts multiple key/IV pairs
simultaneously. The 288-bit state is stored as a stream array `s[288]`.

Trivium is a hardware-oriented stream cipher by Christophe De Canniere and Bart
Preneel, submitted to the eSTREAM competition in 2005. After three years of public
evaluation, it was selected in 2008 for the **eSTREAM portfolio, Profile 2 (Hardware)**,
judged among the best designs for efficient hardware implementation by the ECRYPT
Stream Cipher Project.

> De Canniere, C. and Preneel, B. "Trivium." In *New Stream Cipher Designs*,
> LNCS 4986, pp. 244--266, Springer, 2008. DOI: 10.1007/978-3-540-68351-3_18.
> Selected for the eSTREAM Portfolio, Profile 2 (Hardware), in 2008.

**Why Trivium for bitstreams?** Trivium's design goal was minimal gate count. Each
clock tick requires only 3 AND + 9 XOR = 12 bitwise operations (warmup) or
3 AND + 11 XOR = 14 operations (generation, 2 extra XOR for keystream output). The shift register
updates are array copies (free in our op model). This makes Trivium the simplest
realistic stream cipher for bitsliced evaluation.

**Bitsliced cryptography** is a well-known technique (Biham, 1997) where the cipher
state is transposed into bit-planes. Bit k of each state word corresponds to a
different key. All operations are bitwise, so N keys encrypt simultaneously. Our
bitstream DSL formalizes this approach.

Instead of encrypting one (key, IV) pair at a time, the bitstream version encrypts
**all pairs simultaneously** in a single execution. Each bit position in a stream
represents a different (key, IV) pair, so a 64-bit word on a GPU encrypts 64
independent keystreams per clock cycle. A 1M-bit stream encrypts one million keystreams
at once.

---

## Dataset

All data is **synthetic**. Key and IV values are generated programmatically:

- **Single-pair tests**: All-zero key/IV, single key bit set, single IV bit set
- **Multi-pair test**: 3 distinct (key, IV) pairs encrypted simultaneously
- **External ground truth**: All-zero key/IV verified against the published ECRYPT test
  vector (first 64 keystream bits: `FBE0BF265859051B`)
- **Large-scale**: 1M random (key, IV) pairs (1M-bit wide streams), with 200 pairs
  spot-checked against the scalar reference

No external datasets are required. The ECRYPT test vector provides external validation
independent of our own reference implementation.

## Datasets

Three tiers of benchmark data (tracked by `tests.json`; the `.bsdata`
files are generated on demand with `generate_tests.py --tier <t>`, not
committed):

| Tier | File | bitlength | Stream Data |
|------|------|-----------|-------------|
| small | trivium_tier_small.bsdata | 1.79M pairs | ~50 MB |
| medium | trivium_tier_medium.bsdata | 17.9M pairs | ~500 MB |
| large | trivium_tier_large.bsdata | 179M pairs | ~5 GB |

Each tier stores 224 binary streams: 80 key input streams (`key[80]`) + 80 IV
input streams (`iv[80]`) + 64 keystream output streams (`z`, L=64).

**Source:** Synthetic random key/IV pairs (seeds 1000/1001/1002). Tier expected
output is produced by the Python interpreter and spot-checked at 20 random
positions against the scalar reference (see provenance in `tests.json`).
**Generation:** `python datasets/tests/generate_tests.py --tier {small,medium,large}`

---

## Tutorial

### How Trivium works

Trivium is a stream cipher -- it takes a secret key and a public IV (initialization
vector) and produces an arbitrarily long keystream. XOR the keystream with plaintext to
encrypt, XOR again to decrypt.

### State structure

Trivium's state is 288 bits organized as three coupled shift registers:

```
Register 1: s[0] --- s[1] --- ... --- s[92]    (93 bits)
Register 2: s[93] -- s[94] -- ... --- s[176]   (84 bits)
Register 3: s[177] - s[178] - ... --- s[287]   (111 bits)
```

| Register | Indices | Length | Tap | Feedback tap | AND pair |
|----------|---------|--------|-----|-------------|----------|
| 1 | s[0..92] | 93 | s[65] | s[170] | s[90]*s[91] |
| 2 | s[93..176] | 84 | s[161] | s[263] | s[174]*s[175] |
| 3 | s[177..287] | 111 | s[242] | s[68] | s[285]*s[286] |

### Initialization

```
s[0..79]    = key[0..79]     // 80-bit key
s[80..92]   = 0
s[93..172]  = iv[0..79]      // 80-bit IV
s[173..284] = 0
s[285..287] = 1              // three bits set at end
```

Then 1152 "blank" clocks (4 x 288) to mix the state before generating output.

### Per-clock update

Each clock cycle executes the following operations to produce one keystream bit and
update the state:

```
t1 = s[65] ^ s[92]
t2 = s[161] ^ s[176]
t3 = s[242] ^ s[287]

z  = t1 ^ t2 ^ t3              // output keystream bit

t1 = t1 ^ (s[90] & s[91]) ^ s[170]
t2 = t2 ^ (s[174] & s[175]) ^ s[263]
t3 = t3 ^ (s[285] & s[286]) ^ s[68]

// Shift all three registers by one position, then feed back:
//   s[0]   <-- t3
//   s[93]  <-- t1
//   s[177] <-- t2
```

The cost per clock is **12 bitwise operations** during warmup (3 AND + 9 XOR) or
**14** during generation (3 AND + 11 XOR, 2 extra XOR for keystream output), plus 288
state assignments per clock: 285 shift copies (92 + 83 + 110 for the three
registers) and 3 feedback writes (s[0], s[93], s[177]). The state assignments are
"free" in the operation count because they are simple data movement, not computation.

The three registers are coupled: each register's feedback is shifted into the next
register, and mixes in one tap read from that same next register. Register 1's feedback
`t1` reads `s[170]` from register 2 and is shifted into register 2 (`s[93] <- t1`);
register 2's feedback `t2` reads `s[263]` from register 3 and is shifted into register 3
(`s[177] <- t2`); register 3's feedback `t3` reads `s[68]` from register 1 and is shifted
into register 1 (`s[0] <- t3`). This cross-coupling is what makes Trivium's state mixing
effective despite using only AND and XOR gates.

### How bitstreams parallelize this

In the bitstream model, each bit position is a different (key, IV) pair. When we
compute `t1 = s[65] ^ s[92]`, this XORs the corresponding state bits for ALL key/IV
pairs simultaneously.

With K pairs packed into bitstreams, one Trivium clock costs:
- 12 bitwise operations per warmup clock, 14 per generation clock (each processing K pairs at once)
- 288 state assignments (285 shift copies + 3 feedback writes, free)

Total for the full cipher (1152 warmup + L generation):
- (1152 x 12 + L x 14) bitwise ops, independent of K

The adaptation follows the **bitsliced register file** pattern. Instead of storing each
of the 288 state bits as a single 0 or 1, each state bit becomes a stream:
`stream s[288]`. Bit position `j` of stream `s[i]` holds the value of state bit `i`
for key/IV pair `j`.

### Using the program from Python

```python
from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter

# Load and parse the program
with open("benchmark/apps/trivium/src/trivium.bs") as f:
    program = parse(f.read())

# Prepare a single (key, IV) pair: all zeros
key_arrays = {i: 0 for i in range(80)}
iv_arrays = {i: 0 for i in range(80)}

# Run: generate 64 keystream bits
interp = Interpreter()
result = interp.run(
    program,
    inputs={},
    params={"L": 64},
    input_arrays={"key": key_arrays, "iv": iv_arrays},
)

# Extract the keystream for pair 0
z = result["z"]
keystream_bits = [(z.get(k, 0) >> 0) & 1 for k in range(64)]

print(f"First 8 keystream bits: {keystream_bits[:8]}")
print(f"Total bitwise operations: {interp.op_count}")
```

To encrypt multiple pairs simultaneously, set the appropriate bit positions in each
key/IV stream:

```python
key_arrays = {i: 0 for i in range(80)}
key_arrays[0] = 0b10    # bit 1 is set -> pair 1 has key[0]=1

iv_arrays = {i: 0 for i in range(80)}

# Both pairs run in a single execution
result = interp.run(program, inputs={},
                    params={"L": 64},
                    input_arrays={"key": key_arrays, "iv": iv_arrays})

# Extract pair 0's keystream (bit 0 of each output stream)
ks0 = [(result["z"].get(k, 0) >> 0) & 1 for k in range(64)]

# Extract pair 1's keystream (bit 1 of each output stream)
ks1 = [(result["z"].get(k, 0) >> 1) & 1 for k in range(64)]
```

---

## .bs Program Details

The complete program is in `src/trivium.bs` (122 lines). It has four sections:

### 1. Declarations

```
param int L                  // number of keystream bits to generate

input stream key[80]         // 80-bit key (array of streams)
input stream iv[80]          // 80-bit IV (array of streams)

output stream z[L]           // output keystream (L bits)

stream s[288]                // 288-bit internal state
```

The parameter `L` controls keystream length. It is supplied at runtime.

### 2. Initialization

```
for i in 0..80 {
    s[i] = key[i]
}
for i in 0..80 {
    s[93 + i] = iv[i]
}
s[285] = ONES
s[286] = ONES
s[287] = ONES
```

Uninitialized stream array elements default to `ZERO`, so s[80..92], s[173..176], and
s[177..284] are all zero without explicit assignment.

### 3. Warmup (1152 blank clocks)

The warmup loop runs the same per-clock update as generation but discards the output:

```
for k in 0..1152 {
    stream t1 = s[65] ^ s[92]
    stream t2 = s[161] ^ s[176]
    stream t3 = s[242] ^ s[287]

    t1 = t1 ^ (s[90] & s[91]) ^ s[170]
    t2 = t2 ^ (s[174] & s[175]) ^ s[263]
    t3 = t3 ^ (s[285] & s[286]) ^ s[68]

    for j in 0..92  { s[92 - j]  = s[91 - j]  }    // shift register 1
    s[0] = t3
    for j in 0..83  { s[176 - j] = s[175 - j] }    // shift register 2
    s[93] = t1
    for j in 0..110 { s[287 - j] = s[286 - j] }    // shift register 3
    s[177] = t2
}
```

### 4. Generation (L keystream clocks)

Identical to warmup, except the output keystream bit is stored:

```
for k in 0..L {
    stream t1 = s[65] ^ s[92]
    stream t2 = s[161] ^ s[176]
    stream t3 = s[242] ^ s[287]

    z[k] = t1 ^ t2 ^ t3            // output keystream bit k

    t1 = t1 ^ (s[90] & s[91]) ^ s[170]
    t2 = t2 ^ (s[174] & s[175]) ^ s[263]
    t3 = t3 ^ (s[285] & s[286]) ^ s[68]

    for j in 0..92  { s[92 - j]  = s[91 - j]  }
    s[0] = t3
    for j in 0..83  { s[176 - j] = s[175 - j] }
    s[93] = t1
    for j in 0..110 { s[287 - j] = s[286 - j] }
    s[177] = t2
}
```

### Operation count

- **8 top-level statements** (1 internal state declaration, 2 init for-loops, 3
  initialization assignments, the warmup for-loop, and the generation for-loop)
- **13,824 warmup ops** (1152 clocks x 12 ops) + **14L generation ops**
  (e.g. 15,616 unrolled ops at dependency depth 45 for L=128)
- **288 state assignments per clock** (285 shift copies + 3 feedback writes, free in op count)

### DSL features used

| Feature | Used? | Notes |
|---------|-------|-------|
| XOR (`^`) | Yes | The primary operation (9 per warmup clock, 11 per generation clock) |
| AND (`&`) | Yes | 3 AND gates per clock (s[90]&s[91], etc.) |
| Stream arrays | Yes | `stream s[288]`, `input stream key[80]` |
| For loops | Yes | Clocking (1152 warmup, L generation) and shifting |
| Parameters | Yes | `param int L` -- runtime keystream length |
| ONES | Yes | Initialize s[285..287] |
| OR | No | |
| NOT | No | |
| SHIFT | No | |
| ADD (+) | No | |
| Popcount | No | |

---

## Test Suite (10 tests)

Unit tests (six precomputed `.bsdata` cases):

| Test | Key/IV Pairs | Keystream Length | What It Checks |
|------|-------------|------------------|----------------|
| Zero key/IV 64b | 1 | 64 bits | All-zero key and IV against Python reference |
| Key index 79 64b | 1 | 64 bits | Synthetic single-bit key (hex `80...00` parsed as raw integer, sets key bit-list index 79); not a canonical ECRYPT vector |
| IV index 79 64b | 1 | 64 bits | Synthetic single-bit IV (hex `80...00` parsed as raw integer, sets IV bit-list index 79); not a canonical ECRYPT vector |
| 3 pairs x 64b | 3 | 64 bits | Multiple pairs simultaneously (tests parallelism) |
| Zero key/IV 128b | 1 | 128 bits | Longer keystream generation |
| ECRYPT vector | 1 | 64 bits | External ground truth: Key=0, IV=0 -> `FBE0BF265859051B` |

Generated and tier tests (regenerated by `generate_tests.py`):

| Test | Key/IV Pairs | Keystream Length | What It Checks |
|------|-------------|------------------|----------------|
| Large 1M pairs x 64b | 1,000,000 | 64 bits | 1M-bit wide streams, spot-checked |
| Tier small | 1,790,000 | 64 bits | Synthetic random pairs, seed 1000 |
| Tier medium | 17,900,000 | 64 bits | Synthetic random pairs, seed 1001 |
| Tier large | 179,000,000 | 64 bits | Synthetic random pairs, seed 1002 |

All tests except the ECRYPT vector test verify the bitstream output against a Python
reference implementation of Trivium (`trivium_reference()` in `run.py`). The ECRYPT
vector test additionally validates against the published test vector, providing
verification against a completely external source.

The large-scale test generates 80 key streams and 80 IV streams as random 1M-bit
integers using `getrandbits(W)`, runs the bitstream program once on all 1M pairs, then
spot-checks 200 randomly selected pairs against the scalar Python reference.

---

## Running

```bash
# Standalone runner (detailed output)
python3 benchmark/apps/trivium/src/run.py

# Via unified framework
python3 benchmark/run_all.py trivium

# Run all domains together
python3 benchmark/run_all.py
```

### Relevant source files

| File | Purpose |
|------|---------|
| `benchmark/apps/trivium/src/trivium.bs` | The Trivium bitstream program (122 lines, 12-14 ops/clock) |
| `benchmark/apps/trivium/src/run.py` | Standalone test harness with Python reference implementation |
| `benchmark/run_all.py` | Unified runner for all benchmark domains |

---

## References

- De Canniere & Preneel, "Trivium", in *New Stream Cipher Designs*, LNCS 4986,
  pp. 244--266, Springer, 2008 (DOI 10.1007/978-3-540-68351-3_18). The cipher
  specification chapter of the eSTREAM final report.

- ECRYPT Stream Cipher Project test vectors. External ground truth for validation.

- Biham, "A Fast New DES Implementation in Software", FSE 1997. Introduced bitslicing
  for symmetric cryptography.
