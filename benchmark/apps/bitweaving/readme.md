# BitWeaving-V — Database Column Scan Benchmark

## Overview

This benchmark implements a **BitWeaving-style vertical bit-plane range scan**
using the bitstream DSL. It evaluates the query:

```sql
SELECT count(*) FROM lineitem WHERE lo <= L_QUANTITY <= hi
```

A column of N unsigned B-bit integers is stored vertically as B bit-planes (one
stream per bit position). Two comparisons (val >= lo, val <= hi) are computed
simultaneously in a single MSB-to-LSB pass using accumulator streams.

Specifically, this is the fixed-depth VBP column-scalar comparison
(Algorithm 2 of Li & Patel 2013); the full BitWeaving/V variant
(Algorithm 3) adds early pruning and bit-group layout, which we omit so the
op count is a fixed function of B.

> Li, Y. and Patel, J. M. "BitWeaving: Fast Scans for Main Memory Data
> Processing." *Proceedings of the ACM SIGMOD International Conference on
> Management of Data*, 2013.

**Why BitWeaving for bitstreams?** BitWeaving-V stores column data as bit-planes,
which map directly to bitstreams. The comparison algorithm uses only AND, OR, XOR,
NOT, and popcount, all operating on entire bit-planes in parallel. It is a common
PIM database workload, evaluated by Ambit (MICRO'17) and SIMDRAM (ASPLOS'21).

**Bitstream encoding:** Each bit-plane `data[b]` contains bit b of all N row values.
Bound parameters (lo, hi) are encoded as B broadcast streams (ONES if the
corresponding bit is 1, ZERO otherwise). A `valid` mask stream marks which bit
positions correspond to actual rows.

---

## Dataset

**Synthetic** (unit tests): hand-crafted values and uniform random integers in
[1, 50] (matching the TPC-H `L_QUANTITY` value range). Predicate [1, 23] uses
the TPC-H Q6 validation value `QUANTITY = 24` (i.e., `l_quantity < 24`); the
parameterized form of Q6 selects `QUANTITY` from `[24..25]`.

**Synthetic** (tier data): uniform random bit-planes; the algorithm
performs identical operations regardless of data values.

| Tier | Rows | Size | Description |
|------|------|------|-------------|
| small | ~40M | ~50MB | Uniform random 8-bit column, B=8 |
| medium | ~400M | ~500MB | Same, larger |
| large | ~4B | ~5GB | Same, HPC only |

---

## Algorithm

### BitWeaving-V Range Comparison

To evaluate `lo <= val <= hi` on B-bit unsigned integers stored as bit-planes:

1. **Initialize** two accumulator pairs:
   - `lt = ZERO, eq_lo = ONES` (for val >= lo)
   - `gt = ZERO, eq_hi = ONES` (for val <= hi)

2. **Scan MSB to LSB** (for each bit position b from B-1 down to 0):
   - **val >= lo**: If data bit = 0 and lo bit = 1 and still equal, mark as
     "definitely less than lo". Update equality.
   - **val <= hi**: If data bit = 1 and hi bit = 0 and still equal, mark as
     "definitely greater than hi". Update equality.

3. **Combine**: `result = ~lt & ~gt & valid`
4. **Count**: `count = popcount(result)`

### Concrete trace (B=4, val=5=0101, lo=3=0011, hi=9=1001)

Scan MSB to LSB. Accumulators start `lt=0, eq_lo=ONES, gt=0, eq_hi=ONES`.
Per bit: `new_lt = eq_lo & ~val_b & lo_b` and `new_gt = eq_hi & val_b & ~hi_b`.
An equality accumulator clears whenever `val_b` differs from its bound bit.

```
b=3:  val=0 lo=0 hi=1 | new_lt=0           | new_gt=0  eq_hi->0
b=2:  val=1 lo=0 hi=0 | new_lt=0  eq_lo->0 | new_gt=0
b=1:  val=0 lo=1 hi=0 | new_lt=0           | new_gt=0
b=0:  val=1 lo=1 hi=1 | new_lt=0           | new_gt=0
```

Final: `lt=0` so `ge_lo=ONES`, `gt=0` so `le_hi=ONES`, and
`result = ge_lo & le_hi & valid = 1`. Correct: `3 <= 5 <= 9` holds.

---

## .bs Program Details

The program is in `src/bitweaving.bs` (71 lines). Key structure:

```
param int B
input stream data[B]       // bit-planes
input stream lo_bits[B]    // broadcast bound bits
input stream hi_bits[B]    // broadcast bound bits
input stream valid          // row validity mask
output stream result
output int count
```

### Operation count

Per loop iteration: 14 ops (7 for each comparison)
- XOR (diff detect), NOT, AND, AND (new_lt/gt), OR (accumulate), NOT, AND (eq update)

Total: 14B + 4 bitwise + 1 popcount
- B=4: 56 + 4 + 1 = 61 ops
- B=8: 112 + 4 + 1 = 117 ops
- B=16: 224 + 4 + 1 = 229 ops

### Critical-path depth

Measured on the unrolled dependency DAG:

| B | Ops | Depth (flat) | Depth (log popcount) |
|---|-----|--------------|----------------------|
| 4 | 61  | 11 | 18 |
| 8 | 117 | 15 | 22 |
| 16| 229 | 23 | 30 |

The MSB-to-LSB scan serializes the `eq_lo`/`eq_hi` chain, so depth grows
~linearly in B. The log popcount model adds a fixed
`1 + ceil(log2(64)) = 7` levels for the single final `popcount`,
modeling the per-64-bit-word reduction tree the C++/CUDA backend pays.
This cost is independent of the input length, so every row is exactly
flat depth + 7.

### DSL features used

| Feature | Used? | Notes |
|---------|-------|-------|
| AND (`&`) | Yes | Masking, accumulator update |
| OR (`\|`) | Yes | Accumulate definitely-less/greater |
| XOR (`^`) | Yes | Bit difference detection |
| NOT (`~`) | Yes | Complement for comparisons |
| For loop | Yes | MSB-to-LSB scan |
| Stream arrays | Yes | `data[B]`, `lo_bits[B]`, `hi_bits[B]` |
| Parameters | Yes | `B` (bits per value) |
| Popcount | Yes | Final count reduction |
| ZERO/ONES | Yes | Accumulator initialization |
| Shift | No | |
| Addition | No | |
| If/While | No | |

---

## Test Suite (12 unit tests + 3 tier datasets = 15 entries in `tests.json`)

| # | Test | N | B | What It Tests |
|---|------|---|---|---------------|
| 1-9 | Fixed B=4 cases | 4-5 | 4 | All match, no match, boundary, empty range, etc. |
| 10 | TPC-H Q6 small | 8 | 8 | Synthetic TPC-H-range values [1,50], range [1,23] |
| 11 | Random B=8 x50 | 64 | 8 | 50 random cases verified, 1 stored |
| 12 | Wide 100K TPC-H Q6 | 100K | 8 | Large-scale uniform `L_QUANTITY` distribution |
| 13-15 | Tier small/medium/large | 40M / 400M / 4B | 8 | Synthetic random bit-plane stress tiers |

---

## Running

```bash
python benchmark/run_all.py bitweaving                # Python backend
python benchmark/run_all.py bitweaving --backend cpp  # C++ backend
python benchmark/run_all.py bitweaving --backend cuda # CUDA backend
```

---

## References

- Li, Y. and Patel, J. M. "BitWeaving: Fast Scans for Main Memory Data
  Processing." SIGMOD, 2013.
- TPC-H Benchmark Specification v3.0.1.
  https://www.tpc.org/TPC_Documents_Current_Versions/pdf/TPC-H_v3.0.1.pdf
- Seshadri, V. et al. "Ambit: In-Memory Accelerator for Bulk Bitwise
  Operations Using Commodity DRAM Technology." MICRO, 2017.
  Section 8.2 (BitWeaving evaluation).
- Hajinazar, N. et al. "SIMDRAM: An End-to-End Framework for Bit-Serial
  SIMD Computing in DRAM." ASPLOS, 2021.
  Section 7.3 (BitWeaving workload). DOI: 10.1145/3445814.3446749.
