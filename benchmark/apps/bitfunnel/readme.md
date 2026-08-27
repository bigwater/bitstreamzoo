# BitFunnel — Bit-Sliced Signature Row Intersection

## Overview

The BitFunnel benchmark implements the **core query matching kernel** of the BitFunnel
search index (Goodwin et al., SIGIR 2017) using bitstream programs. Given a conjunctive
query and a bit-sliced Bloom filter index, the program identifies the set of candidate
matching documents by AND-intersecting the index rows that the query hashes to.

BitFunnel was deployed in the Microsoft Bing search engine, replacing an earlier
inverted-index system and achieving a 10× improvement in server query capacity. The
canonical evaluation corpus is TREC Gov2 (§6); bit density d = 0.15 yields the best
DQ (queries-per-second × documents-per-bit) for shards B, C, and D (§6.2;
Table 2 examines shard D).

This benchmark is the primary representative of the **broadcast-expand-and-intersect**
pattern in the suite: higher-rank rows are expanded to rank-0 equivalents via shift and
OR before being ANDed into the accumulator. It is the only benchmark that combines the
rank-expansion idiom (`row | (row << k)`) with a multi-way AND reduction and popcount.

---

## Dataset

### Original corpora

The BitFunnel paper (§6) evaluated on **TREC Gov2** (shard D: 494K documents,
1024–2047 terms/doc, ~98K TREC 2006 queries; target index density d = 0.15 from
Table 2's BTFNL configuration is a tunable parameter, not a corpus property).
Gov2 is not publicly redistributable; the corpus is distributed by the University
of Glasgow under an individual/organizational agreement and fee.

The [GitHub repository](https://github.com/BitFunnel/BitFunnel) ships only the
154 Shakespeare sonnets (hard-coded in `src/Data/src/Sonnets.cpp`) as embedded
test data. A Wikipedia-based test corpus is hosted externally on
[bitfunnel.org](https://bitfunnel.org/wikipedia-as-test-corpus-for-bitfunnel/);
the project's [index-build walkthrough](https://bitfunnel.org/index-build-tools/)
uses a ~17.6K-document Wikipedia sample to build the index.

Neither corpus is directly usable as bitstream rows: the BitFunnel index builder
must be run first to produce the bit-sliced row data from raw documents.

### Benchmark data

For the bitstream benchmark, rows are generated synthetically; no external corpus
is required. N is the stream length (one bit per document). Rows are uniform random
bits (density ≈ 0.5 for rank-0; rank-1/rank-2 rows occupy the lower 1/2 and 1/4 of
the stream respectively), seeded for reproducibility.  K0=3, K1=3, K2=2 for all tiers.

| Tier | N (documents) | Approx total data |
|------|--------------|-------------------|
| small | 50M | ~56 MB |
| medium | 500M | ~563 MB |
| large | 5B | ~5.6 GB |

**Generation:** `python datasets/tests/generate_tests.py --tier small`

---

## Algorithm

Each document is encoded as a Bloom filter signature stored in a bit-sliced layout:
`row[i]` is a bitvector where bit k = 1 iff document k has bit i set in its signature.
A query AND-intersects the rows it hashes to; remaining set bits are candidate matches.

**Higher Rank Rows (§4.1):** a rank-r row has blocking factor 2^r (each bit covers 2^r
documents), making it 2^r times shorter and faster to scan. To intersect rows of
different ranks, each rank-r row is expanded to its rank-0 equivalent by concatenating
2^r copies (Figure 6). In stream terms:

```
rank-1 equivalent:  row | (row << HALF)                        HALF = N/2
rank-2 equivalent:  (row | row << Q) | ((row | row << Q) << HALF)  Q = N/4
```

The trade-off is controlled noise (false positives), which the paper's cost model bounds.

This benchmark models the logical rank-0-equivalent intersection the paper defines
(§4.1.1, "concatenating 2^r copies"). It does not model BitFunnel's memory
optimization of scanning physically shorter rank-r rows (§4.1.2, §6.2: processing a
rank-r row scans 1/2^r of the words a rank-0 row needs), because the bitstream DSL
gives every stream a single fixed length N.

---

## .bs Program Details

The complete program lives at `src/bitfunnel.bs`.

### Declarations

```
param int K0       // rank-0 query rows  (blocking factor 1)
param int K1       // rank-1 query rows  (blocking factor 2)
param int K2       // rank-2 query rows  (blocking factor 4)
param int HALF     // N/2 — expansion shift for rank-1 rows
param int QUARTER  // N/4 — expansion shift for rank-2 rows

input stream rows_r0[K0]
input stream rows_r1[K1]
input stream rows_r2[K2]

output stream matches
output int count
```

`HALF` and `QUARTER` are passed as `param` because the DSL has no `length()`
primitive — the caller derives them from the stream length N.

### Operation count

Total: **K0 + 3·K1 + 5·K2 + 1** (the +1 is `popcount`).

| Query profile | K0 | K1 | K2 | Ops |
|---------------|----|----|-----|-----|
| 3/3/1 (illustrative) | 3 | 3 | 1 | 18 |
| Tier profile (used by `generate_tests.py`) | 3 | 3 | 2 | 23 |
| Rare-term heavy (illustrative) | 2 | 4 | 3 | 30 |
| Rank-0 only (illustrative) | 6 | 0 | 0 | 7 |

### DSL features used

| Feature | Used? | Notes |
|---------|-------|-------|
| `&` (AND) | Yes | Accumulator intersection |
| `\|` (OR) | Yes | Rank-r row expansion |
| `<<` (shift) | Yes | Broadcast to upper half |
| `popcount` | Yes | Final candidate count |
| `for` loop | Yes | Iterate over K rows per rank |
| Stream arrays | Yes | `rows_r0[K0]`, `rows_r1[K1]`, `rows_r2[K2]` |
| Parameters | Yes | K0, K1, K2, HALF, QUARTER |
| `output int` | Yes | Candidate count |
| ONES | Yes | Accumulator initialization |
| `if`/`while` | No | Early termination not expressible in the DSL |

The absence of `while` (early termination) is a meaningful limitation: the BitFunnel
paper (§5.3) derives significant speedup from breaking out of the inner loop when the
accumulator reaches zero. This optimization requires a data-dependent loop bound with
a data-driven array index, which the DSL does not support.

---

## Running

```bash
python benchmark/run_all.py bitfunnel               # Python backend
python benchmark/run_all.py bitfunnel --backend cpp  # C++ backend
python benchmark/run_all.py bitfunnel --backend cuda # CUDA backend
```

> The three tier datasets (`datasets/tests/bitfunnel_tier_{small,medium,large}.bsdata`)
> are gitignored due to size and generated locally; only the small unit-test
> `.bsdata` files are committed to the repo.
> Generate with `python datasets/tests/generate_tests.py --tier small` (or `medium`, `large`).

### Test suite

| # | Test | N | K0/K1/K2 | What it tests |
|---|------|---|----------|---------------|
| 1 | All-ones rows | 8 | 2/0/0 | acc stays ONES, count = N |
| 2 | One zero row | 8 | 2/0/0 | Single zero row clears all bits |
| 3 | No overlap | 8 | 2/0/0 | Disjoint rows → count = 0 |
| 4 | Rank-1 expand | 8 | 0/1/0 | Expansion correctness, HALF=4 |
| 5 | Rank-2 expand | 8 | 0/0/1 | 4-copy broadcast, QUARTER=2 |
| 6 | Mixed ranks | 16 | 1/1/1 | All three ranks together |
| 7 | Random N=1000 | 1K | 3/3/1 | Random rows, verified vs scalar |
| 8 | Large N=65536 | 64K | 3/3/2 | Scale test, verified vs scalar |

---

## References

- Goodwin, B. et al., "BitFunnel: Revisiting Signatures for Search",
  *Proceedings of the 40th International ACM SIGIR Conference on Research and
  Development in Information Retrieval*, 2017.
- [github.com/BitFunnel/BitFunnel](https://github.com/BitFunnel/BitFunnel) —
  reference C++14 implementation,
  `src/Plan/src/RankDownCompiler.cpp`, `RankZeroCompiler.cpp`
