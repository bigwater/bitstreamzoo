# Polar Code Encoding

## Overview

Polar codes, introduced by Arikan (2009), are the first provably
capacity-achieving codes for symmetric binary-input memoryless channels.
They are used in the 5G NR standard (3GPP TS 38.212) for downlink
control information (DCI on PDCCH) and for longer uplink control
information (UCI) payloads; short UCI uses block coding instead.

The encoding operation computes `x = u * F^{otimes n}` where
`F = [[1,0],[1,1]]` and `n = log2(N)`. This produces a butterfly XOR
network: a recursive structure of n stages, each performing N/2
independent XOR operations.

This kernel uses the non-bit-reversed form `F^{otimes n}`, matching the
5G NR encoder (3GPP TS 38.212). Arikan's original generator additionally
applies a bit-reversal permutation (`G_N = B_N * F^{otimes n}`), which
differs only by a relabeling of input indices.

## Algorithm

The butterfly XOR network for code length N has `n = log2(N)` stages:

```
for s in 0..n-1:
    stride = 2^s
    for j in 0, 2*stride, 4*stride, ..., N-1:
        for i in 0..stride-1:
            x[j+i] ^= x[j+i+stride]
```

### Worked example (N=8)

```
Input:  u[0] u[1] u[2] u[3] u[4] u[5] u[6] u[7]

Stage 0 (stride=1): XOR adjacent pairs
  x[0]^=x[1]  x[2]^=x[3]  x[4]^=x[5]  x[6]^=x[7]

Stage 1 (stride=2): XOR pairs at distance 2
  x[0]^=x[2]  x[1]^=x[3]  x[4]^=x[6]  x[5]^=x[7]

Stage 2 (stride=4): XOR pairs at distance 4
  x[0]^=x[4]  x[1]^=x[5]  x[2]^=x[6]  x[3]^=x[7]

Output: x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7]
```

Total: 4 + 4 + 4 = 12 XOR ops = N/2 * log2(N).

## Bitstream Adaptation

Each bit position in a stream represents an independent message being
encoded in parallel. The N input streams carry u[0..N-1] and the N
output streams carry x[0..N-1]. All XOR operations are bitwise,
encoding thousands of messages simultaneously.

## Program Sizes

Three .bs programs are generated from `polar_gen.py`, varying code
length N:

| Program | N | Stages | XOR ops | Streams (in+out) |
|---|---|---|---|---|
| polar_small | 64 | 6 | 192 | 128 |
| polar_medium | 256 | 8 | 1024 | 512 |
| polar_large | 1024 | 10 | 5120 | 2048 |

## Scalability

Two independent axes of scaling:

1. **Code length N** (program structure): Determines the number of
   streams, operations, and stages. N is fixed per .bs file because
   the number of stages = log2(N) cannot be expressed as a dynamic
   loop bound. Different N values require different .bs files.

2. **bitlength** (data parallelism): How many independent messages
   are encoded in parallel. This is the bitstream width, controlled
   by tiers (small/medium/large). Scales freely.

### Complexity

- Operations: N/2 * log2(N) (all XOR)
- I/O streams: 2N (N input + N output)
- Live set: O(N); the PkLive metric counts peak live across DAG
  depth levels (see Characterization).
- ILP per stage: N/2 (all butterflies in a stage are independent)
- Critical path: n = log2(N) stages

## Characterization

| Property | polar_small | polar_medium | polar_large |
|---|---|---|---|
| Code length N | 64 | 256 | 1024 |
| XOR ops | 192 | 1024 | 5120 |
| Stages | 6 | 8 | 10 |
| ILP (per stage) | 32 | 128 | 512 |
| PkLive (DAG depth-based) | 119 | 501 | 2035 |
| I/O streams (in+out) | 128 | 512 | 2048 |
| Op mix | 100% XOR | 100% XOR | 100% XOR |
| Critical path | 6 | 8 | 10 |

PkLive is the peak number of simultaneously live bitstreams across
DAG depth levels.

Key properties:
- **100% XOR**: unique among all benchmarks (others mix AND/OR/XOR/NOT)
- **Butterfly DAG**: perfectly regular, recursive structure
- **High ILP**: N/2 independent ops per stage
- **O(N log N) scaling**: ops grow as N log N, liveness as O(N)

## Dataset

Polar messages are synthetic, so there is no `make_data.py` or raw
`.npz` tier tree (the same convention as BitFunnel, Bitweaving, and
Montgomery Mul). Canonical `.bsdata` is generated directly from the
`.bs` programs by `datasets/tests/generate_tests.py`. The small
unit-test `.bsdata` files are committed; the large tier datasets
(`polar_*_tier_{small,medium,large}.bsdata`) are gitignored and
generated locally with
`python datasets/tests/generate_tests.py --tier {small,medium,large}`.

## Test Suite

Unit tests per program variant:
- All-zero message (1 message)
- One-hot messages (N or N/16 messages)
- Random messages (64-128 messages, seed=42; plus 512 messages
  seed=123 for polar_medium)
- Wide generated test (10K-100K messages at runtime)

Tier tests use random N-bit messages generated as wide bitstreams.

## References

- Arikan, E. "Channel Polarization: A Method for Constructing
  Capacity-Achieving Codes for Symmetric Binary-Input Memoryless
  Channels." IEEE Transactions on Information Theory, vol. 55,
  no. 7, pp. 3051-3073, July 2009.
- 3GPP TS 38.212: "NR; Multiplexing and channel coding."
