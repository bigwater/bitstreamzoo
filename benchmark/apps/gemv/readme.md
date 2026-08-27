# GEMV: Gate-decomposed Matrix-Vector Multiply

## Overview

Computes y = W * x where W is L x N and x is N x 1, with K-bit
precision per element.  All arithmetic (multiply and accumulate)
is decomposed into AND/XOR/OR gates.

This is the core computation of one DNN fully-connected layer:
L output neurons, each computing a dot product with N input features.
M independent instances (batch samples) are processed in parallel.

## Algorithm

For each output neuron l in 0..L-1:
  For each input feature f in 0..N-1:
    1. Bit-serial multiply: prod = x[f] * w[l][f]  (shift-and-add)
    2. Accumulate: y[l] += prod  (full-adder chain)

## Operation count

L * N * (6K^2 + 10K + 2(B-2K)), where B = 2K + ceil(log2(N)).

| Config (L, N, K) | Ops     | Notes |
|-------------------|---------|-------|
| L=1, N=4, K=2     | 192     | Single dot product |
| L=4, N=8, K=4     | 4,544   | Small layer |
| L=16, N=16, K=4   | 36,864  | Medium layer |
| L=8, N=16, K=8    | 60,416  | 8-bit precision |
| L=64, N=64, K=8   | ~2M     | Full hidden layer |

## Connection to hardware

**PIM (Ambit/SIMDRAM substrates; CHOPPER compiler):** The gate
decomposition is what bulk-bitwise PIM substrates (Ambit, SIMDRAM)
execute and what PIM compilers (CHOPPER) target — they have no
hardware adders.

**Stripes/BitL:** Share the bit-plane layout and AND partial products
but use hardware adder trees for accumulation.

## Closest reference

The closest published match to what this benchmark actually
implements is:

- Charles Eckert, Xiaowei Wang, Jingcheng Wang, Arun Subramaniyan,
  Ravi Iyer, Dennis Sylvester, David Blaauw, and Reetuparna Das,
  "Neural Cache: Bit-Serial In-Cache Acceleration of Deep Neural Networks,"
  ISCA 2018.

Why this is the closest:

- it uses transposed / bit-serial data layout
- it performs arithmetic directly as bit-serial operations
- it builds multiplication and accumulation from simple arithmetic primitives
  instead of relying on a conventional hardware MAC datapath

Why it is still not identical:

- Neural Cache is a hardware architecture in SRAM
- this benchmark is a handwritten software/DSL kernel
- this benchmark makes the schoolbook multiply and ripple-carry accumulation
  explicit as AND/XOR/OR chains

Stripes and CHOPPER are still useful context for the hardware motivation
above, but they are not the closest source for the implementation structure
of this benchmark.

## Tier configurations

Tier data varies matrix dimensions (L, N) and precision (K) to exercise
different computational profiles at scale:

| Config | L | N | K | B | IO streams | W streams | Total | Ops/vec | Use case |
|--------|---|---|---|---|------------|-----------|-------|---------|----------|
| `small_k2` | 16 | 16 | 2 | 8 | 160 | 512 | 672 | 13,312 | Small embedding, 2-bit |
| `medium_k4` | 64 | 64 | 4 | 14 | 1,152 | 16,384 | 17,536 | 606,208 | Hidden layer, INT4 |
| `large_k8` | 64 | 128 | 8 | 23 | 2,496 | 65,536 | 68,032 | 3,915,776 | Large hidden layer, INT8 |

Each config generates one .bsdata file per tier (small/medium/large), for
9 total tier files. Weights are random K-bit integers; inputs are random
bitstreams. Tier `.bsdata` files are generated directly by
`generate_tests.py --tier {small,medium,large,all}`.
