# Binary Neural Network (BNN) Inference -- Bitstream Benchmark

## Overview

The BNN benchmark implements a **gate-decomposed BNN-layer stress kernel** as a
parameterized bitstream program, following FINN's MVTU. It is not native
binary-neural-network inference: the popcount reduction is a gate-level tree of
half/full adders rather than a hardware popcount or tensor-core path. It
demonstrates how the core computation of a binarized neural network -- XNOR
matching, gate-level adder-tree reduction, and threshold comparison -- maps to
bitwise operations on streams, classifying many input samples at once.

Binary Neural Networks (BNNs) are neural networks where both weights and activations
are constrained to single bits: +1 or -1, encoded as 1 and 0. This extreme
quantization replaces the expensive floating-point multiply-accumulate at the heart of
conventional neural networks with cheap bitwise operations. The dot product
`sum(w_i * x_i)` becomes `2 * popcount(XNOR(w, x)) - N`, where N is the input
dimension.

The idea was developed in parallel by two groups in 2016:

- **Hubara, Courbariaux et al.**, "Binarized Neural Networks", NeurIPS 2016 -- showed
  that networks with binary weights and activations can be trained end-to-end using
  straight-through estimators for the gradient. The "Running a BNN" procedure
  (Algorithm 4 in the NeurIPS version; Algorithm 5 in the arXiv preprint
  [arXiv:1602.02830]) specifies the inference pipeline as
  `a_k <- XnorDotProduct(a_{k-1}^b, W_k^b); a_k^b <- Sign(BatchNorm(a_k))`.
- **Rastegari et al.**, "XNOR-Net: ImageNet Classification Using Binary Convolutional
  Neural Networks", ECCV 2016 -- demonstrated that the dot product in a binary layer
  can be computed as an XNOR followed by a popcount.

The hardware side was addressed by:

- **Umuroglu et al.**, "FINN: A Framework for Fast, Scalable Binarized Neural Network
  Inference", FPGA 2017 -- built an FPGA accelerator for BNN inference using a
  Matrix-Vector Threshold Unit (MVTU) that realizes the same XNOR-popcount-threshold
  pipeline encoded in our `.bs` program. §4.2.1 "Popcount for Accumulation" reports
  that a 128-bit popcount-accumulate (Vivado HLS, 200 MHz) costs 376 LUTs and 29 FFs,
  roughly half the 759 LUTs and 84 FFs of signed accumulate. §4.2.2
  "Batchnorm-activation as Threshold" folds `Sign(BatchNorm(a, theta))` into a single
  integer-threshold comparison `popcount >= tau`.

This benchmark implements FINN's XNOR-popcount-threshold pipeline on bit-plane
streams: XNOR matching, half/full-adder tree-reduction popcount, and threshold
compare. The specific 16-input half/full-adder tree below is BitstreamZoo's
gate-level realization of FINN's popcount stage; FINN itself documents the
operation-level popcount-accumulate and its LUT/FF cost but leaves the internal
adder topology to Vivado HLS. The same gate-level form is required on any
bit-serial PIM substrate without hardware popcount. On commodity AVX-512 with
hardware `vpopcntq`, the gate-level tree form is one cost point for
emulating the hardware instruction with bit-plane operations.

---

## Dataset

The benchmark uses a combination of **synthetic data** and **real-world datasets**.

### Synthetic tests (6 tests)

All synthetic tests use the 3-layer `bnn_mnist.bs` program with small dimensions to
verify correctness of the full multi-layer pipeline.

- **Fixed test vectors**: 3 hand-crafted cases covering edge conditions (all match,
  no match, multi-neuron chain with mixed outcomes across layers).
- **Exhaustive N1=8**: All 256 possible 8-bit input patterns through a small 3-layer
  network. Exercises the `.bs` program's remainder-loop path (since 8 < 16 = CHUNK_SIZE,
  `N1_CHUNKS = 0` and the entire dot product is computed via the per-input ripple
  fallback).
- **Random test vectors**: 100 random 16-bit trials and 50 random 64-bit trials with
  random weights and thresholds for all 3 layers.

No external packages are required for the synthetic tests.

### MNIST classification (2 tests)

A 3-layer BNN classifier (784 -> 1024 -> 1024 -> 1024) for MNIST digit recognition
using pre-trained weights, implemented as a single `bnn_mnist.bs` program. This follows
the FINN LFC architecture (Umuroglu et al., FPGA 2017) which achieves ~98% on MNIST.
A 4th layer (1024 -> 10, argmax via XNOR-popcount without thresholding) is applied only
during test data generation to compute expected labels -- it is not part of the `.bs`
program.

- Pre-trained parameters: `datasets/small/mnist_params.npz`
- Training script: `datasets/raw/train_mnist.py`

Measured accuracy on the current pre-trained weights: **100% on MNIST 100**,
**97.91% on MNIST 10k** (both above the 95% / 97% pass thresholds).

### Two separate input paths: synthetic tiers vs. optional MNIST check

**Synthetic performance tiers (the standard benchmark inputs).** `Tier small`
and `Tier medium` use random weights and inputs under a fixed RNG seed
(`source: synthetic` in `datasets/tests/tests.json`). They are
git-ignored and regenerated by `build.py`. BNN has no large tier:
`tier_config.py` sets it to `None` because the full emit exceeds the test-time
budget.

**Optional MNIST accuracy fixtures (real data, unit scale only).** The
committed `bnn_mnist_100.bsdata` and `bnn_mnist_10k.bsdata` fixtures hold real
MNIST images and pre-trained weights for the correctness check (100% on MNIST
100, 97.91% on MNIST 10k). They are small (<5 MB), git-tracked, and exercised
by the `--unit` suite.

| Path | Files | Source | Scale |
|------|-------|--------|-------|
| Synthetic tiers | `bnn_tier_small/medium.bsdata` | random, fixed seed | ~50 MB / ~500 MB, git-ignored |
| MNIST accuracy fixtures | `bnn_mnist_100/10k.bsdata` | real MNIST + trained weights | <5 MB, committed |

`datasets/small/mnist_params.npz` holds the trained weights;
`datasets/tests/generate_tests.py` regenerates the MNIST accuracy
fixtures from it.

### Runtime dependencies

The 8 unit and MNIST tests run from precomputed `.bsdata` files committed to the
repo, so they need only the project runtime and a backend -- no NumPy or PyTorch
required. The 2 performance tier tests (`Tier small`, `Tier medium`) use larger
`.bsdata` files regenerated by `datasets/tests/generate_tests.py`, which requires
NumPy. PyTorch/torchvision are needed only to retrain the MNIST weights
(`datasets/raw/train_mnist.py`).

---

## Tutorial

### Binary dot product

In a standard fully-connected layer, each neuron computes a weighted sum of its inputs
and applies an activation function:

```
output = activation(sum(w_i * x_i) + bias)
```

In a BNN, weights and activations are binary (+1/-1). The dot product simplifies to
counting agreements:

```
dot_product = sum(w_i * x_i)
            = (number of matching bits) - (number of mismatched bits)
            = 2 * popcount(XNOR(w, x)) - N
```

where N is the number of input features. The XNOR operation gives 1 wherever the
weight and input agree, and 0 where they disagree.

### Threshold activation (FINN §4.2.2)

After computing the match count, batch normalization is folded into a simple threshold
comparison. The real-valued batch norm expression

```
sign(gamma * (2 * popcount - N - mu) / sqrt(var + eps) + beta)
```

reduces to

```
activation = 1  if  popcount >= threshold
             0  otherwise
```

where the threshold is precomputed per-neuron by folding the BatchNorm parameters
(FINN §4.2.2: `tau+ = (tau + S)/2`, with `S` the synapse fan-in). At inference time
the threshold is a plain integer per neuron. The folded thresholds come from
`train_mnist.py` and are stored in `mnist_params.npz`.

### Bitsliced encoding

In the bitstream encoding, each bit position represents one **sample** in the inference
batch. If we have M samples to classify, each stream is M bits wide. The N input
features become N streams of M bits each: bit j of stream `x[i]` holds the i-th
feature of the j-th sample.

Weights and thresholds are broadcast constants -- the same value applies to all samples.
A weight of 1 becomes an all-ones stream, and a weight of 0 becomes an all-zeros
stream. A single XNOR operation on M-bit streams evaluates the weight-input comparison
for M samples simultaneously.

### Why not use the DSL's popcount?

This is a subtle but important point. The DSL provides a `popcount()` primitive, but it
counts bits **within a single stream** -- that is, it counts across all M samples and
produces a single integer. BNN inference needs the opposite: it must count matches
**across N feature streams** at each bit position, independently for each sample.

These are orthogonal reductions:

| Reduction | Direction | Result |
|-----------|-----------|--------|
| DSL `popcount(stream)` | Across bit positions (samples) | One integer for all samples |
| BNN feature count | Across N feature streams | One count per sample (stays in stream form) |

The only way to use the DSL's `popcount` primitive for BNN would be to repack so that
the N input dimension is along the stream length (one sample per program invocation)
and split each layer into its own `.bs` file with host-side orchestration between
layers. That is a much larger structural change; we instead implement the popcount as
a **gate-level tree of half-adders and full-adders** operating directly on the
bit-plane streams. FINN's MVTU performs popcount-accumulate at the operation level
(§4.2.1) and lets Vivado HLS synthesize the underlying logic; the specific 16-input
half/full-adder tree here is BitstreamZoo's gate-level instantiation, not a
topology that FINN itself prescribes.

### The three pipeline stages (FINN §4.2.1, §4.2.2)

Each of the K neurons executes three stages. CHUNK_SIZE = 16 throughout; `N_CHUNKS =
N / 16` is passed as a runtime param because the `.bs` DSL `int_expr` grammar does
not support integer division.

**Stage 1 -- XNOR Matching**

For each input feature in a chunk of 16, compute whether the input matches the weight:

```
match = ~(x[base + i] ^ w[k*N + base + i])      // 1 XOR + 1 NOT = 2 ops
```

Per chunk: 32 ops. XNOR gives 1 at every bit position where the input and weight agree,
and 0 where they differ.

**Stage 2 -- Chunked Tree-Reduction Popcount (FINN §4.2.1)**

Each chunk of 16 one-bit matches is reduced to a 5-bit chunk popcount via a 4-layer
binary tree of half-adders and full-adders, then the 5-bit chunk count is
ripple-added into a B-bit per-sample running counter.

```
// Layer 0: 16 1-bit -> 8 (sum, carry) pairs via half-adders.  16 ops.
for j in 0..8 {
    s_l0[j] = m[2*j] ^ m[2*j+1]
    c_l0[j] = m[2*j] & m[2*j+1]
}

// Layer 1: 8 2-bit values -> 4 3-bit values via 2-bit adders.  4 * 7 = 28 ops.
// Layer 2: 4 3-bit values -> 2 4-bit values via 3-bit adders.  2 * 12 = 24 ops.
// Layer 3: 2 4-bit values -> 1 5-bit value (q0..q4).  17 ops.

// Ripple-add q[0..4] into the B-bit running counter.
// bit 0: HA.  bits 1..4: FA.  bits 5..B-1: carry propagation HA.   ~32 ops at B=10.
```

Per chunk at B=10: 32 (XNORs: 16 XOR + 16 NOT) + 85 (tree: 16 + 28 + 24 + 17) +
32 (ripple-add: 2 HA + 4 FA + 5 carry-prop HAs) = **149 ops**. At B=11 the
ripple-add is 34 ops (one extra carry-prop HA), giving **151 ops per chunk**.

Per neuron for N inputs: `N_CHUNKS * (149 or 151) + threshold compare (7B+1)`.
For the MNIST layer 1 (N=784, B=10), this is 49 * 149 + 71 = 7,372 ops/neuron.
For layers 2 and 3 (N=1024, B=11): 64 * 151 + 78 = 9,742 ops/neuron.

Each bit position's counter is independent -- this is the power of the bitsliced
representation. After processing all N features, `count[0..B-1]` holds the binary
representation of the match count at every bit position.

The popcount tree is BitstreamZoo's gate-level realization of FINN's
popcount-accumulate stage (§4.2.1): a half/full-adder reduction tree applied to
bit-plane streams. FINN reports 376 LUTs and 29 FFs for a 128-bit
popcount-accumulate at 200 MHz (§4.2.1), vs 759 LUTs and 84 FFs for signed
accumulate -- roughly half the LUT cost.

**Stage 2 remainder fallback.** If `N % 16 != 0` (e.g., unit tests with N=8, 4), the
chunk loop runs zero times and a per-input ripple-carry fallback handles all inputs:

```
for rem in 0..(N - 16 * N_CHUNKS) {
    int idx = N_CHUNKS * 16 + rem
    stream match = ~(x[idx] ^ w[k*N + idx])
    // ripple-add single match bit into count[0..B-1]
}
```

For MNIST's N=784, N=1024, N=1024, all three layers have `N_CHUNKS = 49, 64, 64` and
the remainder loop is empty. For the unit tests (N=8, 4), the remainder loop handles
everything.

**Stage 3 -- Threshold Comparison (7B + 1 ops)**

An MSB-first binary comparator determines whether count >= threshold:

```
gt = ZERO          // "count is strictly greater" accumulator
eq = ONES          // "all bits compared so far are equal" accumulator

for j in 0..B {
    a  = count[B-1-j]             // counter bit (MSB first)
    tb = thresh[k*B+B-1-j]        // threshold bit (MSB first)

    a_gt = a & ~tb                // a=1, tb=0: count bit > threshold bit (2 ops)
    a_eq = ~(a ^ tb)              // a == tb: bits are equal (2 ops)
    gt   = gt | (eq & a_gt)       // update "greater than" (2 ops)
    eq   = eq & a_eq              // update "equal so far" (1 op)
}

act[k] = gt | eq                  // count >= threshold (1 op)
```

This is the same BitWeaving MSB-to-LSB comparator used in the BitWeaving benchmark.
Threshold folding happens at training time (`train_mnist.py`) so the thresholds passed
to the `.bs` program are already in FINN §4.2.2 form.

### Why the tree-reduction form is worth the verbosity

The previous implementation of `bnn_mnist.bs` used a per-input ripple-carry counter
(one half-adder chain of length B applied to every input, costing ~22 ops per input
pixel at B=10). That is a valid gate-level popcount but it has two structural
disadvantages relative to the chunked tree form:

1. **Gate count.** Tree reduction asymptotically uses ~6 ops per input bit (binary
   tree) vs 2B ops per input bit for per-input ripple-carry. At B=10, tree is ~3.7x
   cheaper per input.
2. **Depth.** Per-input ripple is serial in both the input dimension and the counter
   width, giving critical-path depth proportional to N * B. The chunked tree form
   replaces each per-input increment with one 16-input parallel tree per chunk plus
   one ripple-add of the 5-bit chunk count into the B-bit running counter. The
   running counter still serializes work across chunks, so the critical path
   scales with N_CHUNKS = N/16 rather than N (an empirical sweep over the chunk
   count shows ~2 added depth levels per extra chunk). The constant per chunk is
   much smaller than the 2B per input of the ripple form, giving a several-fold
   depth reduction at N=784.

On the MNIST 784->1024->1024->1024 configuration:

| | Old (per-input ripple-carry) | New (chunked tree + popcount) |
|---|---|---|
| total_ops (MNIST) | **68,226,048** | **27,500,544** (2.48x fewer) |
| critical path depth | 2,907 | **463** (6.3x shorter) |
| ILP | 23,437 | **59,396** (2.5x more) |
| gate amplification per XNOR | 23.5 | **9.48** |

The op count drops because most of the 2NB per-input work collapses into the shared
tree layers. The depth drops because the across-input dependency now runs through
N_CHUNKS = N/16 chunk-level ripple-adds instead of N per-input ripple-adds (within
a chunk the tree is logarithmic in 16, but the running counter serializes the
chunks themselves). ILP rises because the tree exposes chunk-level parallelism
that ripple-carry hides behind serial dependencies.

### Bit-plane vs native hardware popcount

Even with the tree form, bit-plane BNN is ~2.9x slower than AVX-512 `vpopcntq` at
saturated batch size. That ratio is the **structural overhead** of emulating a
hardware horizontal-reduction primitive with bit-plane gates on a substrate that
already has hardware popcount. On substrates that **do not** have hardware popcount
(PIM such as SIMDRAM / Ambit / Neural Cache, FPGA BNN accelerators like FINN,
custom ASIC MVTUs), some form of gate-level reduction is required at synthesis
time; the specific 16-input half/full-adder tree here is BitstreamZoo's choice
within that family, not a topology pulled from any of those accelerators.

### FINN's hardware connection

FINN (Umuroglu et al., FPGA 2017) implements BNN inference on FPGAs using the
Matrix-Vector Threshold Unit (MVTU). The MVTU's datapath is:

1. XNOR gates for weight-activation matching
2. Popcount-accumulate to count matches (Vivado HLS synthesizes the gate-level
   reduction; FINN reports the operation-level LUT/FF cost, not a specific
   compressor topology)
3. Threshold comparator for batch-norm-folded activation

Our `.bs` program mirrors this XNOR-popcount-threshold structure. The chunked
16-input half/full-adder tree is BitstreamZoo's gate-level realization of FINN's
popcount stage. The key encoding difference is batching: FINN processes one sample
with an N-wide XNOR-and-popcount unit and replicates the unit spatially across
multiple simultaneously-active neurons; our bitsliced version processes M samples
simultaneously with N streams of M bits each, and the tree-reduction circuit is
applied once symbolically for all M samples.

### Multi-layer MNIST classification

The benchmark includes a 3-layer BNN for MNIST digit recognition, encoded as a single
`bnn_mnist.bs` program:

**Architecture**: 784 -> 1024 -> 1024 -> 1024 (FINN LFC style, 3 hidden binary layers)

- **Input**: 28x28 MNIST images binarized (pixel > 127 -> 1, else 0).
- **Layers 1-3**: All three binary FC layers (XNOR + chunked tree popcount +
  threshold) are in one `.bs` program. Intermediate activations flow between layers
  inside the program via stream arrays `h1[K1]` and `h2[K2]`.
- **Layer 4 (classifier, 1024 -> 10, argmax)**: Uses XNOR-popcount without
  thresholding, followed by argmax. This is applied only during test data generation
  in `train_mnist.py` to compute expected labels -- it is not part of the `.bs` program.

**How bitsliced multi-layer inference works end-to-end:**

1. **Encode images into streams.** Given M binarized 784-pixel images, create 784
   streams, each M bits wide. Stream `x[i]` packs feature `i` across all M images.

2. **Broadcast weights and thresholds.** Each weight stream is either all-1s or all-0s.
   Similarly for threshold bits. All 3 layers' weights and thresholds are passed as
   input streams.

3. **Run `bnn_mnist.bs`.** The program executes all 3 layers sequentially: layer 1
   (784->1024) produces hidden activations h1, layer 2 (1024->1024) reads h1 and
   produces h2, layer 3 (1024->1024) reads h2 and produces the final output
   `act[1024]`. All layers run in parallel across M images via bitsliced streams.

**Batch norm folding math (FINN §4.2.2):**

Training uses binary weights with STE (straight-through estimator) and batch
normalization after each hidden layer. Post-training, batch norm is folded into
per-neuron integer thresholds:

1. Pre-BN output: `z = 2 * popcount(XNOR(w, x)) - N` (in {-N, -N+2, ..., N})
2. Batch norm: `y = gamma * (z - mu) / sqrt(var + eps) + beta`
3. Activation: `sign(y)`, i.e., `y >= 0`

Converting to popcount threshold: `T = ceil((z_thresh + N) / 2)`, clamped to `[0, N]`.

**MNIST op counts (new chunked tree-reduction form):**

| Layer | N | K | B | N_CHUNKS | Ops |
|-------|---|---|---|----------|-----|
| 1 (784->1024) | 784 | 1024 | 10 | 49 | 7,548,928 |
| 2 (1024->1024) | 1024 | 1024 | 11 | 64 | 9,975,808 |
| 3 (1024->1024) | 1024 | 1024 | 11 | 64 | 9,975,808 |
| 4 (1024->10, argmax) | -- | -- | -- | -- | (test data gen only) |
| **Total (layers 1-3)** | | | | | **27,500,544 ops** |

Per-layer cost: `K * (N_CHUNKS * (32 + 85 + 22 + 2*(B-5)) + 7*B + 1)`, where the
per-chunk cost decomposes as 32 XNORs + 85 popcount-tree ops + (22 + 2*(B-5)) ripple-add
into the B-bit running counter. For the MNIST config this gives per-chunk costs of
149 ops at B=10 (layer 1) and 151 ops at B=11 (layers 2 and 3).

For comparison, the previous per-input ripple-carry implementation cost
**68,226,048 ops** on the same configuration. The new chunked tree-reduction form is
**2.48x fewer ops** with **6.3x shorter critical path depth** (463 vs 2907) and
**2.5x more ILP** (59,396 vs 23,437).

**Training details:**

- Optimizer: Adam, lr=1e-3, cosine annealing
- STE: custom `SignSTE` autograd function (hardtanh gradient clipping)
- Weight clamping to [-1, 1] after each step
- MNIST: 100 epochs, 1024-wide, ~98% test accuracy (FINN LFC)

---

## .bs Program Details

One 3-layer program: `src/bnn_mnist.bs` (516 lines, verbose because the popcount tree
is inlined per layer), with layer dimensions 784->1024->1024->1024.

### Declarations

```
param int N1              // layer 1 input features
param int K1              // layer 1 output neurons
param int B1              // layer 1 counter bits = ceil(log2(N1+1))
param int N1_CHUNKS       // = N1 / 16 (passed as param; DSL has no int division)
param int N2              // layer 2 input features (= K1)
param int K2              // layer 2 output neurons
param int B2              // layer 2 counter bits
param int N2_CHUNKS       // = N2 / 16
param int N3              // layer 3 input features (= K2)
param int K3              // layer 3 output neurons
param int B3              // layer 3 counter bits
param int N3_CHUNKS       // = N3 / 16

input stream x[N1]          // features (bitsliced across samples)
input stream w1[N1 * K1]    // layer 1 weights
input stream t1[B1 * K1]    // layer 1 thresholds
input stream w2[N2 * K2]    // layer 2 weights
input stream t2[B2 * K2]    // layer 2 thresholds
input stream w3[N3 * K3]    // layer 3 weights
input stream t3[B3 * K3]    // layer 3 thresholds

output stream act[K3]        // final layer activations
```

Each program is parameterized by 12 integers (4 per layer including `N*_CHUNKS`).
Weights and thresholds are passed as input streams because they participate in
bitwise operations at every bit position. Intermediate activations (`h1`, `h2`) are
internal arrays, not exposed as outputs.

### Per-layer structure

Each of the 3 layers follows the same pattern. For layer 1:

```
stream h1[K1]
for k in 0..K1 {
    stream count[B1]         // B-bit counter for this neuron
    for b in 0..B1 { count[b] = ZERO }

    // Chunked tree-reduction popcount: N/16 chunks of 16 inputs each
    for c in 0..N1_CHUNKS {
        int base = c * 16

        // Step 1: 16 XNORs
        stream m[16]
        for i in 0..16 {
            stream xor_xw = x[base + i] ^ w1[k * N1 + base + i]
            m[i] = ~xor_xw
        }

        // Step 2: 4-layer gate-level tree popcount -> q0..q4 (5-bit chunk count)
        //   Layer 0: 8 HAs (16 ops)
        //   Layer 1: 4 (2-bit + 2-bit -> 3-bit) adders (28 ops)
        //   Layer 2: 2 (3-bit + 3-bit -> 4-bit) adders (24 ops)
        //   Layer 3: 1 (4-bit + 4-bit -> 5-bit) adder (17 ops)
        // ... 85 ops total, inlined explicitly in 3-address SSA form ...

        // Step 3: ripple-add q0..q4 into count[0..B1-1]  (~32 ops at B1=10)
    }

    // Remainder: handle (N1 - 16*N1_CHUNKS) leftover inputs via per-input ripple.
    // Empty for N1 a multiple of 16; runs for unit tests where N1 < 16.
    for rem in 0..(N1 - 16 * N1_CHUNKS) {
        int idx = N1_CHUNKS * 16 + rem
        stream match = ~(x[idx] ^ w1[k * N1 + idx])
        // ... ripple-carry add single bit into count[0..B1-1] ...
    }

    // Stage 3: threshold comparison (MSB-to-LSB, BitWeaving form)
    stream gt = ZERO
    stream eq = ONES
    for j in 0..B1 {
        stream a = count[B1 - 1 - j]
        stream tb = t1[k * B1 + B1 - 1 - j]
        // ... gt = gt | (eq & a_gt); eq = eq & a_eq ...
    }
    h1[k] = gt | eq
}
```

Layer 2 reads `h1` as input and writes `h2`. Layer 3 reads `h2` and writes `act`.
The popcount tree is textually inlined per layer (rather than factored out) because
the `.bs` DSL has no function definitions; the three copies differ only in the
layer-specific `N*`, `K*`, `B*`, and `N*_CHUNKS` parameters.

### DSL Features Used

| Feature | Role |
|---------|------|
| `^` (XOR) | XNOR matching (with NOT), half-adder sum, equality test |
| `~` (NOT) | XNOR matching, comparator logic |
| `&` (AND) | Half-adder carry, comparator logic |
| `\|` (OR) | Full-adder carry-out, comparator accumulation, final activation |
| Arrays | `x[N1]`, `w1[N1*K1]`, `t1[B1*K1]`, `h1[K1]`, `h2[K2]`, `act[K3]`, plus per-chunk tree temps `m[16]`, `s_l0[8]`, `c_l0[8]`, `r_b0[4]`, `r_b1[4]`, `r_b2[4]`, `p_b0[2]`, `p_b1[2]`, `p_b2[2]`, `p_b3[2]` |
| Nested `for` | k (neurons) -> c (chunks) -> { i (xnor), j (tree levels), b (ripple-add) }, repeated x3 layers |
| `param int` | N1, K1, B1, N1_CHUNKS, N2, K2, B2, N2_CHUNKS, N3, K3, B3, N3_CHUNKS |
| `int` locals | `int base = c * 16`, `int idx = N1_CHUNKS * 16 + rem` |
| `ZERO`, `ONES` | Counter initialization, comparator initialization |
| `+` (ADD) | Not used (arithmetic is decomposed into gate-level half/full adders) |
| `popcount` | Not used (reduces across the wrong axis -- see "Why not use the DSL's popcount?") |

### Scaling behavior

The total operation count is the sum over 3 layers. Each layer has cost
`K * (N_CHUNKS * (32 + 85 + 22 + 2*(B-5)) + remainder * 2*(1+B) + 7*B + 1)`,
where the per-chunk term decomposes as 32 XNORs + 85 popcount-tree ops +
`(22 + 2*(B-5))` ripple-add into the B-bit counter. This evaluates to 149 ops/chunk
at B=10, 151 at B=11, and 145/147 at B=8/9 (tier-small layers).

| Configuration | Total ops (new) | Old (ripple-carry) | Speedup |
|---------------|----------------:|-------------------:|--------:|
| MNIST (784->1024->1024->1024) | 27,500,544 | 68,226,048 | 2.48x |
| Tier small (128->256->256->4) | 939,712 | 1,952,256 | 2.08x |
| Unit tests (8->4, 4->4, 4->2) | 760 | 760 | 1.00x (remainder path) |

The unit-test configuration has no speedup because all three layers have N < 16,
so the chunk loop is empty and the remainder loop handles every input via the
original per-input ripple-carry. This is intentional: the new `.bs` file is
backward-compatible with any N >= 0 through the remainder fallback.

The number of operations is independent of how many samples are being classified.
Whether you process 1 sample or 1 million samples, the program executes the same number
of bitwise instructions. The stream *width* changes, but the instruction count does
not. On a GPU with 64-bit words, a 1M-bit stream corresponds to roughly 15,625 words
of SIMD parallelism.

---

## Running

### Prerequisites

Make sure you are in the project root directory and have Python 3 available. The 8
unit and MNIST tests run from committed `.bsdata` files and need only the project
runtime plus a backend -- no NumPy or PyTorch required. The 2 performance tier
tests use larger `.bsdata` files regenerated by `datasets/tests/generate_tests.py`,
which requires NumPy. PyTorch/torchvision are needed only to retrain the MNIST
weights.

### Standalone runner

```bash
python3 benchmark/apps/bnn/src/run.py
```

This runs all 10 tests (6 unit + 2 MNIST regression + 2 performance) via
`tests.json` and `.bsdata` files. Each test compares the program's output
activations against expected values pre-recorded from the bitstream reference;
the print line reports `(bitlength samples, op_count ops)`:

```
  [PASS] All match  (1 samples, 760 ops)
  [PASS] No match  (1 samples, 760 ops)
  [PASS] Multi-neuron chain  (4 samples, 760 ops)
  [PASS] Exhaustive N1=8  (256 samples, 760 ops)
  [PASS] Random 16-bit x100  (1600 samples, 76000 ops)
  [PASS] Random 64-bit x50  (3200 samples, 135400 ops)
  [PASS] MNIST 100  (100 samples, 27500544 ops)
  [PASS] MNIST 10k  (10000 samples, 27500544 ops)
  [PASS] Tier small  (3030303 samples, 939712 ops)
  [PASS] Tier medium  (30303030 samples, 939712 ops)

Results: 10 passed, 0 failed, 10 total
```

The MNIST tests check that the bitstream program reproduces the expected per-neuron
activations of a model whose end-to-end classification accuracy was verified at
test-generation time (100/100 on MNIST 100, 9791/10000 = 97.91% on MNIST 10k).

### Pre-trained weights and training scripts

| File | Description |
|------|-------------|
| `datasets/small/mnist_params.npz` | Pre-trained MNIST weights (1024-wide FINN LFC, ~98% accuracy) |
| `datasets/raw/train_mnist.py` | One-time MNIST training script (requires PyTorch + torchvision + numpy) |

To retrain MNIST:
```bash
pip install torch torchvision numpy
python benchmark/apps/bnn/datasets/raw/train_mnist.py
```

`mnist_params.npz` contains:
- `w1`(1024x784), `w2`(1024x1024), `w3`(1024x1024), `w4`(10x1024) -- binary {0,1}
- `t1`(1024), `t2`(1024), `t3`(1024) -- integer thresholds (folded BatchNorm, FINN §4.2.2)
- `test_images`(10000x784), `test_labels`(10000) -- full binarized MNIST test set
- `accuracy` -- ~0.98 (scalar BNN accuracy on first 1K images)

### Test suite (10 tests)

Tests fall into three categories:

- **Unit tests** (6): Small 3-layer networks with hand-crafted or random data. Verify
  the bitstream program computes correctly. The fixed cases, `Exhaustive N1=8`, and
  `Random 16-bit x100` use 8-input layer-1 dimensions (`N1=8`, so `N1_CHUNKS=0`) and
  exercise only the remainder-loop fallback path. `Random 64-bit x50` uses `N1=16,
  K1=8, N2=N3=8, K2=8, K3=4` so layer 1 runs through one 16-input chunk (the tree
  popcount path) while layers 2 and 3 still use the remainder fallback. Run in
  milliseconds.
- **MNIST regression tests** (2): Real MNIST images with trained weights. Verify
  that the `.bs` program reproduces the expected activations of a model whose
  end-to-end accuracy was checked at test-generation time (>=95% on MNIST 100,
  >=97% on MNIST 10k; current measured 100% and 97.91%). Run on C++ or Python
  backend.
- **Performance tests** (2): Large random data (3M and 30M bitlength) with a smaller
  network (128->256->256->4). Used to benchmark speed on C++ and CUDA backends.

| Program | Test | Category | Description |
|---------|------|----------|-------------|
| bnn_mnist | All match | unit | Small 3-layer, all inputs match (bitlength=1) |
| bnn_mnist | No match | unit | Small 3-layer, no inputs match (bitlength=1) |
| bnn_mnist | Multi-neuron chain | unit | 3-layer with mixed outcomes (bitlength=4) |
| bnn_mnist | Exhaustive N1=8 | unit | All 256 input patterns through 3 layers |
| bnn_mnist | Random 16-bit x100 | unit | 100 random configs, 16 samples each |
| bnn_mnist | Random 64-bit x50 | unit | 50 random configs, 64 samples each |
| bnn_mnist | MNIST 100 | regression | 100 images, 784->1024->1024->1024; gen-time acc >= 95% |
| bnn_mnist | MNIST 10k | regression | 10K images, full test set; gen-time acc >= 97% |
| bnn_mnist | Tier small | performance | 128->256->256->4, 3.03M samples |
| bnn_mnist | Tier medium | performance | 128->256->256->4, 30.3M samples |

### MNIST testing methodology

The MNIST tests verify correctness at three levels:

1. **Training-time cross-validation.** `train_mnist.py` compares scalar Python BNN
   inference against PyTorch inference on 100 images. Result: 100/100 match, confirming
   batch norm folding is correct.

2. **MNIST 100 (test-generation accuracy assertion).** `generate_tests.py` runs the
   3-layer bitstream reference over 100 images, feeds the resulting activations
   through layer 4 (argmax) to get predictions, and asserts accuracy >= 95% before
   recording the per-neuron activations as the expected output of `MNIST 100`.
   Measured at generation: 100%. The runtime test compares the `.bs` program's
   activations against this recorded vector.

3. **MNIST 10k (test-generation accuracy assertion).** Same procedure on the full
   10,000-image test set; one bitwise op classifies 10K images at once. Assertion:
   accuracy >= 97%. Measured at generation: 97.91%. The runtime test again
   compares activations against the recorded vector.

Accuracy is preserved byte-for-byte between the old (per-input ripple-carry) and new
(chunked tree-reduction) implementations because both compute the same popcount and
the same threshold comparison. The `.bsdata` expected outputs are regenerated from the
current `.bs` implementation via the Python reference in `generate_tests.py`, and both
Python and C++ interpreters agree on every test.

---

## References

- Hubara, I., Courbariaux, M. et al., "Binarized Neural Networks", *NeurIPS*, 2016.
  ("Running a BNN" — Algorithm 4 in the NeurIPS version, Algorithm 5 in the
  arXiv preprint [arXiv:1602.02830] — specifies the XnorDotProduct -> Sign(BatchNorm)
  inference pipeline.)
- Rastegari, M. et al., "XNOR-Net: ImageNet Classification Using Binary Convolutional
  Neural Networks", *ECCV*, 2016.
- Umuroglu, Y. et al., "FINN: A Framework for Fast, Scalable Binarized Neural Network
  Inference", *FPGA*, 2017.
  - §4.2.1 "Popcount for Accumulation": the dot product reduction is implemented as
    a popcount-accumulate (Vivado HLS synthesizes it to FPGA logic), which costs
    approximately half the LUT and FF resources of signed accumulation
    (376 vs 759 LUTs for a 128-bit accumulator at 200 MHz).
  - §4.2.2 "Batchnorm-activation as Threshold": folds `Sign(BatchNorm(z, theta))` into
    `popcount >= tau`, where `tau = (tau_orig + S) / 2` and `S` is the synapse fan-in.
