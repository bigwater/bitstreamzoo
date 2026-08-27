# Circuit Simulation — Bitstream Benchmark

## Overview

This benchmark demonstrates bitwise-parallel circuit simulation using the bitstream
DSL. A single execution of the `.bs` program evaluates a digital circuit on many test
vectors simultaneously -- each bit position in a stream is a different test input.

Eleven of the twelve circuits come from the **ISCAS'85 combinational benchmark
suite**, introduced by Brglez and Fujiwara at the IEEE International Symposium on
Circuits and Systems in 1985. These are standard benchmarks used throughout the
electronic design automation (EDA) community for testing logic simulation, fault
simulation, and circuit optimization algorithms. The eleven ISCAS'85 netlists
(`c17`, `c432`, `c499`, `c880`, `c1355`, `c1908`, `c2670`, `c3540`, `c5315`, `c6288`,
`c7552`) were downloaded from the University of Toronto ECE1767 course archive.
The twelfth netlist, `adder4.bench`, is a small local 4-bit ripple-carry adder
included as a sanity-check circuit.

This benchmark implements the same technique used in production EDA tools for
**fault simulation** and **logic verification**:

- A combinational circuit with `n` input ports and `m` test vectors is simulated by
  making each wire a bitstream of length `m`. Bit `t` of wire `w` holds `w`'s value
  for test vector `t`.
- Each gate (AND, OR, NOT, ...) becomes one or more bitwise instructions that evaluate
  the gate on **all m vectors at once**. Total cost scales with the number of gates,
  regardless of `m`.
- Bit-parallel fault simulation -- packing many test vectors into machine words and
  evaluating each gate with a single bitwise instruction -- is a standard technique
  in commercial ATPG and fault-simulation tools. On GPUs, the parallelism scales
  further -- GATSPI (Zhang et al., DAC 2022)
  reports a 449x turnaround speedup in a glitch-optimization flow, with
  simulation-kernel speedups of up to 1668x on a single GPU and 7412x on a multi-GPU
  system.

Our bitstream DSL is a programmable generalization of this: instead of a hard-coded
gate-level simulator, we have a general-purpose language that expresses the same
computation.

### Why ISCAS'85?

**Perfect DSL fit.** ISCAS'85 circuits are purely combinational -- DAGs of
AND/OR/NOT/NAND/NOR/XOR gates with no flip-flops or memory. Each gate maps
mechanically to one or more bitstream operations (NAND/NOR/XNOR expand to two,
multi-input gates to a chain of binary ops). No DSL extensions needed.

**Industry-standard for bitwise-parallel simulation.** The technique we use -- packing
K test vectors into K-bit words, evaluating all gates with single bitwise instructions
-- is how EDA tools do gate-level fault simulation. ISCAS'85 is a long-standing
public benchmark in the gate-level GPU simulation literature (e.g. Chatterjee et al.
2011, GATSPI 2022, which themselves evaluate on OpenSPARC- and industrial-scale
designs). Using ISCAS'85 lets readers compare circuit structure and complexity
against that body of work.

**Size spectrum.** The suite ranges from 6 gates (c17) to 3500+ gates (c7552), showing
how the interpreter scales across program sizes.

**Easy conversion.** The `.bench` format is trivial to parse (one gate per line), and
`datasets/bench2bs.py` does a mechanical topo-sort + gate-to-bitwise translation. No human
effort per circuit.

**Verifiability without domain knowledge.** We don't need to know what a circuit
computes to test it -- generate random inputs, run both the bitstream program and the
gate-level reference simulator, compare outputs.

Note: ISCAS'89 (sequential circuits with flip-flops) would need clock-cycle simulation
and doesn't map as cleanly to a single bitstream program execution. The purely
combinational ISCAS'85 set is the natural starting point.

---

## Dataset

All 12 `.bench` netlists live in `datasets/raw/` and are mechanically converted to `.bs`
by `datasets/raw/convert.py` (which calls `datasets/bench2bs.py`). The resulting
`netlist_*.bs` files are committed to `src/` and discovered at run time by the unified
framework. The eleven ISCAS'85 netlists come from the University of Toronto ECE1767
course archive (https://www.eecg.toronto.edu/~ece1767/project/iscas.html);
`adder4.bench` is a local 4-bit ripple-carry adder.

The "Gates" column below counts gate statements in the committed U Toronto `.bench`
files. These counts can differ from the canonical Brglez 1985 ISCAS'85 summary
numbers (e.g. c432 = 160 gates, c499 = 202 gates), because the U Toronto archive
distributes a variant in which some XOR / multi-input gates are decomposed into
two-input AND / OR / NAND / NOR / NOT primitives. For most circuits only the gate
granularity differs and the structure and I/O behavior are otherwise preserved.
The exception is c2670: the archive version exposes 157 inputs / 64 outputs,
whereas the canonical Brglez 1985 ISCAS'85 specification lists 233 inputs / 140
outputs.

| Netlist | Description | Inputs | Outputs | .bench Gates | .bs Instructions |
|---------|-------------|--------|---------|--------------|------------------|
| `c17.bench` | Tiny NAND network (sanity check) | 5 | 2 | 6 | 12 |
| `adder4.bench` | 4-bit ripple-carry adder | 9 | 5 | 20 | 20 |
| `c432.bench` | 27-channel interrupt controller | 36 | 7 | 232 | 440 |
| `c499.bench` | 32-bit SEC circuit | 41 | 32 | 618 | 974 |
| `c880.bench` | 8-bit ALU | 60 | 26 | 383 | 583 |
| `c1355.bench` | 32-bit SEC circuit (different decomposition) | 41 | 32 | 546 | 1006 |
| `c1908.bench` | 16-bit SEC/DED circuit | 33 | 25 | 880 | 1434 |
| `c2670.bench` | 12-bit ALU + controller | 157 | 64 | 1193 | 1666 |
| `c3540.bench` | 8-bit ALU | 50 | 22 | 1669 | 2349 |
| `c5315.bench` | 9-bit ALU | 178 | 123 | 2307 | 3454 |
| `c6288.bench` | 16x16 multiplier | 32 | 32 | 2416 | 4544 |
| `c7552.bench` | 34-bit adder + magnitude comparator with input parity checking | 207 | 108 | 3512 | 5124 |

The instruction count exceeds the gate count for two reasons: each NAND, NOR, or
XNOR gate decomposes into a base operation followed by a NOT, and each multi-input
gate is expanded into a chain of two-input operations.

Test data is **synthetic**: random input test vectors are generated by `getrandbits()`
for large-scale tests, and exhaustive enumeration is used for small circuits (c17 with
5 inputs = 32 combos, adder4 with 9 inputs = 512 combos).

## Auxiliary raw-input datasets

`datasets/{small,medium,large}/` holds optional auxiliary raw input vectors
(synthetic random stimuli) for two representative circuits. These are separate
from the canonical `.bsdata` test tiers consumed by the benchmark runner; the
canonical ~50 MB / ~500 MB / ~5 GB tiers for all 12 netlists live in
`datasets/tests/` (see Test Suite below). The lone exception is the `adder4`
sanity-check circuit, smaller at ~35 MB / ~350 MB / ~3.5 GB.

| Tier | Files | Vectors (c432/c7552) | File Size |
|------|-------|---------------------|-----------|
| small | c432_vectors.npz, c7552_vectors.npz | 222K / 39K | ~1 MB each |
| medium | c432_vectors.npz, c7552_vectors.npz | 2.2M / 386K | ~10 MB each |
| large | c432_vectors.npz, c7552_vectors.npz | 22M / 3.9M | ~100 MB each |

Only the `small` pair is committed; regenerate `medium`/`large` with
`datasets/make_data.py`.

These cover c432 (36 inputs, 27-channel interrupt controller) and c7552
(207 inputs, 34-bit adder + magnitude comparator with input parity checking).
All 12 netlists (11 ISCAS'85 plus `adder4`) have `.bs` programs and `.bsdata`
tests; only c432 and c7552 have these auxiliary `.npz` inputs.

**Source:** ISCAS-85 netlist structure (real); test vectors (synthetic random)
**Generation:** `python datasets/make_data.py`

---

## Tutorial

### What is a .bench netlist?

A `.bench` file describes a digital circuit as a list of gates and wires. It is one of
the most common formats in EDA research -- the ISCAS'85 benchmark suite uses it.

Here is the smallest ISCAS circuit, `c17.bench`:

```
INPUT(G1)
INPUT(G2)
INPUT(G3)
INPUT(G4)
INPUT(G5)

OUTPUT(G16)
OUTPUT(G17)

G8  = NAND(G1, G3)
G9  = NAND(G3, G4)
G12 = NAND(G2, G9)
G15 = NAND(G9, G5)
G16 = NAND(G8, G12)
G17 = NAND(G12, G15)
```

Each line is either an input declaration, an output declaration, or a gate. Gates
reference previously-defined wires. That is the entire format.

### The original algorithm

A combinational circuit is a directed acyclic graph (DAG) of logic gates. It has
primary inputs, primary outputs, and internal gates. There are no feedback loops or
memory elements -- outputs depend only on the current inputs.

The standard gate types are:

- **AND**, **OR**, **XOR** -- basic logic gates
- **NAND**, **NOR**, **XNOR** -- negated versions
- **NOT** -- single-input inverter
- **BUF** -- single-input buffer (identity)

Gate-level simulation evaluates the circuit by processing gates in topological order
(every gate's inputs are computed before the gate itself). For a given set of input
values, you walk through the gates from inputs to outputs, computing each gate's
output from its inputs. This produces the output values for that one test vector.

In standard EDA simulation, you repeat this process for every test vector. If you have
N test vectors and G gates, the total work is O(N * G).

### Adaptation to bitstream programs

The key insight is **bit-slicing**: instead of storing a single 0 or 1 for each wire,
store an entire bitstream where bit position `t` holds the value of that wire for test
vector `t`. Then every gate operation becomes a bitwise operation on the full stream,
evaluating all test vectors in parallel.

For example, consider an AND gate with inputs `a` and `b`:

```
Standard simulation (one test vector):
    a = 1, b = 0  -->  output = 0

Bit-sliced simulation (four test vectors packed into one stream):
    a = 0b1101    (test vectors 0,2,3 have a=1; test vector 1 has a=0)
    b = 0b1010    (test vectors 1,3 have b=1; test vectors 0,2 have b=0)
    output = a & b = 0b1000    (only test vector 3 has both a=1 and b=1)
```

In the bit-sliced version, a single `&` instruction computes the AND gate for all four
test vectors at once. On a GPU with 64-bit words, you get 64 test vectors per
instruction. With 1M-bit streams (about 15,625 64-bit words), you evaluate one million
test vectors simultaneously.

Each gate type maps to one or two bitwise operations:

| Gate Type  | Bitstream Operation    | Operation Count |
|------------|------------------------|-----------------|
| AND(a,b)   | `g = a & b`            | 1               |
| OR(a,b)    | `g = a \| b`           | 1               |
| XOR(a,b)   | `g = a ^ b`            | 1               |
| NOT(a)     | `g = ~a`               | 1               |
| BUF(a)     | `g = a` (copy)         | 0               |
| NAND(a,b)  | `t = a & b; g = ~t`    | 2               |
| NOR(a,b)   | `t = a \| b; g = ~t`   | 2               |
| XNOR(a,b)  | `t = a ^ b; g = ~t`    | 2               |

```
                    test vectors (SIMD parallelism) -- the bitstream width
                    --------------------------------------------------------->
                    vec 0     vec 1     vec 2     ...   vec K-1
                   +--------------------------------------------------+
  G1               |   1         0         1      ...     0      | <- one bitstream (K bits)
  G2               |   0         1         1      ...     1      |
  G3               |   1         1         0      ...     0      |
  G4               |   0         1         1      ...     1      |
  G5               |   1         0         0      ...     1      |
                   +--------------------------------------------------+
```

**Traditional approach (one test at a time):**
```
6 gates x 1000 tests = 6000 gate evaluations
```

**Bitstream approach (all tests packed into bitstreams):**
```
6 NAND gates x 1 execution = 12 bitwise operations (each NAND = AND + NOT,
                                                    processing 1000 tests each)
```

With W-bit words, each word carries W test vectors per bitwise instruction.
This artifact's CUDA backend supports both 32- and 64-bit words. The circuit
does not change -- you just make the bitstreams wider.

### How bench2bs converts netlists to .bs

The converter (`datasets/bench2bs.py`) translates each gate into bitwise operations:

```bash
python3 benchmark/apps/circuit_sim/datasets/bench2bs.py benchmark/apps/circuit_sim/datasets/raw/c17.bench
```

Output:

```
input stream G1
input stream G2
input stream G3
input stream G4
input stream G5
output stream G16
output stream G17

stream _t_G8 = G1 & G3
stream G8 = ~_t_G8
stream _t_G9 = G3 & G4
stream G9 = ~_t_G9
stream _t_G12 = G2 & G9
stream G12 = ~_t_G12
stream _t_G15 = G9 & G5
stream G15 = ~_t_G15
stream _t_G16 = G8 & G12
G16 = ~_t_G16
stream _t_G17 = G12 & G15
G17 = ~_t_G17
```

Each `NAND(a,b)` becomes an AND followed by a NOT. The converter handles all gate
types (AND, OR, NAND, NOR, XOR, XNOR, NOT, BUF), multi-input gates, and topological
sorting.

The conversion is fully automated by `datasets/bench2bs.py`, which:

1. Parses the `.bench` netlist format.
2. Topologically sorts the gates so that every gate's inputs are defined before the
   gate is used.
3. Emits each gate as one or two bitstream operations.
4. Sanitizes wire names (e.g., adding an `n` prefix to numeric names, handling
   reserved words).

### Verification

We use two independent implementations and compare their outputs:

1. **`simulate_bench()`** (in `datasets/bench2bs.py`): Evaluates the original `.bench`
   netlist one test vector at a time. Straightforward gate-by-gate simulation -- easy
   to trust.

2. **Bitstream execution**: Converts `.bench` to `.bs`, runs the bitstream program
   with all test vectors packed into bitstreams.

For each test vector, we check that every output wire matches between the two
implementations. If they agree on all vectors, the conversion and bitstream execution
are correct.

### Example: 4-bit ripple-carry adder

**Original netlist (`adder4.bench`):**

```
INPUT(a0) .. INPUT(a3)
INPUT(b0) .. INPUT(b3)
INPUT(cin)

OUTPUT(s0) .. OUTPUT(s3)
OUTPUT(cout)

p0 = XOR(a0, b0)
s0 = XOR(p0, cin)
g0 = AND(a0, b0)
pc0 = AND(p0, cin)
c0 = OR(g0, pc0)
... (repeats for bits 1, 2, 3)
```

**Generated bitstream program:**

```
input stream a0 .. a3, b0 .. b3, cin
output stream s0 .. s3, cout

stream p0 = a0 ^ b0
s0 = p0 ^ cin
stream g0 = a0 & b0
stream pc0 = p0 & cin
stream c0 = g0 | pc0
... (continues for bits 1, 2, 3)
cout = g3 | pc3
```

Since the adder uses only AND, OR, and XOR gates (no NAND/NOR/XNOR), there is a
one-to-one mapping between netlist gates and `.bs` operations. Each full adder
contributes 5 operations (2 XOR + 2 AND + 1 OR), for 20 total in the 4-bit adder.

### Using the converter from Python

```python
from benchmark.apps.circuit_sim.datasets.bench2bs import convert as bench2bs
from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter

# Convert .bench to .bs source code
with open("benchmark/apps/circuit_sim/datasets/raw/adder4.bench") as f:
    bs_source = bench2bs(f.read())

# Parse the .bs program
program = parse(bs_source)

# Prepare input streams (each bit position = one test vector)
inputs = {
    "a0": 0b01, "a1": 0b01, "a2": 0b01, "a3": 0b01,
    "b0": 0b00, "b1": 0b01, "b2": 0b00, "b3": 0b00,
    "cin": 0b00,
}

# Run the bitstream program
interp = Interpreter()
result = interp.run(program, inputs=inputs)

# Extract results for test vector 0
s0 = (result["s0"] >> 0) & 1
s1 = (result["s1"] >> 0) & 1
s2 = (result["s2"] >> 0) & 1
s3 = (result["s3"] >> 0) & 1
cout = (result["cout"] >> 0) & 1

print(f"Test vector 0: s={s3}{s2}{s1}{s0}, cout={cout}")
print(f"Total bitwise operations: {interp.op_count}")
```

---

## .bs Program Details

A generated `.bs` program has three sections: input declarations, output declarations,
and gate logic. The code is entirely straight-line: no loops, no arrays, no
conditionals. This is characteristic of combinational circuits.

**DSL features used:** AND (`&`), OR (`|`), XOR (`^`), NOT (`~`). No loops, no arrays,
no shifts, no stream addition, no popcount. This benchmark uses the simplest subset of
the DSL.

| Feature | Used? | Notes |
|---------|-------|-------|
| AND     | Yes   | Gate simulation |
| OR      | Yes   | Gate simulation |
| XOR     | Yes   | Gate simulation |
| NOT     | Yes   | NAND/NOR/XNOR decomposition |
| SHIFT   | No    | |
| Params  | No    | |
| Arrays  | No    | |
| Loops   | No    | |
| ADD (+) | No    | |
| Popcount| No    | |

### Conversion pipeline

```
.bench netlist  ->  bench2bs (datasets/bench2bs.py)  ->  .bs program  ->  parse  ->  execute
```

---

## Test Suite (58 tests)

Each netlist is verified by comparing bitstream execution against `simulate_bench()`,
a gate-level reference simulator that evaluates the original `.bench` netlist one
vector at a time.

| Netlist | Tests | Notes |
|---------|-------|-------|
| c17 | 4 | Exhaustive 32 + small/medium/large tiers |
| adder4 | 4 | Exhaustive 512 + small/medium/large tiers |
| c432, c499, c880, c1355, c1908, c2670, c3540, c5315, c6288, c7552 | 5 each | Random 200 + Large 1M (spot-check 200) + small/medium/large tiers |

Total: 4 + 4 + 10 \* 5 = **58 tests**.

The two small netlists (c17, adder4) use exhaustive testing as their unit test;
the other circuits use 200 random vectors verified per-output against
`simulate_bench()`, plus a 1M-vector spot-check. Every netlist additionally has
three precomputed tiers (small/medium/large) sized by `.bsdata` byte count
(~50 MB, ~500 MB, ~5 GB; the `adder4` sanity-check circuit is
smaller at ~35 MB / ~350 MB / ~3.5 GB), generated via `getrandbits()` with the
expected outputs spot-checked against the gate-level reference.

---

## Running

```bash
# Standalone runner
python3 benchmark/apps/circuit_sim/src/run.py

# All 12 netlists via unified framework
python3 benchmark/run_all.py circuit_sim

# List all discovered programs
python3 benchmark/run_all.py --list
```

The unified framework discovers the committed `netlist_*.bs` files in `src/` and
verifies each against the reference simulator. To regenerate them, run
`python benchmark/apps/circuit_sim/datasets/raw/convert.py`.

### Converting your own netlists

You can convert any `.bench` format netlist to a `.bs` bitstream program:

```bash
# Print the .bs program to stdout
python3 benchmark/apps/circuit_sim/datasets/bench2bs.py path/to/circuit.bench

# Save to a file
python3 benchmark/apps/circuit_sim/datasets/bench2bs.py path/to/circuit.bench output.bs
```

### Relevant source files

| File | Purpose |
|------|---------|
| `datasets/bench2bs.py` | Converts `.bench` netlists to `.bs` programs; also provides the `simulate_bench()` reference simulator |
| `benchmark/apps/circuit_sim/src/run.py` | Standalone test harness with `run_generated()` callback |
| `benchmark/apps/circuit_sim/datasets/raw/` | The 12 `.bench` netlist files |
| `benchmark/apps/circuit_sim/datasets/raw/convert.py` | Batch converts `.bench` → `netlist_*.bs` in `src/` |
| `benchmark/run_all.py` | Unified runner for all benchmark domains |

---

## References

- Brglez & Fujiwara, *"Neutral Netlist of Ten Combinational Benchmark Circuits and
  a Target Translator in Fortran"*, ISCAS 1985. The original ISCAS'85 benchmark suite.

- Hansen, Yalcin & Hayes, *"Unveiling the ISCAS-85 Benchmarks: A Case Study in
  Reverse Engineering"*, IEEE Design & Test of Computers, 1999. Source for the
  high-level functional descriptions of each ISCAS'85 circuit (e.g. c432 as a
  27-channel interrupt controller, c7552 as a 34-bit adder plus magnitude
  comparator with input parity checking).

- Chatterjee et al., *"Gate-Level Simulation with GPU Computing"*, ACM TODAES, 2011.
  Bitwise-parallel simulation on GPUs.

- Zhang, Ren, Sridharan, Khailany, *"GATSPI: GPU Accelerated Gate-Level Simulation
  for Power Improvement"*, DAC 2022. Reports a 449x turnaround speedup in a glitch-
  optimization flow and simulation-kernel speedups of up to 1668x (single GPU) and
  7412x (multi-GPU) over a commercial CPU simulator.

- ISCAS'85 benchmark circuits: standard combinational test suite widely used in EDA
  research. University of Toronto ECE archive:
  https://www.eecg.toronto.edu/~ece1767/project/iscas.html

- The **bitwise-parallel** technique: pack `K` test vectors into `K`-bit words. Each
  gate operation (AND, OR, XOR) evaluates the gate on all `K` inputs in a single
  instruction. Standard practice in EDA for fault simulation and design verification.
