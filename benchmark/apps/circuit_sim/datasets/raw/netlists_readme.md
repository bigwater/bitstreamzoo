# Circuit Netlists (.bench format)

Gate-level netlists in ISCAS `.bench` format, for conversion to `.bs` via
`datasets/bench2bs.py`.

## Included netlists

All ISCAS'85 circuits downloaded from the University of Toronto ECE archive:
https://www.eecg.toronto.edu/~ece1767/project/iscas.html

| File | Description | Inputs | Outputs | .bs instructions |
|------|-------------|--------|---------|-----------------|
| `c17.bench` | Tiny example | 5 | 2 | 12 |
| `c432.bench` | Priority decoder | 36 | 7 | 440 |
| `c499.bench` | ECAT | 41 | 32 | 974 |
| `c880.bench` | ALU / control | 60 | 26 | 583 |
| `c1355.bench` | ECAT | 41 | 32 | 1006 |
| `c1908.bench` | ECAT | 33 | 25 | 1434 |
| `c2670.bench` | ALU / control | 157 | 64 | 1666 |
| `c3540.bench` | ALU / control | 50 | 22 | 2349 |
| `c5315.bench` | ALU | 178 | 123 | 3454 |
| `c6288.bench` | 16x16 multiplier | 32 | 32 | 4544 |
| `c7552.bench` | ALU / control | 207 | 108 | 5124 |
| `adder4.bench` | 4-bit ripple-carry adder (hand-written) | 9 | 5 | 20 |

## Sources

**c17 through c7552** — ISCAS'85 combinational benchmark suite (Brglez &
Fujiwara, 1985). Downloaded from the University of Toronto ECE1767 course page
(`https://www.eecg.toronto.edu/~ece1767/project/circuits/<name>.bench`).
Original reference: *A Neutral Netlist of 10 Combinational Benchmark Circuits
and a Target Translator in FORTRAN*, IEEE International Symposium on Circuits
and Systems (ISCAS), pp. 663–698, 1985.

**adder4.bench** — Hand-written 4-bit ripple-carry adder using XOR, AND,
and OR gates. Verifiable against Python `a + b + cin`.

## Additional benchmark sources

### EPFL combinational benchmark suite (modern, larger)

Hosted on GitHub: https://github.com/lsils/benchmarks

Includes arithmetic (adders, multipliers, dividers), random/control circuits,
and optimized versions. Circuits range from ~100 to ~170k gates.
Available in Verilog, BLIF, and AIG formats (convert via ABC or Yosys to
`.bench`).

## How to convert .bench to .bs and run

### Step 1: Convert a .bench netlist to .bs

```bash
# Print .bs to stdout
python3 benchmark/apps/circuit_sim/datasets/bench2bs.py benchmark/apps/circuit_sim/datasets/raw/c17.bench

# Or save to a file
python3 benchmark/apps/circuit_sim/datasets/bench2bs.py benchmark/apps/circuit_sim/datasets/raw/c432.bench benchmark/apps/circuit_sim/src/netlist_c432.bs
```

The converter handles:
- Gate types: AND, OR, NAND, NOR, XOR, XNOR, NOT, BUF
- NAND/NOR/XNOR decomposition (e.g. `NAND(a,b)` → `AND` + `NOT`)
- Multi-input gates (e.g. `AND(a,b,c,d)`) → chain of binary ops
- Topological sorting (gates emitted in dependency order)
- Wire name sanitization (numeric names get `n` prefix)

### Step 2: Use from Python (as a library)

```python
from benchmark.apps.circuit_sim.datasets.bench2bs import convert as bench2bs
from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter

# Convert
with open("benchmark/apps/circuit_sim/datasets/raw/c432.bench") as f:
    bs_source = bench2bs(f.read())

# Parse
program = parse(bs_source)

# Run — each bit position in a stream is a different test vector
interp = Interpreter()
result = interp.run(program, inputs={"G1": 0b101, "G2": 0b110, ...})
# result["G426"] etc. are output bitstreams
```

### Step 3: Run the automated tests

```bash
python benchmark/apps/circuit_sim/src/run.py
```

This runs exhaustive verification of c17 (32 input combos) and adder4 (512
combos). Use `python benchmark/run_all.py circuit_sim` to test all 12 netlists.

### Example: what the converter produces

**Input** (`c17.bench`):
```
INPUT(G1)
INPUT(G2)
...
G8 = NAND(G1,G3)
G16 = NAND(G8,G12)
```

**Output** (`.bs`):
```
input stream G1
input stream G2
...
output stream G16
output stream G17

stream _t_G8 = G1 & G3
stream G8 = ~_t_G8
...
G16 = ~_t_G16
```

Each `NAND(a,b)` becomes an AND followed by a NOT. Output wires are declared
at the top and assigned without the `stream` keyword (since they're already
declared as `output stream`).
