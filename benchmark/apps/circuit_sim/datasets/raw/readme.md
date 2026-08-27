# Circuit Simulation -- Real-Life Data Provenance

## Source

ISCAS'85 combinational benchmark circuits in `.bench` netlist format.

## Data files

The `.bench` files already reside in this directory (12 netlists: c17, c432,
c499, c880, c1355, c1908, c2670, c3540, c5315, c6288, c7552, adder4).
See `netlists_readme.md` for full details.

## Origin

- **c17 through c7552**: Downloaded from the University of Toronto ECE1767
  course page: https://www.eecg.toronto.edu/~ece1767/project/iscas.html
- **adder4**: Hand-written 4-bit ripple-carry adder.
- Original reference: Brglez & Fujiwara, "A Neutral Netlist of 10 Combinational
  Benchmark Circuits and a Target Translator in FORTRAN", Proceedings of the IEEE
  International Symposium on Circuits and Systems (ISCAS), pp. 663–698, 1985.

## Conversion

The `.bench` files are converted to `.bs` netlist programs by
`datasets/bench2bs.py`. `datasets/raw/convert.py` is the batch driver: for
each `.bench` it calls `bench2bs` and writes the result to
`src/netlist_<name>.bs`.

## Download

The ISCAS'85 `.bench` files are already committed. For additional circuits:

```bash
# EPFL combinational benchmark suite (BLIF/Verilog, needs ABC to convert to .bench)
git clone https://github.com/lsils/benchmarks epfl_benchmarks

# ISCAS'85 originals (if re-downloading)
wget https://www.eecg.toronto.edu/~ece1767/project/circuits/c17.bench
wget https://www.eecg.toronto.edu/~ece1767/project/circuits/c432.bench
wget https://www.eecg.toronto.edu/~ece1767/project/circuits/c499.bench
wget https://www.eecg.toronto.edu/~ece1767/project/circuits/c880.bench
wget https://www.eecg.toronto.edu/~ece1767/project/circuits/c1355.bench
wget https://www.eecg.toronto.edu/~ece1767/project/circuits/c1908.bench
wget https://www.eecg.toronto.edu/~ece1767/project/circuits/c2670.bench
wget https://www.eecg.toronto.edu/~ece1767/project/circuits/c3540.bench
wget https://www.eecg.toronto.edu/~ece1767/project/circuits/c5315.bench
wget https://www.eecg.toronto.edu/~ece1767/project/circuits/c6288.bench
wget https://www.eecg.toronto.edu/~ece1767/project/circuits/c7552.bench
```
