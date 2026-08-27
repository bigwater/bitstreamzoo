# BitstreamZoo

A benchmark suite of **bitstream programs** — a DSL operating on
unbounded bitsets with bitwise operations (AND, OR, XOR, NOT, SHIFT),
stream addition, and popcount — drawn from circuit simulation,
bitsliced cryptography, regex matching, string matching, JSON
structural indexing, BNN inference, error-correcting codes, database
column scans, and bioinformatics.

The suite ships **15 workload families (30 `.bs` programs)** and three
reference backends that all execute the same programs:

- Python reference interpreter (`simulator/pythonsim/`)
- C++ AVX-512 SIMD backend (`simulator/csim/`, target `bsim`)
- CUDA backend (`simulator/csim/`, target `bsim_cuda`)

The programs are *algorithm kernels*; the standard input sizes are
synthetic stress data, and some families take an optional real-world
input as an extra correctness check.

## Quick start

Requires Python 3.10+. The unit suite needs nothing else.

```bash
# Unit-correctness suite: committed fixtures only, no build step.
# Passes from a clean clone.
python benchmark/run_all.py --unit

# Optional: performance runs over ~50MB/500MB/5GB generated inputs
# (numpy required; the regex tiers additionally need the C++ backend
# built first — see below — because their expected outputs are
# computed by bsim)
pip install numpy
python build.py --generate-only --tier small
python benchmark/run_all.py --size small
```

Example program — the MatchStar regex cursor-extension primitive:

```
input stream M      // cursor positions
input stream C      // character class positions
output stream result

stream _t0 = M & C
stream _t1 = _t0 + C
stream _t2 = _t1 ^ C
result = _t2 | M
```

Each bit position is a parallel work-item; the DSL descends from
[Parabix](https://github.com/parabix). Full spec in [`DSL.md`](DSL.md).

## Workload families

Circuit Sim (11 ISCAS'85 netlists + adder), Shift-And (DNA exact
match), Trivium, AES S-box, Regex (icgrep-compiled Snort patterns,
271–1.2M ops), Epistasis, GEMV, Mison (JSON index), BitFunnel,
BNN (FINN MVTU), Edit Distance (Myers), SIMON, Polar encoder,
BitWeaving/V, Montgomery multiplication.

Every family directory has a `readme.md` with the algorithm, source
citation, op counts, and a walkthrough — see `benchmark/apps/<family>/`.

## C++ / CUDA backends

```bash
cmake -B simulator/csim/build simulator/csim
cmake --build simulator/csim/build -j
python benchmark/run_all.py --unit --backend cpp
python benchmark/run_all.py --unit --backend cuda
```

Prerequisites: CMake 3.24+, a C++17 compiler with OpenMP, an
AVX-512-capable CPU (the SIMD kernels are compiled with explicit
AVX-512 flags), and network access at configure time (nlohmann/json is
fetched via FetchContent). CUDA is optional and auto-detected (pass
`-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` if detection misses
it). Note for AMD Zen4 CPUs running medium/large inputs by hand:
`export GLIBC_TUNABLES=glibc.cpu.x86_rep_stosb_threshold=2000000000`
(a hardware `rep stosb` erratum silently corrupts large memsets;
`benchmark/base.py` sets this automatically for every backend
subprocess it launches).

## Layout and documentation

```
simulator/pythonsim/   lexer, parser, reference interpreter
simulator/csim/        C++ AVX-512 + CUDA backends (CMake)
benchmark/             test runner, .bsdata format, apps/<family>/
build.py               dataset download/convert/generate pipeline
DSL.md                 DSL specification
```

- [`benchmark/README.md`](benchmark/README.md) — runner flags, the
  `.bsdata` test-data format, dataset tiers, adding a new family
- [`benchmark/apps/<family>/readme.md`](benchmark/apps/) — per-family
  algorithm docs and data provenance

## License

Suite code, DSL, backends, and synthetic data: MIT (see
[`LICENSE`](LICENSE)). Bundled third-party-derived inputs keep their
upstream licenses: the ISCAS-85 netlists (long-standing academic
benchmark), the NIST AES S-box circuit (public domain), the Snort
pattern set via AutomataZoo (`snort.regex`, `pattern_manifest.json`,
and the `regex_*.bs` programs compiled from those patterns — GPLv2,
redistributed with attribution to Snort and AutomataZoo), committed
extracts of human chr1 (UCSC hg38, free for research) and the WRCCDC
2012 capture (publicly released for research), and MNIST fixtures.
GH Archive data (Mison) is never redistributed — it contains real
usernames and emails — and is fetched and converted locally by
`build.py`. Per-family `datasets/raw/readme.md` files record exact
provenance.
