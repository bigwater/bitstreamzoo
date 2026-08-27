# Benchmark suite

Each subdirectory of `apps/` is one benchmark domain. Each domain has:

- `src/*.bs` — one or more bitstream programs
- `src/run.py` — standalone test harness
- `datasets/tests/` — `tests.json` metadata + `.bsdata` test data +
  `generate_tests.py`
- `datasets/raw/convert.py`, `datasets/{small,medium,large}/` —
  real-corpus domains only: `convert.py` turns a raw corpus into `.npz`
  bit-plane arrays under those tier dirs, consumed at test time via
  `run_generated()`. Synthetic domains have neither; their inputs are
  generated directly into `.bsdata` by `generate_tests.py`. (The
  `.bsdata` files in `tests/` are always the canonical test inputs.)
- `readme.md` — algorithm explanation + source citation

Domains are auto-discovered from the filesystem; no manual registration
is needed when adding a new one.

## Running tests

```bash
python benchmark/run_all.py                   # all tests must pass
python benchmark/run_all.py <domain>          # one domain
python benchmark/run_all.py --backend cpp     # C++ SIMD backend (fastest)
python benchmark/run_all.py --backend all     # python, cpp, and cuda
python benchmark/run_all.py --size small      # ~50MB per test
python benchmark/run_all.py --size medium     # ~500MB per test
python benchmark/run_all.py --size large      # ~5GB per test (HPC only)
python benchmark/run_all.py --list            # list all programs
```

`run_all.py` defaults to the Python backend for correctness comparison.
Use `--backend cpp` for real performance runs.

## Test data format (.bsdata)

All test data uses the unified `.bsdata` binary format:

- `tests.json` is metadata-only: `name`, `bitlength`, `data_file`, plus
  optional per-test flags
- `.bsdata` files contain inputs, params, and expected outputs in a
  single binary blob
- `generate_tests.py` scripts produce `.bsdata` files via
  `write_bsdata()` from `benchmark/bsdata.py`
- C++/CUDA binaries load `.bsdata` directly via `--input` (no stdin JSON
  piping)
- Python backend reads `.bsdata` via `read_bsdata()`
- Inspect any `.bsdata` file with `python -m benchmark.bsdata <file>`

## Dataset tiers

Three tiers, defined by total binary data size (input + output streams):

| Tier     | Target size per test | Filter range     |
| -------- | -------------------- | ---------------- |
| `small`  | ~50 MB               | 25 MB – 250 MB   |
| `medium` | ~500 MB              | 250 MB – 2.5 GB  |
| `large`  | ~5 GB                | > 2.5 GB         |

`bitlength = target_bytes × 8 / n_binary_streams`

- **Per-program tier table:** `benchmark/tier_config.py`
  (`TIER_BITLENGTH` dict).
- **Shared tier-generation helpers:** `benchmark/tier_generate.py`
  (CLI parsing, provenance, SHA-256).
- **Test entries in `tests.json`** include `size_bytes` and `provenance`
  fields. Filtering is by file size. Without `--size`,
  every test whose data is present runs; tier datasets that have not
  been generated are skipped with an up-front notice.

Generate tier data for a domain:

```bash
python benchmark/apps/<domain>/datasets/tests/generate_tests.py --tier small
python benchmark/apps/<domain>/datasets/tests/generate_tests.py --describe
python benchmark/apps/<domain>/datasets/tests/generate_tests.py --verify
```

The standard tiers are **synthetic stress data**: random streams under
a fixed RNG seed, recorded with a `provenance` block in `tests.json`.
Separately, a few domains accept an **optional real-world input** as an
extra correctness check (it never replaces the synthetic tiers):
shift_and (human chr1, hg38), mison (GH Archive JSON), and regex
(WRCCDC 2012 pcap). Small extracts of chr1 and the WRCCDC capture are
committed (`datasets/small/*.npz`) and run from a clean checkout when
numpy is present; everything larger — and all Mison real data, which
contains real GitHub usernames/emails and is not redistributed — is
fetched and derived locally by `python build.py <domain>`. bnn ships
committed MNIST fixtures (unit-scale accuracy only). Tests backed by
missing downloads are optional and skipped silently.

## Adding a new domain

1. Create `benchmark/apps/<name>/` mirroring an existing domain's
   layout.
2. Write `.bs` files in `src/`.
3. Create `datasets/tests/generate_tests.py` that uses `write_bsdata()`
   to emit `.bsdata` test data.
4. Add an entry to `benchmark/tier_config.py` (`TIER_BITLENGTH`) if the
   program should be part of the tiered benchmark pipeline.
5. Write `readme.md` in the domain directory documenting the algorithm
   and citing its source.

No registration step is required beyond the above — domains are
auto-discovered.
