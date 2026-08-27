#!/usr/bin/env python3
"""Generate tiered .npz datasets for circuit_sim.

Input format: one packed bitstream per primary input of each netlist.
The circuits themselves (ISCAS-85 .bench files) are the real data;
the test vectors are random stimuli.

Representative netlists:
  c432:  36 inputs, 7 outputs
  c7552: 207 inputs, 108 outputs

Total input data per netlist = n_inputs * bitlength / 8 bytes.

Tiers (per netlist):
  small:  ~1 MB total input -> c432: N=222K, c7552: N=38K
  medium: ~10 MB total input -> c432: N=2.2M, c7552: N=386K
  large:  ~100 MB total input -> c432: N=22M, c7552: N=3.8M

Each .npz contains:
  - One array per primary input name, packed as uint8 (little-endian bits)
  - n_vectors: number of test vectors
  - netlist: name string
  - source: provenance string

Usage:
    python make_data.py                   # Generate all tiers
    python make_data.py --tier small      # Generate one tier
"""

import argparse
import os
import sys

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
NETLISTS_DIR = os.path.join(DIR, "raw")

sys.path.insert(0, os.path.join(DIR, ".."))
sys.path.insert(0, os.path.join(DIR, "..", "..", "..", ".."))

# Representative netlists and their input counts
NETLISTS = {
    "c432": {"bench": "c432.bench", "n_inputs": 36},
    "c7552": {"bench": "c7552.bench", "n_inputs": 207},
}

# Target ~1 MB / ~10 MB / ~100 MB of total input data per netlist.
# Total = n_inputs * bitlength / 8 bytes.
# bitlength = target_bytes * 8 / n_inputs
TARGET_BYTES = {
    "small":  1_000_000,
    "medium": 10_000_000,
    "large":  100_000_000,
}


def parse_bench_inputs(bench_path: str) -> list[str]:
    """Parse INPUT declarations from a .bench file."""
    import re
    inputs = []
    with open(bench_path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"INPUT\s*\(\s*(\w+)\s*\)", line)
            if m:
                inputs.append(m.group(1))
    return inputs


def generate_netlist_tier(netlist_name: str, tier: str, seed: int = 42):
    """Generate one tier of data for one netlist."""
    info = NETLISTS[netlist_name]
    bench_path = os.path.join(NETLISTS_DIR, info["bench"])

    if not os.path.isfile(bench_path):
        print(f"  [{tier}/{netlist_name}] SKIP: bench file not found: {bench_path}")
        return

    # Parse actual input names from the bench file
    input_names = parse_bench_inputs(bench_path)
    n_inputs = len(input_names)

    target_bytes = TARGET_BYTES[tier]
    bitlength = (target_bytes * 8) // n_inputs

    out_dir = os.path.join(DIR, tier)
    out_path = os.path.join(out_dir, f"{netlist_name}_vectors.npz")

    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"  [{tier}/{netlist_name}] Already exists: {out_path} ({size:,} bytes) -- skipping")
        return

    print(f"  [{tier}/{netlist_name}] Generating {bitlength:,} random test vectors "
          f"({n_inputs} inputs)...")

    rng = np.random.RandomState(seed)
    n_bytes = (bitlength + 7) // 8

    arrays = {}
    for name in input_names:
        # Generate random packed bits
        packed = rng.randint(0, 256, size=n_bytes, dtype=np.uint8)
        # Mask off unused bits in last byte
        remainder = bitlength % 8
        if remainder > 0:
            packed[-1] &= (1 << remainder) - 1
        arrays[name] = packed

    # Key name matches the committed *_vectors.npz artifacts (n_vectors).
    arrays["n_vectors"] = np.array(bitlength, dtype=np.int64)
    arrays["netlist"] = np.array(netlist_name)
    arrays["source"] = np.array(
        f"ISCAS-85 {netlist_name} ({n_inputs} inputs), "
        f"{bitlength:,} random vectors (seed={seed})"
    )

    os.makedirs(out_dir, exist_ok=True)
    np.savez(out_path, **arrays)
    size = os.path.getsize(out_path)
    total_input = n_inputs * n_bytes
    print(f"  [{tier}/{netlist_name}] Wrote {out_path} ({size:,} bytes)")
    print(f"  [{tier}/{netlist_name}] Input data: {n_inputs} x {n_bytes:,} bytes = "
          f"{total_input:,} bytes ({total_input / 1e6:.1f} MB)")


def generate_tier(tier: str):
    """Generate one tier for all representative netlists."""
    for netlist_name in NETLISTS:
        generate_netlist_tier(netlist_name, tier)


def main():
    parser = argparse.ArgumentParser(description="Generate circuit_sim tiered datasets")
    parser.add_argument("--tier", choices=["small", "medium", "large"],
                        help="Generate only this tier (default: all)")
    args = parser.parse_args()

    print("circuit_sim: generating tiered datasets from ISCAS-85 netlists")
    print(f"  Netlists dir: {NETLISTS_DIR}")
    print()

    tiers = [args.tier] if args.tier else ["small", "medium", "large"]
    for tier in tiers:
        generate_tier(tier)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
