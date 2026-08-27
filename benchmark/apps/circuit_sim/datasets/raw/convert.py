#!/usr/bin/env python3
"""Convert ISCAS'85 .bench netlists to .bs bitstream programs.

Converts each .bench netlist in this directory to a .bs file in src/,
with the naming convention netlist_<name>.bs. This is the standard
conversion step (like other domains' convert.py), run by build.py.

Usage:
    python3 convert.py                    # Convert all .bench files
    python3 convert.py c432.bench         # Convert a single file
"""

import os
import sys
import glob

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(DIR, "..", "..", "..", "..", "..")
SRC_DIR = os.path.join(DIR, "..", "..", "src")
sys.path.insert(0, ROOT)

from benchmark.apps.circuit_sim.datasets.bench2bs import convert as bench2bs


def convert_bench_file(bench_path: str) -> str:
    """Convert a .bench netlist to a netlist_<name>.bs file in src/.

    Returns the output path.
    """
    with open(bench_path) as f:
        bench_text = f.read()
    bs_text = bench2bs(bench_text)

    os.makedirs(SRC_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(bench_path))[0]
    out_path = os.path.join(SRC_DIR, f"netlist_{name}.bs")
    with open(out_path, "w") as f:
        f.write(bs_text)
    return out_path


def convert_all() -> int:
    """Convert all .bench files in this directory to .bs in src/."""
    bench_files = sorted(glob.glob(os.path.join(DIR, "*.bench")))
    if not bench_files:
        print("No .bench files found in", DIR)
        return 0
    for path in bench_files:
        out = convert_bench_file(path)
        print(f"  {os.path.basename(path)} -> {os.path.relpath(out, ROOT)}")
    return len(bench_files)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = os.path.join(DIR, sys.argv[1])
        out = convert_bench_file(path)
        print(f"  {sys.argv[1]} -> {os.path.relpath(out, ROOT)}")
    else:
        n = convert_all()
        print(f"Converted {n} netlists.")
