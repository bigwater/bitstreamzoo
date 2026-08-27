#!/usr/bin/env python3
"""Precompute .bsdata files for shift_and generated tests.

Handles both required (wide random DNA) and optional (real genome) tests.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter
from benchmark.bsdata import write_bsdata

TESTS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(TESTS_DIR, "../../src")
ALPHABET = "ACGT"


def load_program():
    bs_path = os.path.join(SRC_DIR, "shift_and.bs")
    with open(bs_path) as f:
        source = f.read()
    program = parse(source)
    return program


def build_mask_arrays(pattern):
    """Build pattern mask arrays for the .bs program."""
    M = len(pattern)
    mask_arrays = {}
    for c in ALPHABET:
        arr = {}
        for j in range(M):
            arr[j] = -1 if pattern[j] == c else 0
        mask_arrays[c] = arr
    return mask_arrays


def precompute_wide(program, N, pattern_len, seed_text, seed_pat):
    """Precompute wide random DNA test."""
    M = pattern_len
    mask = (1 << N) - 1

    # Generate random pattern
    pat_rng = random.Random(seed_pat)
    pattern = "".join(pat_rng.choice(ALPHABET) for _ in range(M))

    # Generate random DNA directly as basis bitstreams
    rng = random.Random(seed_text)
    b0 = rng.getrandbits(N)
    b1 = rng.getrandbits(N)
    basis = {
        "A": (~b0 & ~b1) & mask,
        "C": (~b0 & b1) & mask,
        "G": (b0 & ~b1) & mask,
        "T": (b0 & b1) & mask,
    }

    mask_arrays = build_mask_arrays(pattern)

    print(f"  Running Shift-And wide (N={N}, M={M})...")
    interp = Interpreter()
    result = interp.run(
        program,
        inputs={"is_A": basis["A"], "is_C": basis["C"],
                "is_G": basis["G"], "is_T": basis["T"]},
        params={"M": M},
        input_arrays={"mask_A": mask_arrays["A"], "mask_C": mask_arrays["C"],
                      "mask_G": mask_arrays["G"], "mask_T": mask_arrays["T"]},
    )

    bsdata_file = f"shift_and_wide_{N}.bsdata"
    bsdata_path = os.path.join(TESTS_DIR, bsdata_file)
    write_bsdata(bsdata_path, N,
                 inputs={"is_A": basis["A"], "is_C": basis["C"],
                         "is_G": basis["G"], "is_T": basis["T"]},
                 params={"M": M},
                 input_arrays={"mask_A": mask_arrays["A"], "mask_C": mask_arrays["C"],
                               "mask_G": mask_arrays["G"], "mask_T": mask_arrays["T"]},
                 expected={"matches": result["matches"] & mask})

    size_mb = os.path.getsize(bsdata_path) / 1e6
    print(f"  Wrote {bsdata_file} ({size_mb:.1f} MB)")
    return bsdata_file


def precompute_file(program, dataset, pattern):
    """Precompute real genome DNA test."""
    sys.path.insert(0, SRC_DIR)
    from run import load_reallife_dna

    try:
        basis, seq_len = load_reallife_dna(dataset)
    except (FileNotFoundError, ImportError):
        print(f"  [skip] {dataset} not available")
        return None

    M = len(pattern)
    mask_arrays = build_mask_arrays(pattern)
    mask = (1 << seq_len) - 1

    print(f"  Running Shift-And file ({dataset}, {seq_len} bp, pattern={pattern})...")
    interp = Interpreter()
    result = interp.run(
        program,
        inputs={"is_A": basis["A"], "is_C": basis["C"],
                "is_G": basis["G"], "is_T": basis["T"]},
        params={"M": M},
        input_arrays={"mask_A": mask_arrays["A"], "mask_C": mask_arrays["C"],
                      "mask_G": mask_arrays["G"], "mask_T": mask_arrays["T"]},
    )

    bsdata_file = f"shift_and_{dataset}.bsdata"
    bsdata_path = os.path.join(TESTS_DIR, bsdata_file)
    write_bsdata(bsdata_path, seq_len,
                 inputs={"is_A": basis["A"], "is_C": basis["C"],
                         "is_G": basis["G"], "is_T": basis["T"]},
                 params={"M": M},
                 input_arrays={"mask_A": mask_arrays["A"], "mask_C": mask_arrays["C"],
                               "mask_G": mask_arrays["G"], "mask_T": mask_arrays["T"]},
                 expected={"matches": result["matches"] & mask})

    size_mb = os.path.getsize(bsdata_path) / 1e6
    print(f"  Wrote {bsdata_file} ({size_mb:.1f} MB)")
    return bsdata_file


def main():
    program = load_program()

    tests_json_path = os.path.join(TESTS_DIR, "tests.json")
    with open(tests_json_path) as f:
        tests = json.load(f)

    for case in tests:
        if case.get("category") != "generated":
            continue
        g = case["generate"]
        gen_type = g.get("type", "wide")

        if gen_type == "wide":
            bsdata_file = precompute_wide(
                program, N=g["N"], pattern_len=g["pattern_len"],
                seed_text=g["seed_text"], seed_pat=g["seed_pat"])
        elif gen_type == "file":
            bsdata_file = precompute_file(
                program, dataset=g["dataset"], pattern=g["pattern"])
        else:
            continue

        if bsdata_file:
            case["data_file"] = bsdata_file

    with open(tests_json_path, "w") as f:
        json.dump(tests, f, indent=2)
    print("Updated tests.json")


if __name__ == "__main__":
    main()
