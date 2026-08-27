#!/usr/bin/env python3
"""Precompute .bsdata files for edit_distance generated tests."""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter
from benchmark.bsdata import write_bsdata
from benchmark.apps.edit_distance.src.run import (
    encode_text_columns, encode_pattern, encode_dist_init,
    decode_distance, myers_reference, DNA,
)

TESTS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(TESTS_DIR, "../../src")


def load_program():
    bs_path = os.path.join(SRC_DIR, "edit_distance.bs")
    with open(bs_path) as f:
        source = f.read()
    return parse(source)


def precompute_wide(program, K, N, M, seed):
    """Precompute a wide edit distance test case.

    Replicates the exact random generation from run_edit_distance_wide in run.py.
    """
    rng = random.Random(seed)
    B = max(N, M).bit_length()

    # Generate random pattern (same as run.py)
    pattern = ''.join(rng.choice(DNA) for _ in range(M))

    # Generate K random texts column-by-column (same as run.py)
    pos_arrays = {f"pos_{ch}": {} for ch in DNA}
    for j in range(N):
        a_bits = 0
        c_bits = 0
        g_bits = 0
        t_bits = 0
        for k in range(K):
            ch = rng.choice(DNA)
            if ch == 'A':
                a_bits |= 1 << k
            elif ch == 'C':
                c_bits |= 1 << k
            elif ch == 'G':
                g_bits |= 1 << k
            else:
                t_bits |= 1 << k
        pos_arrays["pos_A"][j] = a_bits
        pos_arrays["pos_C"][j] = c_bits
        pos_arrays["pos_G"][j] = g_bits
        pos_arrays["pos_T"][j] = t_bits

    # Encode pattern (broadcast)
    eq_arrays = encode_pattern(pattern)

    # Encode dist_init
    dist_init = encode_dist_init(M, B)

    # Merge all input arrays
    input_arrays = {}
    for key, val in pos_arrays.items():
        input_arrays[key] = val
    for key, val in eq_arrays.items():
        input_arrays[key] = val
    input_arrays["dist_init"] = dist_init

    print(f"  Running edit_distance (K={K}, N={N}, M={M}, B={B})...")
    interp = Interpreter()
    result = interp.run(
        program, inputs={},
        params={"N": N, "M": M, "B": B},
        input_arrays=input_arrays,
    )
    print(f"  Interpreter done ({interp.op_count} ops)")

    # Spot-check a few distances against reference
    spot_rng = random.Random(seed + 1000)
    n_verify = min(200, K)
    positions = spot_rng.sample(range(K), n_verify)
    for k in positions:
        # Extract text k from pos_arrays
        text = []
        for j in range(N):
            for ch in DNA:
                if (pos_arrays[f"pos_{ch}"][j] >> k) & 1:
                    text.append(ch)
                    break
        text = ''.join(text)
        got_dist = 0
        for b in range(B):
            if (result["dist"].get(b, 0) >> k) & 1:
                got_dist |= 1 << b
        ref_dist = myers_reference(text, pattern)
        assert got_dist == ref_dist, (
            f"Spot-check failed: text {k}, got={got_dist} ref={ref_dist}")
    print(f"  Spot-check passed ({n_verify} texts verified)")

    # Write .bsdata
    bsdata_file = f"edit_distance_wide_{K}.bsdata"
    bsdata_path = os.path.join(TESTS_DIR, bsdata_file)
    write_bsdata(bsdata_path, K,
                 params={"N": N, "M": M, "B": B},
                 input_arrays=input_arrays,
                 expected={"dist": result["dist"]})

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
        print(f"Precomputing: {case['name']}")
        bsdata_file = precompute_wide(program,
                                      K=g["K"], N=g["N"], M=g["M"],
                                      seed=g["seed"])
        case["data_file"] = bsdata_file

    with open(tests_json_path, "w") as f:
        json.dump(tests, f, indent=2)
    print("Updated tests.json")


if __name__ == "__main__":
    main()
