#!/usr/bin/env python3
"""Precompute .bsdata files for simon generated tests."""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter
from benchmark.bsdata import write_bsdata
from benchmark.apps.simon.src.run import (
    simon_key_schedule, encode_round_keys,
)

TESTS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(TESTS_DIR, "../../src")


def load_program():
    bs_path = os.path.join(SRC_DIR, "simon.bs")
    with open(bs_path) as f:
        source = f.read()
    program = parse(source)
    return program


def precompute_wide(program, W, seed):
    """Precompute wide SIMON test."""
    rng = random.Random(seed)

    # Random key (must match run_simon_wide RNG sequence exactly)
    key = [rng.randint(0, 0xFFFF) for _ in range(4)]
    round_keys = simon_key_schedule(key)

    # Generate W-bit random streams for plaintext
    plainL_arrays = {b: rng.getrandbits(W) for b in range(16)}
    plainR_arrays = {b: rng.getrandbits(W) for b in range(16)}

    # Encode round keys
    rk_arrays = encode_round_keys(round_keys)

    print(f"  Running SIMON wide (W={W}, seed={seed})...")
    interp = Interpreter()
    result = interp.run(
        program, inputs={}, params={},
        input_arrays={"plainL": plainL_arrays, "plainR": plainR_arrays,
                      "round_key": rk_arrays},
    )

    bsdata_file = f"simon_wide_{W}.bsdata"
    bsdata_path = os.path.join(TESTS_DIR, bsdata_file)
    write_bsdata(bsdata_path, W,
                 input_arrays={"plainL": plainL_arrays,
                               "plainR": plainR_arrays,
                               "round_key": rk_arrays},
                 expected={"cipherL": result["cipherL"],
                           "cipherR": result["cipherR"]})

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
        bsdata_file = precompute_wide(program,
                                      W=g["W"], seed=g["seed"])
        case["data_file"] = bsdata_file

    with open(tests_json_path, "w") as f:
        json.dump(tests, f, indent=2)
    print("Updated tests.json")


if __name__ == "__main__":
    main()
