#!/usr/bin/env python3
"""Precompute .bsdata files for trivium generated tests."""

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


def load_program():
    bs_path = os.path.join(SRC_DIR, "trivium.bs")
    with open(bs_path) as f:
        source = f.read()
    program = parse(source)
    return program


def precompute_wide(program, W, length, seed):
    """Precompute wide Trivium test."""
    rng = random.Random(seed)
    key_arrays = {i: rng.getrandbits(W) for i in range(80)}
    iv_arrays = {i: rng.getrandbits(W) for i in range(80)}

    print(f"  Running Trivium wide (W={W}, L={length})...")
    interp = Interpreter()
    result = interp.run(program, inputs={},
                        params={"L": length},
                        input_arrays={"key": key_arrays, "iv": iv_arrays})

    bsdata_file = f"trivium_wide_{W}.bsdata"
    bsdata_path = os.path.join(TESTS_DIR, bsdata_file)
    write_bsdata(bsdata_path, W,
                 params={"L": length},
                 input_arrays={"key": key_arrays, "iv": iv_arrays},
                 expected={"z": result["z"]})

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
                                      W=g["W"], length=g["length"],
                                      seed=g["seed"])
        case["data_file"] = bsdata_file

    with open(tests_json_path, "w") as f:
        json.dump(tests, f, indent=2)
    print("Updated tests.json")


if __name__ == "__main__":
    main()
