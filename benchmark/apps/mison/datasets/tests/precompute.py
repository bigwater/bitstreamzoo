#!/usr/bin/env python3
"""Precompute .bsdata files for mison generated tests."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter
from benchmark.bsdata import write_bsdata

TESTS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(TESTS_DIR, "../../src")


def load_program():
    bs_path = os.path.join(SRC_DIR, "mison.bs")
    with open(bs_path) as f:
        source = f.read()
    program = parse(source)
    return program


def precompute_file(program, dataset):
    """Precompute real-world JSON Mison test."""
    sys.path.insert(0, SRC_DIR)
    from run import load_reallife_json, make_even_bits, compute_log_w

    try:
        b, text_length = load_reallife_json(dataset)
    except (FileNotFoundError, ImportError):
        print(f"  [skip] {dataset} not available")
        return None

    even_bits = make_even_bits(text_length)
    log_w = compute_log_w(text_length)
    mask = (1 << text_length) - 1

    print(f"  Running Mison file ({dataset}, {text_length} chars)...")
    interp = Interpreter()
    result = interp.run(program,
                        inputs={"even_bits": even_bits},
                        params={"LOG_W": log_w},
                        input_arrays={"b": b})

    bsdata_file = f"mison_{dataset}.bsdata"
    bsdata_path = os.path.join(TESTS_DIR, bsdata_file)
    write_bsdata(bsdata_path, text_length,
                 inputs={"even_bits": even_bits},
                 params={"LOG_W": log_w},
                 input_arrays={"b": b},
                 expected={"real_quote": result["real_quote"] & mask,
                           "str_mask": result["str_mask"] & mask,
                           "structural": result["structural"] & mask})

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
        bsdata_file = precompute_file(program, dataset=g["dataset"])

        if bsdata_file:
            case["data_file"] = bsdata_file

    with open(tests_json_path, "w") as f:
        json.dump(tests, f, indent=2)
    print("Updated tests.json")


if __name__ == "__main__":
    main()
