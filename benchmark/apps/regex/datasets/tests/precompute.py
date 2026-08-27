#!/usr/bin/env python3
"""Precompute .bsdata files for regex generated tests (WRCCDC pcap data)."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter
from benchmark.bsdata import write_bsdata
from benchmark.apps.regex.src.run import load_reallife_npz

TESTS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(TESTS_DIR, "../../src")


def precompute_file(program, dataset, prog_name):
    """Precompute a file-based regex test."""
    b, payload_length = load_reallife_npz(dataset)

    print(f"  Running {prog_name} on {dataset} ({payload_length} bytes)...")
    interp = Interpreter()
    result = interp.run(program, inputs={}, input_arrays={'b': b})

    # Mask outputs to payload_length bits
    mask = (1 << payload_length) - 1
    expected = {k: v & mask for k, v in result.items()}

    bsdata_file = f"{prog_name}_{dataset}.bsdata"
    bsdata_path = os.path.join(TESTS_DIR, bsdata_file)
    write_bsdata(bsdata_path, payload_length,
                 input_arrays={"b": b},
                 expected=expected)

    size_mb = os.path.getsize(bsdata_path) / 1e6
    print(f"  Wrote {bsdata_file} ({size_mb:.1f} MB)")
    return bsdata_file


def main():
    tests_json_path = os.path.join(TESTS_DIR, "tests.json")
    with open(tests_json_path) as f:
        all_tests = json.load(f)

    for prog_name, cases in all_tests.items():
        bs_path = os.path.join(SRC_DIR, f"{prog_name}.bs")
        with open(bs_path) as f:
            source = f.read()
        program = parse(source)

        for case in cases:
            if case.get("category") != "generated":
                continue
            g = case.get("generate", {})
            if g.get("type") != "file":
                continue

            dataset = g["dataset"]
            bsdata_file = precompute_file(program, dataset, prog_name)
            case["data_file"] = bsdata_file

    with open(tests_json_path, "w") as f:
        json.dump(all_tests, f, indent=2)
    print("Updated tests.json")


if __name__ == "__main__":
    main()
