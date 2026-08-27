#!/usr/bin/env python3
"""Generate precomputed bitstream-level test data for AES S-box.

Usage:
    python generate_tests.py                 # unit tests + Large 1M (default)
    python generate_tests.py --tier small    # 25M vectors
    python generate_tests.py --tier medium   # 250M vectors
    python generate_tests.py --tier large    # 2.5B vectors
    python generate_tests.py --tier all      # all three tiers
    python generate_tests.py --describe      # print tier info and exit
    python generate_tests.py --verify        # verify existing .bsdata files
"""

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter
from benchmark.apps.aes_sbox.src.run import SBOX, run_aes_sbox_bs
from benchmark.bsdata import write_bsdata, read_bsdata
from benchmark.tier_generate import (
    getrandbits_large, file_sha256, make_provenance, tier_test_entry,
)

TESTS_DIR = os.path.dirname(__file__)
BS_PATH = os.path.join(TESTS_DIR, "../../src/aes_sbox.bs")

TIER_CONFIGS = {
    "small":  {"bitlength":   25_000_000, "seed": 1000, "label": "Small 25M bytes"},
    "medium": {"bitlength":  250_000_000, "seed": 1001, "label": "Medium 250M bytes"},
    "large":  {"bitlength": 2_500_000_000, "seed": 1002, "label": "Large 2.5B bytes"},
}


def bitslice_bytes(input_bytes):
    """Bit-slice a list of bytes into 8 streams (U[0]=MSB, U[7]=LSB).

    Returns dict {0: stream0, 1: stream1, ..., 7: stream7}.
    """
    K = len(input_bytes)
    u_arrays = {}
    for i in range(8):
        bits = 0
        for j in range(K):
            if (input_bytes[j] >> (7 - i)) & 1:
                bits |= 1 << j
        u_arrays[i] = bits
    return u_arrays


def run_bs(program, input_bytes):
    """Run interpreter on bit-sliced input bytes, return S array dict."""
    u_arrays = bitslice_bytes(input_bytes)
    interp = Interpreter()
    result = interp.run(program, inputs={}, params={},
                        input_arrays={"U": u_arrays})
    return result["S"], u_arrays


def verify_sbox(input_bytes, s_array):
    """Verify output S array matches SBOX lookup for all input bytes."""
    K = len(input_bytes)
    for j in range(K):
        out_byte = 0
        for i in range(8):
            if s_array.get(i, 0) & (1 << j):
                out_byte |= 1 << (7 - i)
        expected = SBOX[input_bytes[j]]
        assert out_byte == expected, (
            f"byte {j}: in=0x{input_bytes[j]:02x} got=0x{out_byte:02x} "
            f"expected=0x{expected:02x}")


def spot_check_sbox(u_arrays, s_array, bitlength, num_checks=500, seed=None):
    """Spot-check random positions in large streams against SBOX table."""
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(bitlength), min(num_checks, bitlength)))
    for j in indices:
        in_byte = 0
        for i in range(8):
            if (u_arrays[i] >> j) & 1:
                in_byte |= 1 << (7 - i)
        out_byte = 0
        for i in range(8):
            if (s_array.get(i, 0) >> j) & 1:
                out_byte |= 1 << (7 - i)
        expected_byte = SBOX[in_byte]
        assert out_byte == expected_byte, (
            f"spot-check byte {j}: in=0x{in_byte:02x} got=0x{out_byte:02x} "
            f"expected=0x{expected_byte:02x}")
    return len(indices)


def estimate_bsdata_size(bitlength, n_streams=16):
    """Estimate .bsdata file size: n_streams * ceil(bitlength/8) + header."""
    bytes_per_stream = math.ceil(bitlength / 8)
    return n_streams * bytes_per_stream + 256  # 256 bytes for header estimate


def generate_unit_tests(program):
    """Generate the original unit tests (known values, first 64, exhaustive 256, large 1M)."""
    tests = []

    # -- Known values (5 bytes) ----------------------------------------
    known_bytes = [0, 1, 16, 83, 255]
    s_array, u_arrays = run_bs(program, known_bytes)
    verify_sbox(known_bytes, s_array)
    mask5 = (1 << len(known_bytes)) - 1
    write_bsdata(
        os.path.join(TESTS_DIR, "known_values.bsdata"),
        len(known_bytes),
        input_arrays={"U": u_arrays},
        expected={"S": {k: v & mask5 for k, v in s_array.items()}},
    )
    tests.append({
        "name": "Known values",
        "bitlength": len(known_bytes),
        "data_file": "known_values.bsdata",
    })

    # -- First 64 (bytes 0-63) ----------------------------------------
    first64_bytes = list(range(0, 64))
    s_array, u_arrays = run_bs(program, first64_bytes)
    verify_sbox(first64_bytes, s_array)
    mask64 = (1 << 64) - 1
    write_bsdata(
        os.path.join(TESTS_DIR, "first_64.bsdata"),
        64,
        input_arrays={"U": u_arrays},
        expected={"S": {k: v & mask64 for k, v in s_array.items()}},
    )
    tests.append({
        "name": "First 64",
        "bitlength": 64,
        "data_file": "first_64.bsdata",
    })

    # -- Exhaustive 256 (all bytes, stored as data_file) ---------------
    all_bytes = list(range(256))
    s_array, u_arrays = run_bs(program, all_bytes)
    verify_sbox(all_bytes, s_array)
    mask256 = (1 << 256) - 1
    write_bsdata(
        os.path.join(TESTS_DIR, "aes_sbox_exhaustive_256.bsdata"),
        256,
        input_arrays={"U": u_arrays},
        expected={"S": {k: v & mask256 for k, v in s_array.items()}},
    )
    tests.append({
        "name": "Exhaustive 256",
        "bitlength": 256,
        "data_file": "aes_sbox_exhaustive_256.bsdata",
    })

    # -- Large 1M bytes (precomputed, stored as data_file) ---------------
    W = 1000000
    rng = random.Random(42)
    u_arrays_wide = {i: rng.getrandbits(W) for i in range(8)}

    interp = Interpreter()
    result_wide = interp.run(program, inputs={}, params={},
                             input_arrays={"U": u_arrays_wide})
    s_array_wide = result_wide["S"]

    mask_wide = (1 << W) - 1

    # Spot-check 200 random bytes against SBOX table
    spot_check_sbox(u_arrays_wide, s_array_wide, W, num_checks=200, seed=43)

    write_bsdata(
        os.path.join(TESTS_DIR, "aes_sbox_large_1M.bsdata"),
        W,
        input_arrays={"U": u_arrays_wide},
        expected={"S": {k: v & mask_wide for k, v in s_array_wide.items()}},
    )
    tests.append({
        "name": "Large 1M bytes",
        "bitlength": W,
        "data_file": "aes_sbox_large_1M.bsdata",
    })

    return tests


def generate_tier(program, tier_name):
    """Generate a single tier test (small/medium/large).

    Uses random streams directly (getrandbits) for efficiency, runs the
    interpreter to compute expected outputs, and spot-checks against SBOX.
    """
    cfg = TIER_CONFIGS[tier_name]
    W = cfg["bitlength"]
    seed = cfg["seed"]
    label = cfg["label"]
    data_file = f"aes_sbox_{tier_name}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_file)

    print(f"  Generating {label} (bitlength={W:,}, seed={seed})...")

    # Generate 8 random input streams directly
    t0 = time.time()
    rng = random.Random(seed)
    u_arrays = {i: getrandbits_large(rng, W) for i in range(8)}
    t_gen = time.time() - t0
    print(f"    Input generation: {t_gen:.1f}s")

    # Run interpreter
    t0 = time.time()
    interp = Interpreter()
    result = interp.run(program, inputs={}, params={},
                        input_arrays={"U": u_arrays})
    s_array = result["S"]
    t_interp = time.time() - t0
    print(f"    Interpreter: {t_interp:.1f}s ({interp.op_count} ops)")

    # Spot-check 500 random positions
    t0 = time.time()
    n_checked = spot_check_sbox(u_arrays, s_array, W, num_checks=500,
                                seed=seed + 5000)
    t_check = time.time() - t0
    print(f"    Spot-check: {n_checked} positions verified in {t_check:.1f}s")

    # Mask to bitlength bits
    mask = (1 << W) - 1

    # Write .bsdata
    t0 = time.time()
    write_bsdata(
        data_path,
        W,
        input_arrays={"U": u_arrays},
        expected={"S": {k: v & mask for k, v in s_array.items()}},
    )
    t_write = time.time() - t0

    file_size = os.path.getsize(data_path)
    print(f"    Written: {data_path} ({file_size:,} bytes, {t_write:.1f}s)")

    sha = file_sha256(data_path)
    prov = make_provenance(
        source="synthetic",
        seed=seed,
        description=(f"8 random streams via getrandbits({W}), seed={seed}; "
                     f"expected via Python interpreter; "
                     f"spot-checked 500 positions (seed={seed + 5000})"),
        generated_by="generate_tests.py --tier",
    )
    prov["sha256"] = sha

    return tier_test_entry(
        name=label,
        bitlength=W,
        data_file=data_file,
        size_bytes=file_size,
        provenance=prov,
    )


def do_describe():
    """Print tier descriptions and estimated sizes."""
    print("AES S-box tier configurations:")
    print()
    for tier_name, cfg in TIER_CONFIGS.items():
        W = cfg["bitlength"]
        est = estimate_bsdata_size(W)
        data_file = f"aes_sbox_{tier_name}.bsdata"
        data_path = os.path.join(TESTS_DIR, data_file)
        exists = os.path.exists(data_path)
        actual_size = os.path.getsize(data_path) if exists else None
        print(f"  {tier_name}:")
        print(f"    label:     {cfg['label']}")
        print(f"    bitlength: {W:,}")
        print(f"    seed:      {cfg['seed']}")
        print(f"    est. size: {est:,} bytes ({est / 1e9:.2f} GB)")
        print(f"    file:      {data_file}")
        if exists:
            print(f"    status:    EXISTS ({actual_size:,} bytes)")
        else:
            print(f"    status:    not generated")
        print()


def do_verify():
    """Verify existing .bsdata files by spot-checking against SBOX table."""
    tests_json_path = os.path.join(TESTS_DIR, "tests.json")
    if not os.path.exists(tests_json_path):
        print("ERROR: tests.json not found")
        sys.exit(1)

    with open(tests_json_path) as f:
        tests = json.load(f)

    all_ok = True
    for entry in tests:
        data_path = os.path.join(TESTS_DIR, entry["data_file"])
        name = entry["name"]
        if not os.path.exists(data_path):
            print(f"  SKIP {name}: {entry['data_file']} not found")
            continue

        print(f"  Verifying {name} ({entry['bitlength']:,} vectors)...")
        t0 = time.time()
        data = read_bsdata(data_path)
        t_read = time.time() - t0

        bitlength = data["bitlength"]
        u_arrays = data.get("input_arrays", {}).get("U", {})
        s_array = data.get("expected", {}).get("S", {})

        if not u_arrays or not s_array:
            print(f"    SKIP: missing U or S arrays")
            continue

        num_checks = min(500, bitlength)
        try:
            spot_check_sbox(u_arrays, s_array, bitlength,
                            num_checks=num_checks, seed=99999)
            t_total = time.time() - t0
            print(f"    OK: {num_checks} spot-checks passed "
                  f"(read {t_read:.1f}s, total {t_total:.1f}s)")
        except AssertionError as e:
            print(f"    FAIL: {e}")
            all_ok = False

    if all_ok:
        print("  All verifications passed.")
    else:
        print("  SOME VERIFICATIONS FAILED")
        sys.exit(1)


def load_tests_json():
    """Load existing tests.json if it exists."""
    path = os.path.join(TESTS_DIR, "tests.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_tests_json(tests):
    """Write tests.json, preserving unit tests and merging tier entries."""
    path = os.path.join(TESTS_DIR, "tests.json")
    with open(path, "w") as f:
        json.dump(tests, f, indent=2)
    print(f"  Wrote {path} ({len(tests)} entries)")


def merge_tier_entry(tests, new_entry):
    """Merge a tier entry into the test list, replacing any existing entry
    with the same data_file."""
    data_file = new_entry["data_file"]
    for i, entry in enumerate(tests):
        if entry.get("data_file") == data_file:
            tests[i] = new_entry
            return tests
    tests.append(new_entry)
    return tests


def main():
    parser = argparse.ArgumentParser(
        description="Generate AES S-box test data")
    parser.add_argument("--tier", choices=["small", "medium", "large", "all"],
                        help="Generate tier test data")
    parser.add_argument("--describe", action="store_true",
                        help="Print tier descriptions and exit")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing .bsdata files")
    args = parser.parse_args()

    if args.describe:
        do_describe()
        return

    if args.verify:
        do_verify()
        return

    with open(BS_PATH) as f:
        source = f.read()
    program = parse(source)

    if args.tier:
        # Tier generation mode: generate specified tier(s), merge into tests.json
        tier_names = list(TIER_CONFIGS.keys()) if args.tier == "all" else [args.tier]

        # Load existing tests.json to preserve unit tests
        tests = load_tests_json()

        for tier_name in tier_names:
            entry = generate_tier(program, tier_name)
            tests = merge_tier_entry(tests, entry)

        save_tests_json(tests)
    else:
        # Default: generate unit tests (original behavior)
        tests = generate_unit_tests(program)

        # Preserve any existing tier entries in tests.json
        existing = load_tests_json()
        tier_data_files = {f"aes_sbox_{t}.bsdata" for t in TIER_CONFIGS}
        for entry in existing:
            if entry.get("data_file") in tier_data_files:
                tests.append(entry)

        save_tests_json(tests)
        print(f"aes_sbox: generated {len(tests)} test entries")


if __name__ == "__main__":
    main()
