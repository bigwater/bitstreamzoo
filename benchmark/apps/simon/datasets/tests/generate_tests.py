#!/usr/bin/env python3
"""Generate precomputed bitstream-level test data for SIMON 32/64.

Usage:
    python generate_tests.py                 # unit tests only (default)
    python generate_tests.py --tier small    # 6.25M vectors (~50MB)
    python generate_tests.py --tier medium   # 62.5M vectors (~500MB)
    python generate_tests.py --tier large    # 625M vectors (~5GB)
    python generate_tests.py --tier all      # all three tiers
    python generate_tests.py --describe      # print tier info and exit
    python generate_tests.py --verify        # verify existing .bsdata files
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter
from benchmark.apps.simon.src.run import (
    simon_key_schedule, simon_encrypt,
    bitslice_16bit, encode_round_keys, decode_16bit,
)
from benchmark.bsdata import write_bsdata, read_bsdata
from benchmark.tier_config import get_tier_vectors, estimate_size_bytes
from benchmark.tier_generate import (
    parse_generate_args, file_sha256, make_provenance, tier_test_entry,
    print_describe, verify_files,
)

TESTS_DIR = os.path.dirname(__file__)
BS_PATH = os.path.join(TESTS_DIR, "../../src/simon.bs")

# Paper test key: k3=0x1918, k2=0x1110, k1=0x0908, k0=0x0100
PAPER_KEY = [0x0100, 0x0908, 0x1110, 0x1918]

TIER_CONFIGS = {
    "small":  {"bitlength":    6_250_000, "seed": 1000, "label": "Small 6.25M blocks"},
    "medium": {"bitlength":   62_500_000, "seed": 1001, "label": "Medium 62.5M blocks"},
    "large":  {"bitlength":  625_000_000, "seed": 1002, "label": "Large 625M blocks"},
}



def run_bs(program, L_vals, R_vals, round_keys):
    """Run SIMON on K plaintext pairs with given round keys.

    Returns (result_dict, op_count).
    """
    K = len(L_vals)
    plainL = bitslice_16bit(L_vals)
    plainR = bitslice_16bit(R_vals)
    rk_arrays = encode_round_keys(round_keys)

    interp = Interpreter()
    result = interp.run(
        program,
        inputs={},
        params={},
        input_arrays={"plainL": plainL, "plainR": plainR,
                      "round_key": rk_arrays},
    )
    return result, interp.op_count


def make_test(name, L_vals, R_vals, key_words, program):
    """Build a single test entry: run BS, verify against reference, write .bsdata."""
    K = len(L_vals)
    round_keys = simon_key_schedule(key_words)

    plainL = bitslice_16bit(L_vals)
    plainR = bitslice_16bit(R_vals)
    rk_arrays = encode_round_keys(round_keys)

    result, ops = run_bs(program, L_vals, R_vals, round_keys)
    got_L = decode_16bit(result, "cipherL", K)
    got_R = decode_16bit(result, "cipherR", K)

    # Verify against reference
    for k in range(K):
        ref_L, ref_R = simon_encrypt(L_vals[k], R_vals[k], round_keys)
        assert got_L[k] == ref_L and got_R[k] == ref_R, (
            f"{name} block {k}: got=({got_L[k]:04x},{got_R[k]:04x}) "
            f"ref=({ref_L:04x},{ref_R:04x})")

    # Mask to K bits
    mask = (1 << K) - 1
    expected_cL = {}
    expected_cR = {}
    for b in range(16):
        expected_cL[b] = result["cipherL"].get(b, 0) & mask
        expected_cR[b] = result["cipherR"].get(b, 0) & mask

    # Write .bsdata file
    safe_name = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    data_filename = f"simon_{safe_name}.bsdata"
    write_bsdata(
        os.path.join(TESTS_DIR, data_filename),
        K,
        input_arrays={
            "plainL": plainL,
            "plainR": plainR,
            "round_key": rk_arrays,
        },
        expected={
            "cipherL": expected_cL,
            "cipherR": expected_cR,
        },
    )
    return {
        "name": name,
        "bitlength": K,
        "data_file": data_filename,
    }


def spot_check_simon(plainL_arrays, plainR_arrays, cipherL_arrays, cipherR_arrays,
                     round_keys, bitlength, num_checks=500, seed=None):
    """Spot-check random positions in large bitsliced streams against reference."""
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(bitlength), min(num_checks, bitlength)))
    for k in indices:
        L = sum(((plainL_arrays[b] >> k) & 1) << b for b in range(16))
        R = sum(((plainR_arrays[b] >> k) & 1) << b for b in range(16))
        ref_L, ref_R = simon_encrypt(L, R, round_keys)

        got_L = sum(((cipherL_arrays.get(b, 0) >> k) & 1) << b for b in range(16))
        got_R = sum(((cipherR_arrays.get(b, 0) >> k) & 1) << b for b in range(16))

        assert got_L == ref_L and got_R == ref_R, (
            f"spot-check block {k}: got=({got_L:04x},{got_R:04x}) "
            f"ref=({ref_L:04x},{ref_R:04x})")
    return len(indices)


def estimate_bsdata_size(bitlength, n_streams=64):
    """Estimate .bsdata file size: n_streams * ceil(bitlength/8) + header.

    SIMON has 64 variable-width streams (16 plainL + 16 plainR + 16 cipherL + 16 cipherR).
    The 512 round_key streams are constant (0 or -1) and stored compactly.
    """
    bytes_per_stream = math.ceil(bitlength / 8)
    # Only the 64 variable streams occupy full-width binary space. The 512
    # round_key streams are constant (0 or -1) and stored as a compact inline
    # list in the JSON header, so they do not scale with bitlength. This
    # matches benchmark/tier_config.py (n_streams=64) and the real files
    # (e.g. small = 64*ceil(W/8) + ~1.5 KB header).
    return n_streams * bytes_per_stream + 256  # 256 bytes header estimate


def generate_unit_tests(program):
    """Generate the original unit tests."""
    tests = []

    # -- SIMON paper test vector ------------------------------------------
    # Key = 1918 1110 0908 0100
    # Plaintext = 6565 6877
    # Ciphertext = c69b e9bb
    paper_L = [0x6565]
    paper_R = [0x6877]
    tests.append(make_test(
        "SIMON paper test vector",
        paper_L, paper_R, PAPER_KEY, program))

    # Verify paper vector matches expected ciphertext
    rk = simon_key_schedule(PAPER_KEY)
    cL, cR = simon_encrypt(0x6565, 0x6877, rk)
    assert cL == 0xc69b and cR == 0xe9bb, (
        f"Paper vector: got ({cL:04x},{cR:04x}), expected (c69b,e9bb)")

    # -- Zero key/plaintext -----------------------------------------------
    tests.append(make_test(
        "Zero key and plaintext",
        [0x0000], [0x0000], [0x0000, 0x0000, 0x0000, 0x0000], program))

    # -- All-ones key/plaintext -------------------------------------------
    tests.append(make_test(
        "All-ones key and plaintext",
        [0xFFFF], [0xFFFF], [0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF], program))

    # -- Multi-block batch ------------------------------------------------
    rng = random.Random(42)
    key = [rng.randint(0, 0xFFFF) for _ in range(4)]
    L_vals = [rng.randint(0, 0xFFFF) for _ in range(8)]
    R_vals = [rng.randint(0, 0xFFFF) for _ in range(8)]
    tests.append(make_test(
        "Random 8-block batch",
        L_vals, R_vals, key, program))

    # -- Random 16-block x50 (data file) ----------------------------------
    rng = random.Random(43)
    sub_cases = []
    for _ in range(50):
        key = [rng.randint(0, 0xFFFF) for _ in range(4)]
        K = 16
        Ls = [rng.randint(0, 0xFFFF) for _ in range(K)]
        Rs = [rng.randint(0, 0xFFFF) for _ in range(K)]
        round_keys = simon_key_schedule(key)

        plainL = bitslice_16bit(Ls)
        plainR = bitslice_16bit(Rs)
        rk_arrays = encode_round_keys(round_keys)

        result, ops = run_bs(program, Ls, Rs, round_keys)
        got_L = decode_16bit(result, "cipherL", K)
        got_R = decode_16bit(result, "cipherR", K)

        # Verify
        for k in range(K):
            ref_L, ref_R = simon_encrypt(Ls[k], Rs[k], round_keys)
            assert got_L[k] == ref_L and got_R[k] == ref_R

        mask = (1 << K) - 1
        expected_cL = {}
        expected_cR = {}
        for b in range(16):
            expected_cL[b] = result["cipherL"].get(b, 0) & mask
            expected_cR[b] = result["cipherR"].get(b, 0) & mask

        sub_cases.append({
            "bitlength": K,
            "input_arrays": {
                "plainL": plainL,
                "plainR": plainR,
                "round_key": rk_arrays,
            },
            "expected": {
                "cipherL": expected_cL,
                "cipherR": expected_cR,
            },
        })
    write_bsdata(
        os.path.join(TESTS_DIR, "simon_random_16block_50.bsdata"),
        800,  # total blocks = 50 cases x 16
        cases=sub_cases,
    )
    tests.append({
        "name": "Random 16-block x50",
        "bitlength": 800,  # 50 cases x 16 blocks
        "data_file": "simon_random_16block_50.bsdata",
    })

    # -- Random 64-block x20 (data file) ----------------------------------
    rng = random.Random(44)
    sub_cases = []
    for _ in range(20):
        key = [rng.randint(0, 0xFFFF) for _ in range(4)]
        K = 64
        Ls = [rng.randint(0, 0xFFFF) for _ in range(K)]
        Rs = [rng.randint(0, 0xFFFF) for _ in range(K)]
        round_keys = simon_key_schedule(key)

        plainL = bitslice_16bit(Ls)
        plainR = bitslice_16bit(Rs)
        rk_arrays = encode_round_keys(round_keys)

        result, ops = run_bs(program, Ls, Rs, round_keys)
        got_L = decode_16bit(result, "cipherL", K)
        got_R = decode_16bit(result, "cipherR", K)

        for k in range(K):
            ref_L, ref_R = simon_encrypt(Ls[k], Rs[k], round_keys)
            assert got_L[k] == ref_L and got_R[k] == ref_R

        mask = (1 << K) - 1
        expected_cL = {}
        expected_cR = {}
        for b in range(16):
            expected_cL[b] = result["cipherL"].get(b, 0) & mask
            expected_cR[b] = result["cipherR"].get(b, 0) & mask

        sub_cases.append({
            "bitlength": K,
            "input_arrays": {
                "plainL": plainL,
                "plainR": plainR,
                "round_key": rk_arrays,
            },
            "expected": {
                "cipherL": expected_cL,
                "cipherR": expected_cR,
            },
        })
    write_bsdata(
        os.path.join(TESTS_DIR, "simon_random_64block_20.bsdata"),
        1280,  # total blocks = 20 cases x 64
        cases=sub_cases,
    )
    tests.append({
        "name": "Random 64-block x20",
        "bitlength": 1280,  # 20 cases x 64 blocks
        "data_file": "simon_random_64block_20.bsdata",
    })

    # -- Large 1M blocks (precomputed) ------------------------------------
    tests.append({
        "name": "Large 1M blocks",
        "category": "generated",
        "bitlength": 1000000,
        "generate": {"W": 1000000, "seed": 42},
    })

    return tests


def generate_tier(program, tier_name):
    """Generate a single tier test (small/medium/large).

    Uses the SIMON paper test key. Generates random plaintext as direct
    bitstreams (getrandbits), computes round keys, runs the interpreter
    to get ciphertext, and spot-checks against the scalar reference.
    """
    cfg = TIER_CONFIGS[tier_name]
    W = cfg["bitlength"]
    seed = cfg["seed"]
    label = cfg["label"]
    data_file = f"simon_{tier_name}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_file)

    print(f"  Generating {label} (bitlength={W:,}, seed={seed})...")

    # Fixed key: SIMON paper test key
    key_words = PAPER_KEY
    round_keys = simon_key_schedule(key_words)

    # Generate 32 random input streams (16 for L, 16 for R) directly
    t0 = time.time()
    rng = random.Random(seed)
    plainL_arrays = {b: rng.getrandbits(W) for b in range(16)}
    plainR_arrays = {b: rng.getrandbits(W) for b in range(16)}
    t_gen = time.time() - t0
    print(f"    Input generation: {t_gen:.1f}s")

    # Encode round keys (512 constant streams)
    rk_arrays = encode_round_keys(round_keys)

    # Run interpreter
    t0 = time.time()
    interp = Interpreter()
    result = interp.run(
        program,
        inputs={},
        params={},
        input_arrays={"plainL": plainL_arrays, "plainR": plainR_arrays,
                      "round_key": rk_arrays},
    )
    cipherL_arrays = result["cipherL"]
    cipherR_arrays = result["cipherR"]
    t_interp = time.time() - t0
    print(f"    Interpreter: {t_interp:.1f}s ({interp.op_count} ops)")

    # Spot-check 500 random positions
    t0 = time.time()
    n_checked = spot_check_simon(plainL_arrays, plainR_arrays,
                                 cipherL_arrays, cipherR_arrays,
                                 round_keys, W,
                                 num_checks=500, seed=seed + 5000)
    t_check = time.time() - t0
    print(f"    Spot-check: {n_checked} positions verified in {t_check:.1f}s")

    # Mask to bitlength bits
    mask = (1 << W) - 1

    # Write .bsdata
    t0 = time.time()
    write_bsdata(
        data_path,
        W,
        input_arrays={
            "plainL": {b: v & mask for b, v in plainL_arrays.items()},
            "plainR": {b: v & mask for b, v in plainR_arrays.items()},
            "round_key": rk_arrays,
        },
        expected={
            "cipherL": {b: v & mask for b, v in cipherL_arrays.items()},
            "cipherR": {b: v & mask for b, v in cipherR_arrays.items()},
        },
    )
    t_write = time.time() - t0

    file_size = os.path.getsize(data_path)
    print(f"    Written: {data_path} ({file_size:,} bytes, {t_write:.1f}s)")

    sha = file_sha256(data_path)
    prov = make_provenance(
        source="synthetic",
        seed=seed,
        description=(f"32 random plaintext streams via getrandbits({W}), "
                     f"key=SIMON paper key [0x0100,0x0908,0x1110,0x1918]; "
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
    print("SIMON 32/64 tier configurations:")
    print()
    for tier_name, cfg in TIER_CONFIGS.items():
        W = cfg["bitlength"]
        est = estimate_bsdata_size(W)
        data_file = f"simon_{tier_name}.bsdata"
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
    """Verify existing .bsdata files by spot-checking against SIMON reference."""
    tests_json_path = os.path.join(TESTS_DIR, "tests.json")
    if not os.path.exists(tests_json_path):
        print("ERROR: tests.json not found")
        sys.exit(1)

    with open(tests_json_path) as f:
        tests = json.load(f)

    all_ok = True
    for entry in tests:
        data_file = entry.get("data_file", "")
        if not data_file:
            continue
        data_path = os.path.join(TESTS_DIR, data_file)
        name = entry["name"]
        if not os.path.exists(data_path):
            print(f"  SKIP {name}: {data_file} not found")
            continue

        print(f"  Verifying {name} ({entry['bitlength']:,} vectors)...")
        t0 = time.time()
        data = read_bsdata(data_path)
        t_read = time.time() - t0

        # Multi-case .bsdata files return {"cases": [...]};
        # single-case files return a flat dict with "bitlength" at top level.
        if "cases" in data:
            case_list = data["cases"]
        else:
            case_list = [data]

        total_checks = 0
        for case_data in case_list:
            bitlength = case_data["bitlength"]
            plainL_arrays = case_data.get("input_arrays", {}).get("plainL", {})
            plainR_arrays = case_data.get("input_arrays", {}).get("plainR", {})
            rk_raw = case_data.get("input_arrays", {}).get("round_key", {})
            cipherL_arrays = case_data.get("expected", {}).get("cipherL", {})
            cipherR_arrays = case_data.get("expected", {}).get("cipherR", {})

            if not plainL_arrays or not cipherL_arrays:
                print(f"    SKIP case: missing plainL or cipherL arrays")
                continue

            # Reconstruct round keys from the stored round_key arrays
            round_keys = []
            for r in range(32):
                rk_val = 0
                for b in range(16):
                    idx = r * 16 + b
                    if rk_raw.get(idx, 0) & 1:  # -1 has bit 0 set, 0 does not
                        rk_val |= 1 << b
                round_keys.append(rk_val)

            num_checks = min(500, bitlength)
            try:
                spot_check_simon(plainL_arrays, plainR_arrays,
                                 cipherL_arrays, cipherR_arrays,
                                 round_keys, bitlength,
                                 num_checks=num_checks, seed=99999)
                total_checks += num_checks
            except AssertionError as e:
                print(f"    FAIL: {e}")
                all_ok = False

        if total_checks > 0:
            t_total = time.time() - t0
            print(f"    OK: {total_checks} spot-checks passed "
                  f"(read {t_read:.1f}s, total {t_total:.1f}s)")

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
        description="Generate SIMON 32/64 test data")
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
        tier_data_files = {f"simon_{t}.bsdata" for t in TIER_CONFIGS}
        for entry in existing:
            if entry.get("data_file") in tier_data_files:
                tests.append(entry)

        save_tests_json(tests)
        print(f"simon: generated {len(tests)} test entries")


if __name__ == "__main__":
    main()
