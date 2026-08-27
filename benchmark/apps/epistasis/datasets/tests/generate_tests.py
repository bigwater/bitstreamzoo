#!/usr/bin/env python3
"""Generate precomputed bitstream-level test data for Epistasis.

Usage:
    python generate_tests.py                # unit tests; preserves existing tier entries
    python generate_tests.py --tier small   # small tier only (133M vectors), merged into tests.json
    python generate_tests.py --tier medium  # medium tier only (1.33B vectors), merged into tests.json
    python generate_tests.py --tier large   # large tier only (13.3B vectors), merged into tests.json
    python generate_tests.py --describe     # print provenance info
    python generate_tests.py --verify       # verify SHA-256 of existing files
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter
from benchmark.apps.epistasis.src.run import epistasis_reference
from benchmark.bsdata import write_bsdata
from benchmark.tier_generate import (
    parse_generate_args, file_sha256, make_provenance,
    tier_test_entry, print_describe, verify_files,
    getrandbits_large,
)
from benchmark.tier_config import get_tier_vectors

TESTS_DIR = os.path.dirname(__file__)
BS_PATH = os.path.join(TESTS_DIR, "../../src/epistasis.bs")

DOMAIN = "epistasis"

# Seed offsets for each tier
TIER_SEEDS = {"small": 1000, "medium": 1001, "large": 1002}

# Tier display names
TIER_NAMES = {
    "small":  "Tier small (133M individuals)",
    "medium": "Tier medium (1.33B individuals)",
    "large":  "Tier large (13.3B individuals)",
}


def run_bs(program, snp_a, snp_b, valid):
    """Run interpreter, return (n_11, n_10, n_01, n_00)."""
    interp = Interpreter()
    result = interp.run(program,
                        inputs={"snp_a": snp_a, "snp_b": snp_b, "valid": valid})
    return (result["n_11"], result["n_10"], result["n_01"], result["n_00"])


def epistasis_popcount_reference(snp_a, snp_b, valid, bitlength):
    """Compute expected outputs using direct bit operations (no interpreter).

    This is equivalent to the .bs program but uses pure Python bitwise ops,
    which is efficient for large bitlength where the interpreter would be slow.
    """
    all_ones = (1 << bitlength) - 1
    not_a = all_ones ^ snp_a
    not_b = all_ones ^ snp_b
    n_11 = bin(snp_a & snp_b & valid).count('1')
    n_10 = bin(snp_a & not_b & valid).count('1')
    n_01 = bin(not_a & snp_b & valid).count('1')
    n_00 = bin(not_a & not_b & valid).count('1')
    return (n_11, n_10, n_01, n_00)


def generate_tier_data(tier, bitlength):
    """Generate a tier .bsdata file with random SNP data.

    Uses uniform random bits for snp_a and snp_b (~50% density).
    Uses OR of 4 independent random streams for valid (~93.75% density,
    approximating 95% validity rate in GWAS cohort data).

    Returns (data_file, size_bytes, expected_tuple).
    """
    seed = TIER_SEEDS[tier]
    rng = random.Random(seed)

    print(f"  Generating {tier} tier: {bitlength:,} vectors (seed={seed})...")

    # Generate random bitstreams
    print(f"    Generating snp_a ({bitlength:,} bits)...")
    snp_a = getrandbits_large(rng, bitlength)

    print(f"    Generating snp_b ({bitlength:,} bits)...")
    snp_b = getrandbits_large(rng, bitlength)

    # valid: OR of 4 independent random streams -> ~93.75% density
    print(f"    Generating valid ({bitlength:,} bits, ~94% density)...")
    valid = getrandbits_large(rng, bitlength)
    for _ in range(3):
        valid = valid | getrandbits_large(rng, bitlength)

    # Compute expected outputs via direct popcount (no interpreter needed)
    print(f"    Computing expected outputs via popcount...")
    expected = epistasis_popcount_reference(snp_a, snp_b, valid, bitlength)
    n_11, n_10, n_01, n_00 = expected

    # Sanity check: counts should sum to popcount(valid)
    total_valid = bin(valid).count('1')
    assert n_11 + n_10 + n_01 + n_00 == total_valid, \
        f"Count mismatch: {n_11}+{n_10}+{n_01}+{n_00} != {total_valid}"
    print(f"    Expected: n_11={n_11:,} n_10={n_10:,} n_01={n_01:,} n_00={n_00:,}")
    print(f"    Total valid: {total_valid:,} / {bitlength:,} ({100*total_valid/bitlength:.1f}%)")

    # Write .bsdata
    data_file = f"epistasis_tier_{tier}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_file)
    print(f"    Writing {data_file}...")
    write_bsdata(
        data_path,
        bitlength,
        inputs={"snp_a": snp_a, "snp_b": snp_b, "valid": valid},
        expected={"n_11": n_11, "n_10": n_10, "n_01": n_01, "n_00": n_00},
    )
    size_bytes = os.path.getsize(data_path)
    print(f"    Wrote {data_file} ({size_bytes:,} bytes, {size_bytes/1e6:.1f} MB)")

    return data_file, size_bytes, expected


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
    args = parse_generate_args(DOMAIN)

    # Handle --describe and --verify on existing tests.json
    if args.describe or args.verify:
        tests_json_path = os.path.join(TESTS_DIR, "tests.json")
        if not os.path.exists(tests_json_path):
            print(f"No tests.json found at {tests_json_path}")
            sys.exit(1)
        with open(tests_json_path) as f:
            tests = json.load(f)
        if args.describe:
            print(f"epistasis: {len(tests)} test entries")
            print_describe(TESTS_DIR, tests)
        if args.verify:
            print(f"epistasis: verifying SHA-256 checksums")
            ok = verify_files(TESTS_DIR, tests)
            sys.exit(0 if ok else 1)
        return

    with open(BS_PATH) as f:
        source = f.read()
    program = parse(source)

    # -- Tier generation (early return: load existing, merge, save) ----
    if args.tier:
        tier = args.tier
        bitlength = get_tier_vectors(DOMAIN, tier)
        if bitlength is None:
            print(f"epistasis: tier '{tier}' not applicable (skipped)")
            return
        data_file, size_bytes, expected = generate_tier_data(tier, bitlength)
        sha = file_sha256(os.path.join(TESTS_DIR, data_file))
        prov = make_provenance(
            source="synthetic",
            seed=TIER_SEEDS[tier],
            description=(
                f"Random GWAS SNP data: snp_a/snp_b ~50% density (uniform), "
                f"valid ~94% density (OR of 4 uniform random streams). "
                f"{bitlength:,} individuals."
            ),
            generated_by="generate_tests.py --tier " + tier,
        )
        prov["sha256"] = sha
        entry = tier_test_entry(
            name=TIER_NAMES[tier],
            bitlength=bitlength,
            data_file=data_file,
            size_bytes=size_bytes,
            provenance=prov,
        )
        tests = load_tests_json()
        tests = merge_tier_entry(tests, entry)
        save_tests_json(tests)
        print(f"epistasis: tier '{tier}' merged into tests.json")
        return

    tests = []

    # -- Fixed tests ---------------------------------------------------
    fixed = [
        ("All 1/1",       255, 255, 255, 8),
        ("All 1/0",       255,   0, 255, 8),
        ("Mixed half",    240,  15, 255, 8),
        ("Missing data",  255, 255, 170, 8),
        ("No valid",      255, 255,   0, 8),
        ("Single individual", 1, 0,   1, 1),
    ]

    for name, snp_a, snp_b, valid, width in fixed:
        bs_result = run_bs(program, snp_a, snp_b, valid)
        ref = epistasis_reference(snp_a, snp_b, valid, width)
        assert bs_result == ref, f"{name}: bs={bs_result} ref={ref}"
        fname = name.lower().replace(" ", "_").replace("/", "_")
        write_bsdata(
            os.path.join(TESTS_DIR, f"{fname}.bsdata"),
            width,
            inputs={"snp_a": snp_a, "snp_b": snp_b, "valid": valid},
            expected={"n_11": ref[0], "n_10": ref[1],
                      "n_01": ref[2], "n_00": ref[3]},
        )
        tests.append({
            "name": name,
            "bitlength": width,
            "data_file": f"{fname}.bsdata",
        })

    # -- Random 16-bit x100 -------------------------------------------
    rng = random.Random(42)
    cases_16 = []
    for _ in range(100):
        snp_a = rng.randint(0, 65535)
        snp_b = rng.randint(0, 65535)
        valid = rng.randint(0, 65535)
        bs_result = run_bs(program, snp_a, snp_b, valid)
        ref = epistasis_reference(snp_a, snp_b, valid, 16)
        assert bs_result == ref
        cases_16.append({
            "bitlength": 16,
            "inputs": {"snp_a": snp_a, "snp_b": snp_b, "valid": valid},
            "expected": {"n_11": ref[0], "n_10": ref[1],
                         "n_01": ref[2], "n_00": ref[3]},
        })
    write_bsdata(
        os.path.join(TESTS_DIR, "epistasis_random_16bit_100.bsdata"),
        16,
        cases=cases_16,
    )
    tests.append({
        "name": "Random 16-bit x100",
        "bitlength": 100,
        "data_file": "epistasis_random_16bit_100.bsdata",
    })

    # -- Random 64-bit x50 --------------------------------------------
    rng = random.Random(43)
    cases_64 = []
    for _ in range(50):
        snp_a = rng.randint(0, (1 << 64) - 1)
        snp_b = rng.randint(0, (1 << 64) - 1)
        valid = rng.randint(0, (1 << 64) - 1)
        bs_result = run_bs(program, snp_a, snp_b, valid)
        ref = epistasis_reference(snp_a, snp_b, valid, 64)
        assert bs_result == ref
        cases_64.append({
            "bitlength": 64,
            "inputs": {"snp_a": snp_a, "snp_b": snp_b, "valid": valid},
            "expected": {"n_11": ref[0], "n_10": ref[1],
                         "n_01": ref[2], "n_00": ref[3]},
        })
    write_bsdata(
        os.path.join(TESTS_DIR, "epistasis_random_64bit_50.bsdata"),
        64,
        cases=cases_64,
    )
    tests.append({
        "name": "Random 64-bit x50",
        "bitlength": 50,
        "data_file": "epistasis_random_64bit_50.bsdata",
    })

    # -- Large 1M individuals (precomputed) ----------------------------
    W = 1000000
    rng_wide = random.Random(42)
    snp_a_wide = rng_wide.getrandbits(W)
    snp_b_wide = rng_wide.getrandbits(W)
    valid_wide = rng_wide.getrandbits(W)
    bs_wide = run_bs(program, snp_a_wide, snp_b_wide, valid_wide)
    # Verify against direct popcount reference
    all_ones = (1 << W) - 1
    not_a_wide = all_ones ^ snp_a_wide
    not_b_wide = all_ones ^ snp_b_wide
    ref_11 = bin(snp_a_wide & snp_b_wide & valid_wide).count('1')
    ref_10 = bin(snp_a_wide & not_b_wide & valid_wide).count('1')
    ref_01 = bin(not_a_wide & snp_b_wide & valid_wide).count('1')
    ref_00 = bin(not_a_wide & not_b_wide & valid_wide).count('1')
    assert bs_wide == (ref_11, ref_10, ref_01, ref_00), \
        f"Large 1M: bs={bs_wide} ref={ref_11, ref_10, ref_01, ref_00}"
    # Sanity: counts should sum to popcount(valid)
    assert ref_11 + ref_10 + ref_01 + ref_00 == bin(valid_wide).count('1')
    write_bsdata(
        os.path.join(TESTS_DIR, "epistasis_large_1M.bsdata"),
        W,
        inputs={"snp_a": snp_a_wide, "snp_b": snp_b_wide, "valid": valid_wide},
        expected={"n_11": ref_11, "n_10": ref_10,
                  "n_01": ref_01, "n_00": ref_00},
    )
    tests.append({
        "name": "Large 1M individuals",
        "bitlength": W,
        "data_file": "epistasis_large_1M.bsdata",
    })

    # -- Preserve existing tier entries and write tests.json --------------
    existing = load_tests_json()
    tier_data_files = {f"epistasis_tier_{t}.bsdata" for t in TIER_NAMES}
    for entry in existing:
        if entry.get("data_file") in tier_data_files:
            tests.append(entry)

    save_tests_json(tests)
    print(f"epistasis: generated {len(tests)} test entries")


if __name__ == "__main__":
    main()
