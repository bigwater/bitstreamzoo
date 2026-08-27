#!/usr/bin/env python3
"""Generate precomputed bitstream-level test data for BitFunnel.

Usage:
    python generate_tests.py                # unit tests only
    python generate_tests.py --tier small   # unit tests + small tier (~50MB)
    python generate_tests.py --tier medium  # unit tests + medium tier (~500MB)
    python generate_tests.py --tier large   # unit tests + large tier (~5GB)
    python generate_tests.py --describe     # print provenance info
    python generate_tests.py --verify       # verify SHA-256 of existing files
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from simulator.pythonsim import parse
from benchmark.apps.bitfunnel.src.run import bitfunnel_reference, run_bitfunnel_bs
from benchmark.bsdata import write_bsdata
from benchmark.tier_generate import (
    parse_generate_args, file_sha256, make_provenance,
    tier_test_entry, print_describe, verify_files,
    getrandbits_large,
)
from benchmark.tier_config import get_tier_bitlength

TESTS_DIR = os.path.dirname(__file__)
BS_PATH = os.path.join(TESTS_DIR, "../../src/bitfunnel.bs")

DOMAIN = "bitfunnel"

TIER_SEEDS = {"small": 3000, "medium": 3001, "large": 3002}

TIER_NAMES = {
    "small":  "Tier small (~50M documents)",
    "medium": "Tier medium (~500M documents)",
    "large":  "Tier large (~5B documents)",
}

# Typical query profile used for tier data
TIER_K0, TIER_K1, TIER_K2 = 3, 3, 2


def make_row_arrays(rng, K0, K1, K2, N):
    """Generate random row streams for a given (K0, K1, K2, N) configuration.

    Rank-0 rows: N random bits (density ~0.5, uniform random).
    Rank-1 rows: lower N/2 bits random, upper N/2 bits zero.
    Rank-2 rows: lower N/4 bits random, upper 3N/4 bits zero.

    Returns (rows_r0, rows_r1, rows_r2) as dicts {index: int_stream}.
    """
    HALF = N // 2
    QUARTER = N // 4
    rows_r0 = {i: rng.getrandbits(N)    for i in range(K0)}
    rows_r1 = {i: rng.getrandbits(HALF) for i in range(K1)}
    rows_r2 = {i: rng.getrandbits(QUARTER) for i in range(K2)}
    return rows_r0, rows_r1, rows_r2


def make_row_arrays_large(rng, K0, K1, K2, N):
    """Large-N version of make_row_arrays using getrandbits_large."""
    HALF = N // 2
    QUARTER = N // 4
    rows_r0 = {i: getrandbits_large(rng, N)       for i in range(K0)}
    rows_r1 = {i: getrandbits_large(rng, HALF)    for i in range(K1)}
    rows_r2 = {i: getrandbits_large(rng, QUARTER) for i in range(K2)}
    return rows_r0, rows_r1, rows_r2


def bitfunnel_fast_reference(rows_r0, rows_r1, rows_r2, K0, K1, K2, N, HALF, QUARTER):
    """Fast bitwise reference (no interpreter) for tier data verification."""
    mask = (1 << N) - 1
    acc = mask
    for i in range(K0):
        acc &= rows_r0[i]
    for i in range(K1):
        row = rows_r1[i]
        acc &= row | (row << HALF)
    for i in range(K2):
        row = rows_r2[i]
        stage = row | (row << QUARTER)
        acc &= stage | (stage << HALF)
    acc &= mask
    return acc, bin(acc).count('1')


def load_tests_json():
    path = os.path.join(TESTS_DIR, "tests.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_tests_json(tests):
    path = os.path.join(TESTS_DIR, "tests.json")
    with open(path, "w") as f:
        json.dump(tests, f, indent=2)
    print(f"  Wrote {path} ({len(tests)} entries)")


def merge_tier_entry(tests, new_entry):
    data_file = new_entry["data_file"]
    for i, entry in enumerate(tests):
        if entry.get("data_file") == data_file:
            tests[i] = new_entry
            return tests
    tests.append(new_entry)
    return tests


def generate_tier_data(tier, N):
    """Generate tier .bsdata with random sparse rows."""
    K0, K1, K2 = TIER_K0, TIER_K1, TIER_K2
    seed = TIER_SEEDS[tier]
    rng = random.Random(seed)
    HALF = N // 2
    QUARTER = N // 4

    print(f"  Generating {tier} tier: N={N:,} docs (K0={K0},K1={K1},K2={K2}, seed={seed})...")

    print(f"    Generating {K0} rank-0 rows ({N:,} bits each)...")
    rows_r0, rows_r1, rows_r2 = make_row_arrays_large(rng, K0, K1, K2, N)

    print(f"    Computing expected via fast reference...")
    expected_matches, expected_count = bitfunnel_fast_reference(
        rows_r0, rows_r1, rows_r2, K0, K1, K2, N, HALF, QUARTER)
    print(f"    Expected count: {expected_count:,} / {N:,} "
          f"({100*expected_count/N:.3f}% candidates)")

    data_file = f"bitfunnel_tier_{tier}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_file)
    print(f"    Writing {data_file}...")
    write_bsdata(
        data_path,
        N,
        params={"K0": K0, "K1": K1, "K2": K2, "HALF": HALF, "QUARTER": QUARTER},
        input_arrays={
            "rows_r0": rows_r0,
            "rows_r1": rows_r1,
            "rows_r2": rows_r2,
        },
        expected={"count": expected_count, "matches": expected_matches},
    )
    size_bytes = os.path.getsize(data_path)
    print(f"    Wrote {data_file} ({size_bytes:,} bytes, {size_bytes/1e6:.1f} MB)")
    return data_file, size_bytes


def main():
    args = parse_generate_args(DOMAIN)

    if args.describe or args.verify:
        tests_json_path = os.path.join(TESTS_DIR, "tests.json")
        if not os.path.exists(tests_json_path):
            print(f"No tests.json found at {tests_json_path}")
            sys.exit(1)
        with open(tests_json_path) as f:
            tests = json.load(f)
        if args.describe:
            print(f"bitfunnel: {len(tests)} test entries")
            print_describe(TESTS_DIR, tests)
        if args.verify:
            print("bitfunnel: verifying SHA-256 checksums")
            ok = verify_files(TESTS_DIR, tests)
            sys.exit(0 if ok else 1)
        return

    with open(BS_PATH) as f:
        source = f.read()
    program = parse(source)

    # -- Tier generation -----------------------------------------------
    if args.tier:
        tier = args.tier
        N = get_tier_bitlength(DOMAIN, tier)
        if N is None:
            print(f"bitfunnel: tier '{tier}' not applicable (skipped)")
            return
        data_file, size_bytes = generate_tier_data(tier, N)
        sha = file_sha256(os.path.join(TESTS_DIR, data_file))
        K0, K1, K2 = TIER_K0, TIER_K1, TIER_K2
        prov = make_provenance(
            source="synthetic",
            seed=TIER_SEEDS[tier],
            description=(
                f"Random bit-sliced rows: rank-0 uniform random (d≈0.5), "
                f"rank-1 lower-half random, rank-2 lower-quarter random. "
                f"K0={K0}, K1={K1}, K2={K2}, N={N:,} documents."
            ),
            generated_by="generate_tests.py --tier " + tier,
        )
        prov["sha256"] = sha
        entry = tier_test_entry(
            name=TIER_NAMES[tier],
            bitlength=N,
            data_file=data_file,
            size_bytes=size_bytes,
            provenance=prov,
        )
        tests = load_tests_json()
        tests = merge_tier_entry(tests, entry)
        save_tests_json(tests)
        print(f"bitfunnel: tier '{tier}' merged into tests.json")
        return

    tests = []

    # -- Fixed unit tests -----------------------------------------------
    def write_test(name, rows_r0, rows_r1, rows_r2, K0, K1, K2, N):
        HALF = N // 2
        QUARTER = N // 4
        ref_matches, ref_count = bitfunnel_reference(
            rows_r0, rows_r1, rows_r2, K0, K1, K2, N, HALF, QUARTER)
        bs_count, bs_matches, interp = run_bitfunnel_bs(
            program, rows_r0, rows_r1, rows_r2, K0, K1, K2, N, HALF, QUARTER)
        assert bs_count == ref_count, (
            f"{name}: bs_count={bs_count} ref={ref_count}")
        assert (bs_matches & ((1 << N) - 1)) == ref_matches, (
            f"{name}: bs_matches mismatch")
        fname = "bf_" + name.lower().replace(" ", "_").replace("=", "") + ".bsdata"
        write_bsdata(
            os.path.join(TESTS_DIR, fname),
            N,
            params={"K0": K0, "K1": K1, "K2": K2, "HALF": HALF, "QUARTER": QUARTER},
            input_arrays={
                "rows_r0": rows_r0,
                "rows_r1": rows_r1,
                "rows_r2": rows_r2,
            },
            expected={"count": ref_count, "matches": ref_matches},
        )
        tests.append({"name": name, "bitlength": N, "data_file": fname})
        print(f"  [{name}] N={N} K0={K0}/K1={K1}/K2={K2} count={ref_count} OK")

    # 1. All-ones rows: both rows are all-ones → acc stays ONES
    write_test("All-ones rows", {0: 0xFF, 1: 0xFF}, {}, {}, 2, 0, 0, 8)

    # 2. One zero row: second row is all-zeros → acc = 0
    write_test("One zero row", {0: 0xFF, 1: 0x00}, {}, {}, 2, 0, 0, 8)

    # 3. No overlap: rows are disjoint halves → count = 0
    write_test("No overlap", {0: 0xF0, 1: 0x0F}, {}, {}, 2, 0, 0, 8)

    # 4. Rank-1 expand: row=0x05, expanded=0x05|(0x05<<4)=0x55, count=4
    #    (bits 0,2 in lower half broadcast to 4,6 in upper half)
    write_test("Rank-1 expand", {}, {0: 0x05}, {}, 0, 1, 0, 8)

    # 5. Rank-2 expand: row=0x01, stage=0x05, expanded=0x55, count=4
    #    (bit 0 in lower quarter broadcasts to positions 0,2,4,6)
    write_test("Rank-2 expand", {}, {}, {0: 0x01}, 0, 0, 1, 8)

    # 6. Mixed ranks: N=16, K0=1, K1=1, K2=1
    #    r0=0x00FF, r1_expanded=0x3333, r2_expanded=0x5555
    #    acc = 0x00FF & 0x3333 & 0x5555 = 0x0011, count=2
    write_test("Mixed ranks",
               {0: 0x00FF},
               {0: 0x0033},
               {0: 0x0005},
               1, 1, 1, 16)

    # 7. Random d≈0.5, N=1000, verified vs scalar
    rng = random.Random(42)
    N8 = 1000
    rows_r0_8 = {i: rng.getrandbits(N8) for i in range(3)}
    rows_r1_8 = {i: rng.getrandbits(N8 // 2) for i in range(3)}
    rows_r2_8 = {i: rng.getrandbits(N8 // 4) for i in range(1)}
    write_test("Random N=1000", rows_r0_8, rows_r1_8, rows_r2_8, 3, 3, 1, N8)

    # 8. Large random, N=65536, verified vs scalar
    rng9 = random.Random(99)
    N9 = 65536
    rows_r0_9 = {i: rng9.getrandbits(N9)       for i in range(3)}
    rows_r1_9 = {i: rng9.getrandbits(N9 // 2)  for i in range(3)}
    rows_r2_9 = {i: rng9.getrandbits(N9 // 4)  for i in range(2)}
    write_test("Large N=65536", rows_r0_9, rows_r1_9, rows_r2_9, 3, 3, 2, N9)

    # Preserve existing tier entries
    existing = load_tests_json()
    tier_files = {f"bitfunnel_tier_{t}.bsdata" for t in TIER_NAMES}
    for entry in existing:
        if entry.get("data_file") in tier_files:
            tests.append(entry)

    save_tests_json(tests)
    print(f"\nbitfunnel: generated {len(tests)} test entries")


if __name__ == "__main__":
    main()
