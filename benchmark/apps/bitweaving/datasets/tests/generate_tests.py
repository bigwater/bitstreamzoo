#!/usr/bin/env python3
"""Generate precomputed bitstream-level test data for BitWeaving-V.

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
from simulator.pythonsim.interpreter import Interpreter
from benchmark.apps.bitweaving.src.run import (
    bitweaving_reference, encode_column, encode_bound, make_valid_mask,
)
from benchmark.bsdata import write_bsdata
from benchmark.tier_generate import (
    parse_generate_args, file_sha256, make_provenance,
    tier_test_entry, print_describe, verify_files,
    getrandbits_large,
)
from benchmark.tier_config import get_tier_vectors

TESTS_DIR = os.path.dirname(__file__)
BS_PATH = os.path.join(TESTS_DIR, "../../src/bitweaving.bs")

DOMAIN = "bitweaving"

TIER_SEEDS = {"small": 2000, "medium": 2001, "large": 2002}

TIER_NAMES = {
    "small":  "Tier small (40M rows, TPC-H Q6-style)",
    "medium": "Tier medium (400M rows, TPC-H Q6-style)",
    "large":  "Tier large (4B rows, TPC-H Q6-style)",
}

# TPC-H Q6 parameters
TPCH_B = 8          # bits per value (L_QUANTITY fits in 8 bits)
TPCH_LO = 1         # L_QUANTITY >= 1
TPCH_HI = 23        # L_QUANTITY < 24, i.e. <= 23
TPCH_VAL_MIN = 1    # TPC-H L_QUANTITY range
TPCH_VAL_MAX = 50


def run_bs(program, values, lo, hi, B):
    """Run interpreter, return (count, result_stream)."""
    N = len(values)
    interp = Interpreter()
    result = interp.run(
        program,
        inputs={"valid": make_valid_mask(N)},
        params={"B": B},
        input_arrays={
            "data": encode_column(values, B),
            "lo_bits": encode_bound(lo, B),
            "hi_bits": encode_bound(hi, B),
        },
    )
    return result["count"], result.get("result", 0), interp


def bitweaving_popcount_reference(data_streams, lo, hi, B, bitlength):
    """Compute expected count using direct bit operations (no interpreter).

    Mirrors the .bs program logic for efficient large-scale computation.
    """
    all_ones = (1 << bitlength) - 1

    lt = 0
    eq_lo = all_ones
    gt = 0
    eq_hi = all_ones

    for i in range(B):
        b = B - 1 - i
        lo_stream = all_ones if ((lo >> b) & 1) else 0
        hi_stream = all_ones if ((hi >> b) & 1) else 0
        d = data_streams[b]

        # val >= lo
        d_xor_l = d ^ lo_stream
        not_d = all_ones ^ d
        t1 = eq_lo & not_d
        new_lt = t1 & lo_stream
        lt = lt | new_lt
        not_dxl = all_ones ^ d_xor_l
        eq_lo = eq_lo & not_dxl

        # val <= hi
        d_xor_h = d ^ hi_stream
        not_h = all_ones ^ hi_stream
        t2 = eq_hi & d
        new_gt = t2 & not_h
        gt = gt | new_gt
        not_dxh = all_ones ^ d_xor_h
        eq_hi = eq_hi & not_dxh

    ge_lo = all_ones ^ lt
    le_hi = all_ones ^ gt
    matched = ge_lo & le_hi
    result = matched & all_ones  # valid mask = all_ones for tier data
    return bin(result).count('1'), result


def generate_tpch_column(rng, n_rows):
    """Generate TPC-H L_QUANTITY column: uniform integers in [1, 50]."""
    return [rng.randint(TPCH_VAL_MIN, TPCH_VAL_MAX) for _ in range(n_rows)]


def encode_column_to_streams_large(rng, bitlength, B, val_min, val_max):
    """Generate B independent random bit-plane streams for tier stress data.

    This intentionally produces a synthetic uniform [0, 2^B - 1] column,
    not the TPC-H [val_min, val_max] quantity distribution. The metadata
    records the data as synthetic stress data.
    """
    streams = {}
    for b in range(B):
        print(f"      bit-plane {b}/{B}...")
        streams[b] = getrandbits_large(rng, bitlength)
    return streams


def generate_tier_data(tier, bitlength):
    """Generate a tier .bsdata file with synthetic column data."""
    seed = TIER_SEEDS[tier]
    rng = random.Random(seed)
    B = TPCH_B
    lo = TPCH_LO
    hi = TPCH_HI

    print(f"  Generating {tier} tier: {bitlength:,} rows (seed={seed}, B={B})...")

    # Generate random bit-planes (uniform [0, 255] distribution)
    print(f"    Generating {B} bit-plane streams...")
    data_streams = encode_column_to_streams_large(rng, bitlength, B,
                                                  TPCH_VAL_MIN, TPCH_VAL_MAX)

    # Compute expected via direct bitwise ops
    print(f"    Computing expected count via bitwise reference...")
    expected_count, expected_result = bitweaving_popcount_reference(
        data_streams, lo, hi, B, bitlength)

    print(f"    Expected count: {expected_count:,} / {bitlength:,} "
          f"({100*expected_count/bitlength:.1f}% selectivity)")

    # Encode bounds as broadcast
    lo_arrays = encode_bound(lo, B)
    hi_arrays = encode_bound(hi, B)

    # Valid mask: all rows are valid
    valid_mask = (1 << bitlength) - 1

    # Write .bsdata
    data_file = f"bitweaving_tier_{tier}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_file)
    print(f"    Writing {data_file}...")
    write_bsdata(
        data_path,
        bitlength,
        params={"B": B},
        inputs={"valid": valid_mask},
        input_arrays={
            "data": data_streams,
            "lo_bits": lo_arrays,
            "hi_bits": hi_arrays,
        },
        expected={"count": expected_count, "result": expected_result},
    )
    size_bytes = os.path.getsize(data_path)
    print(f"    Wrote {data_file} ({size_bytes:,} bytes, {size_bytes/1e6:.1f} MB)")

    return data_file, size_bytes


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


def main():
    args = parse_generate_args(DOMAIN)

    # Handle --describe and --verify
    if args.describe or args.verify:
        tests_json_path = os.path.join(TESTS_DIR, "tests.json")
        if not os.path.exists(tests_json_path):
            print(f"No tests.json found at {tests_json_path}")
            sys.exit(1)
        with open(tests_json_path) as f:
            tests = json.load(f)
        if args.describe:
            print(f"bitweaving: {len(tests)} test entries")
            print_describe(TESTS_DIR, tests)
        if args.verify:
            print(f"bitweaving: verifying SHA-256 checksums")
            ok = verify_files(TESTS_DIR, tests)
            sys.exit(0 if ok else 1)
        return

    with open(BS_PATH) as f:
        source = f.read()
    program = parse(source)

    # -- Tier generation -------------------------------------------------
    if args.tier:
        tier = args.tier
        bitlength = get_tier_vectors(DOMAIN, tier)
        if bitlength is None:
            print(f"bitweaving: tier '{tier}' not applicable (skipped)")
            return
        data_file, size_bytes = generate_tier_data(tier, bitlength)
        sha = file_sha256(os.path.join(TESTS_DIR, data_file))
        prov = make_provenance(
            source="synthetic",
            seed=TIER_SEEDS[tier],
            description=(
                f"BitWeaving-V range scan: {TPCH_LO} <= val <= {TPCH_HI}, "
                f"B={TPCH_B}, {bitlength:,} rows. "
                f"Uniform random bit-planes for synthetic stress testing; "
                f"not TPC-H-distributed column data."
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
        print(f"bitweaving: tier '{tier}' merged into tests.json")
        return

    tests = []

    # -- Fixed unit tests ------------------------------------------------
    B4_MAX = (1 << 4) - 1  # 15

    fixed = [
        # (name, values, lo, hi, B)
        ("All match B=4",       [3, 5, 7, 9, 11], 0, B4_MAX, 4),
        ("No match B=4",        [0, 1, 2, 3, 4], 8, B4_MAX, 4),
        ("Single match B=4",    [1, 5, 10, 15, 3], 5, 5, 4),
        ("Boundary lo B=4",     [5, 5, 4, 6, 3], 5, B4_MAX, 4),
        ("Boundary hi B=4",     [5, 5, 4, 6, 3], 0, 5, 4),
        ("Full range B=4",      [0, 7, B4_MAX, 1, 8], 0, B4_MAX, 4),
        ("Empty range B=4",     [0, 7, B4_MAX, 1, 8], 10, 5, 4),
        ("All zeros B=4",       [0, 0, 0, 0], 0, 0, 4),
        ("All max B=4",         [B4_MAX]*4, B4_MAX, B4_MAX, 4),
        ("TPC-H Q6 small",     [12, 24, 1, 50, 23, 5, 30, 48], 1, 23, 8),
    ]

    for name, values, lo, hi, B in fixed:
        ref_count = bitweaving_reference(values, lo, hi)
        bs_count, bs_result, interp = run_bs(program, values, lo, hi, B)
        assert bs_count == ref_count, (
            f"{name}: bs_count={bs_count} ref={ref_count}"
        )
        fname = f"bw_{name.lower().replace(' ', '_').replace('=', '')}.bsdata"
        write_bsdata(
            os.path.join(TESTS_DIR, fname),
            len(values),
            params={"B": B},
            inputs={"valid": make_valid_mask(len(values))},
            input_arrays={
                "data": encode_column(values, B),
                "lo_bits": encode_bound(lo, B),
                "hi_bits": encode_bound(hi, B),
            },
            expected={"count": ref_count, "result": bs_result},
        )
        tests.append({
            "name": name,
            "bitlength": len(values),
            "data_file": fname,
        })
        print(f"  [{name}] count={ref_count} (B={B}, N={len(values)}) OK")

    # -- Random tests (B=8, 64 values x 50 cases) ----------------------
    rng = random.Random(42)
    multi_cases = []
    for _ in range(50):
        values = [rng.randint(TPCH_VAL_MIN, TPCH_VAL_MAX) for __ in range(64)]
        lo = rng.randint(0, 30)
        hi = rng.randint(lo, 55)
        ref_count = bitweaving_reference(values, lo, hi)
        bs_count, bs_result, _ = run_bs(program, values, lo, hi, 8)
        assert bs_count == ref_count, f"Random test: bs={bs_count} ref={ref_count}"
        multi_cases.append({
            "values": values, "lo": lo, "hi": hi,
            "count": ref_count, "result": bs_result,
        })

    # Write multi-case .bsdata
    # For simplicity, write as individual .bsdata (first and last only to keep small)
    fname = "bw_random_b8_x50.bsdata"
    # Use first case for a single stored test
    c = multi_cases[0]
    write_bsdata(
        os.path.join(TESTS_DIR, fname),
        64,
        params={"B": 8},
        inputs={"valid": make_valid_mask(64)},
        input_arrays={
            "data": encode_column(c["values"], 8),
            "lo_bits": encode_bound(c["lo"], 8),
            "hi_bits": encode_bound(c["hi"], 8),
        },
        expected={"count": c["count"], "result": c["result"]},
    )
    tests.append({
        "name": "Random B=8 (50 cases verified, 1 stored)",
        "bitlength": 64,
        "data_file": fname,
    })
    print(f"  [Random B=8 x50] All 50 verified, 1 stored OK")

    # -- Wide test (100K rows, B=8, TPC-H Q6) ---------------------------
    rng_wide = random.Random(999)
    N_WIDE = 100_000
    wide_values = [rng_wide.randint(TPCH_VAL_MIN, TPCH_VAL_MAX)
                   for _ in range(N_WIDE)]
    wide_ref = bitweaving_reference(wide_values, TPCH_LO, TPCH_HI)
    fname_wide = "bw_wide_100k_tpch_q6.bsdata"
    write_bsdata(
        os.path.join(TESTS_DIR, fname_wide),
        N_WIDE,
        params={"B": TPCH_B},
        inputs={"valid": make_valid_mask(N_WIDE)},
        input_arrays={
            "data": encode_column(wide_values, TPCH_B),
            "lo_bits": encode_bound(TPCH_LO, TPCH_B),
            "hi_bits": encode_bound(TPCH_HI, TPCH_B),
        },
        expected={"count": wide_ref},
    )
    tests.append({
        "name": f"Wide 100K TPC-H Q6 (count={wide_ref})",
        "bitlength": N_WIDE,
        "data_file": fname_wide,
    })
    print(f"  [Wide 100K TPC-H Q6] count={wide_ref}/{N_WIDE} "
          f"({100*wide_ref/N_WIDE:.1f}%) OK")

    save_tests_json(tests)
    print(f"\nbitweaving: generated {len(tests)} tests")


if __name__ == "__main__":
    main()
