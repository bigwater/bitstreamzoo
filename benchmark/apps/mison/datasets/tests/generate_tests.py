#!/usr/bin/env python3
"""Generate precomputed bitstream-level test data for Mison.

Usage:
    python generate_tests.py                # unit tests only (existing behavior)
    python generate_tests.py --tier small   # unit tests + small tier (33.3M chars)
    python generate_tests.py --tier medium  # unit tests + medium tier (333M chars)
    python generate_tests.py --tier large   # unit tests + large tier (3.33B chars)
    python generate_tests.py --describe     # print provenance info
    python generate_tests.py --verify       # verify SHA-256 of existing files
"""

import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter
from benchmark.apps.mison.src.run import (
    text_to_basis_streams, make_even_bits, compute_log_w,
    mison_reference, random_json,
)
from benchmark.bsdata import write_bsdata
from benchmark.tier_generate import (
    parse_generate_args, file_sha256, make_provenance,
    tier_test_entry, print_describe, verify_files,
    getrandbits_large,
)
from benchmark.tier_config import get_tier_vectors

TESTS_DIR = os.path.dirname(__file__)
BS_PATH = os.path.join(TESTS_DIR, "../../src/mison.bs")

DOMAIN = "mison"

# Seed offsets for each tier
TIER_SEEDS = {"small": 1000, "medium": 1001, "large": 1002}

# Tier display names
TIER_NAMES = {
    "small":  "Tier small (33.3M chars)",
    "medium": "Tier medium (333M chars)",
    "large":  "Tier large (3.33B chars)",
}


def run_bs(program, text):
    """Run interpreter on text, return output dict."""
    n = len(text)
    b = text_to_basis_streams(text)
    even_bits = make_even_bits(n)
    log_w = compute_log_w(n)

    interp = Interpreter()
    result = interp.run(
        program,
        inputs={"even_bits": even_bits},
        params={"LOG_W": log_w},
        input_arrays={"b": b},
    )
    return result


def mison_bitstream_reference(b, even_bits, log_w, bitlength):
    """Compute Mison expected outputs directly on basis streams.

    This replicates the .bs program logic using pure Python bitwise ops
    on large integers, which is efficient for tier-scale bitlength where
    the interpreter would be too slow.

    Args:
        b: dict {0..7: int} basis bit-planes
        even_bits: int, alternating 0101... pattern
        log_w: int, log2 of stream width
        bitlength: int, number of bit positions

    Returns:
        dict with 'real_quote', 'str_mask', 'structural'
    """
    mask = (1 << bitlength) - 1
    ones = mask  # all-ones constant

    # Stage 1: Character classification
    nb0 = (~b[0]) & mask
    nb1 = (~b[1]) & mask
    nb2 = (~b[2]) & mask
    nb3 = (~b[3]) & mask
    nb4 = (~b[4]) & mask
    nb5 = (~b[5]) & mask
    nb6 = (~b[6]) & mask

    b3_b4 = b[3] & b[4]
    low = nb6 & b[5]
    high = b[6] & b3_b4

    # Quote " (0x22)
    _q0 = low & nb4
    _q1 = _q0 & nb3
    _q2 = _q1 & nb2
    _q3 = _q2 & b[1]
    quote = _q3 & nb0

    # Backslash \ (0x5C)
    _bs0 = high & nb5
    _bs1 = _bs0 & b[2]
    _bs2 = _bs1 & nb1
    backslash = _bs2 & nb0

    # Brackets
    bracket_base = high & b[0]
    _open0 = bracket_base & b[1]
    open_br = _open0 & nb2
    _close0 = bracket_base & nb1
    close_br = _close0 & b[2]

    lbrace = open_br & b[5]
    rbrace = close_br & b[5]
    lbracket = open_br & nb5
    rbracket = close_br & nb5

    # Colon : (0x3A)
    _col0 = low & b3_b4
    _col1 = _col0 & nb2
    _col2 = _col1 & b[1]
    colon = _col2 & nb0

    # Comma , (0x2C)
    _com0 = low & nb4
    _com1 = _com0 & b[3]
    _com2 = _com1 & b[2]
    _com3 = _com2 & nb1
    comma = _com3 & nb0

    # Stage 2: Escape detection
    odd_bits = even_bits ^ ones
    bs_shifted = (backslash << 1) & mask
    _not_bs_shifted = (~bs_shifted) & mask
    bs_starts = backslash & _not_bs_shifted
    even_starts = bs_starts & even_bits
    odd_starts = bs_starts & odd_bits
    nbs = (~backslash) & mask
    ec_raw = (even_starts + backslash) & mask
    even_carries = ec_raw & nbs
    oc_raw = (odd_starts + backslash) & mask
    odd_carries = oc_raw & nbs
    _ec_odd = even_carries & odd_bits
    _oc_even = odd_carries & even_bits
    odd_ends = _ec_odd | _oc_even
    _not_odd_ends = (~odd_ends) & mask
    real_quote = quote & _not_odd_ends

    # Stage 3: String mask via prefix XOR
    str_mask = real_quote
    stride = 1
    for _ in range(log_w):
        _shifted = (str_mask << stride) & mask
        str_mask = str_mask ^ _shifted
        stride *= 2

    # Stage 4: Structural filtering
    not_str = (~str_mask) & mask
    struct_raw = lbrace | rbrace
    struct_raw = struct_raw | lbracket
    struct_raw = struct_raw | rbracket
    struct_raw = struct_raw | colon
    struct_raw = struct_raw | comma
    structural = struct_raw & not_str

    return {
        'real_quote': real_quote & mask,
        'str_mask': str_mask & mask,
        'structural': structural & mask,
    }


def generate_tier_data(tier, bitlength):
    """Generate a tier .bsdata file with random ASCII byte data.

    Uses random basis streams b[0..6] and b[7]=0 (ASCII range),
    matching the approach of the existing "Large 1M-bit wide" test.
    Computes expected outputs via direct bitstream operations (no interpreter).

    Returns (data_file, size_bytes).
    """
    seed = TIER_SEEDS[tier]
    rng = random.Random(seed)

    print(f"  Generating {tier} tier: {bitlength:,} vectors (seed={seed})...")

    # Generate random basis streams (b[0..6] random, b[7]=0 for ASCII)
    print(f"    Generating b[0..6] ({bitlength:,} bits each)...")
    b = {}
    for k in range(7):
        b[k] = getrandbits_large(rng, bitlength)
    b[7] = 0  # ASCII range

    # Generate even_bits and log_w
    even_bits = make_even_bits(bitlength)
    log_w = compute_log_w(bitlength)

    # Compute expected outputs via direct bitstream ops (no interpreter)
    print(f"    Computing expected outputs via bitstream reference...")
    expected = mison_bitstream_reference(b, even_bits, log_w, bitlength)

    mask = (1 << bitlength) - 1
    bs_structural = expected['structural']
    bs_str_mask = expected['str_mask']
    bs_real_quote = expected['real_quote']

    # Invariant check: structural chars should be outside strings
    assert bs_structural & bs_str_mask == 0, \
        "INVARIANT FAIL: structural & str_mask != 0"

    # Spot-check character classification at random positions
    struct_chars = set('{}[]:,')
    n_check = min(200, bitlength)
    positions = rng.sample(range(bitlength), n_check)
    for pos in positions:
        code = sum(((b[k] >> pos) & 1) << k for k in range(8))
        ch = chr(code) if code < 128 else '?'
        is_structural_char = ch in struct_chars
        pos_is_structural = (bs_structural >> pos) & 1
        pos_in_string = (bs_str_mask >> pos) & 1
        if is_structural_char and not pos_in_string:
            assert pos_is_structural, (
                f"spot-check pos {pos}: char '{ch}' outside string but not structural")
        elif pos_is_structural and not is_structural_char:
            assert False, (
                f"spot-check pos {pos}: char '{ch}' marked structural but is not")

    n_structural = bin(bs_structural).count('1')
    n_in_string = bin(bs_str_mask).count('1')
    n_quotes = bin(bs_real_quote).count('1')
    print(f"    structural: {n_structural:,} / {bitlength:,} ({100*n_structural/bitlength:.2f}%)")
    print(f"    in_string:  {n_in_string:,} / {bitlength:,} ({100*n_in_string/bitlength:.2f}%)")
    print(f"    real_quote: {n_quotes:,} / {bitlength:,} ({100*n_quotes/bitlength:.4f}%)")

    # Write .bsdata
    data_file = f"mison_tier_{tier}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_file)
    print(f"    Writing {data_file}...")
    write_bsdata(
        data_path,
        bitlength,
        inputs={"even_bits": even_bits & mask},
        params={"LOG_W": log_w},
        input_arrays={"b": {k: v & mask for k, v in b.items()}},
        expected={
            "structural": bs_structural & mask,
            "str_mask": bs_str_mask & mask,
            "real_quote": bs_real_quote & mask,
        },
    )
    size_bytes = os.path.getsize(data_path)
    print(f"    Wrote {data_file} ({size_bytes:,} bytes, {size_bytes/1e6:.1f} MB)")

    return data_file, size_bytes


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
            print(f"mison: {len(tests)} test entries")
            print_describe(TESTS_DIR, tests)
        if args.verify:
            print(f"mison: verifying SHA-256 checksums")
            ok = verify_files(TESTS_DIR, tests)
            sys.exit(0 if ok else 1)
        return

    with open(BS_PATH) as f:
        source = f.read()
    program = parse(source)

    # -- Tier generation (early return: load existing, merge, save) -----------
    if args.tier:
        tier = args.tier
        bitlength = get_tier_vectors(DOMAIN, tier)
        if bitlength is None:
            print(f"mison: tier '{tier}' not applicable (skipped)")
            return
        data_file, size_bytes = generate_tier_data(tier, bitlength)
        sha = file_sha256(os.path.join(TESTS_DIR, data_file))
        prov = make_provenance(
            source="synthetic",
            seed=TIER_SEEDS[tier],
            description=(
                f"Random ASCII byte data: b[0..6] uniform random, b[7]=0 (ASCII range). "
                f"Expected outputs computed via direct bitstream operations "
                f"(mison_bitstream_reference). "
                f"{bitlength:,} characters."
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
        print(f"mison: tier '{tier}' merged into tests.json")
        return

    tests = []

    # -- Fixed tests ----------------------------------------------------------
    fixed = [
        ("Empty object", "{}"),
        ("Simple KV", '{"a":1}'),
        ("Nested", '{"a":{"b":[1,2]}}'),
        ("Escaped quotes", '{"a":"he said \\"hi\\""}'),
        ("Escaped backslash", '{"a":"path\\\\\\\\"}'),
        ("Mixed escapes", '{"k":"a\\\\\\"b"}'),
        ("All structural", '{"a":[{"b":1},{"c":2}]}'),
    ]

    for name, text in fixed:
        n = len(text)
        mask = (1 << n) - 1

        bs_result = run_bs(program, text)
        ref = mison_reference(text)

        for key in ['real_quote', 'str_mask', 'structural']:
            bs_val = bs_result[key] & mask
            ref_val = ref[key]
            assert bs_val == ref_val, (
                f"{name} [{key}]: bs={bs_val:#x} ref={ref_val:#x}"
            )

        b = text_to_basis_streams(text)
        even_bits = make_even_bits(n)
        log_w = compute_log_w(n)

        # Sanitize name for filename
        safe_name = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
        data_filename = f"mison_{safe_name}.bsdata"
        write_bsdata(
            os.path.join(TESTS_DIR, data_filename),
            n,
            inputs={"even_bits": even_bits},
            params={"LOG_W": log_w},
            input_arrays={"b": b},
            expected={
                "structural": ref["structural"],
                "str_mask": ref["str_mask"],
                "real_quote": ref["real_quote"],
            },
        )
        tests.append({
            "name": name,
            "bitlength": n,
            "data_file": data_filename,
        })

    # -- Random: 64B JSON x100 ------------------------------------------------
    rng = random.Random(42)
    sub_cases = []
    for _ in range(100):
        text = random_json(rng, 64)
        n = len(text)
        mask = (1 << n) - 1

        bs_result = run_bs(program, text)
        ref = mison_reference(text)

        for key in ['real_quote', 'str_mask', 'structural']:
            bs_val = bs_result[key] & mask
            ref_val = ref[key]
            assert bs_val == ref_val, f"Random 64B [{key}] mismatch"

        b = text_to_basis_streams(text)
        even_bits = make_even_bits(n)
        log_w = compute_log_w(n)

        sub_cases.append({
            "bitlength": n,
            "inputs": {"even_bits": even_bits},
            "params": {"LOG_W": log_w},
            "input_arrays": {"b": b},
            "expected": {
                "structural": ref["structural"],
                "str_mask": ref["str_mask"],
                "real_quote": ref["real_quote"],
            },
        })
    write_bsdata(
        os.path.join(TESTS_DIR, "mison_random_64b_100.bsdata"),
        100,
        cases=sub_cases,
    )
    tests.append({
        "name": "Random 64B JSON x100",
        "bitlength": 100,
        "data_file": "mison_random_64b_100.bsdata",
    })

    # -- Random: 256B JSON x50 ------------------------------------------------
    rng = random.Random(43)
    sub_cases = []
    for _ in range(50):
        text = random_json(rng, 256)
        n = len(text)
        mask = (1 << n) - 1

        bs_result = run_bs(program, text)
        ref = mison_reference(text)

        for key in ['real_quote', 'str_mask', 'structural']:
            bs_val = bs_result[key] & mask
            ref_val = ref[key]
            assert bs_val == ref_val, f"Random 256B [{key}] mismatch"

        b = text_to_basis_streams(text)
        even_bits = make_even_bits(n)
        log_w = compute_log_w(n)

        sub_cases.append({
            "bitlength": n,
            "inputs": {"even_bits": even_bits},
            "params": {"LOG_W": log_w},
            "input_arrays": {"b": b},
            "expected": {
                "structural": ref["structural"],
                "str_mask": ref["str_mask"],
                "real_quote": ref["real_quote"],
            },
        })
    write_bsdata(
        os.path.join(TESTS_DIR, "mison_random_256b_50.bsdata"),
        50,
        cases=sub_cases,
    )
    tests.append({
        "name": "Random 256B JSON x50",
        "bitlength": 50,
        "data_file": "mison_random_256b_50.bsdata",
    })

    # -- Large 1M-bit wide (precomputed, stored as data_file) -----------------
    W = 1000000
    rng_wide = random.Random(42)
    b_wide = {k: rng_wide.getrandbits(W) for k in range(7)}
    b_wide[7] = 0  # ASCII range
    even_bits_wide = make_even_bits(W)
    log_w_wide = compute_log_w(W)

    interp = Interpreter()
    result_wide = interp.run(
        program,
        inputs={"even_bits": even_bits_wide},
        params={"LOG_W": log_w_wide},
        input_arrays={"b": b_wide},
    )

    mask_wide = (1 << W) - 1
    bs_structural = result_wide['structural'] & mask_wide
    bs_str_mask = result_wide['str_mask'] & mask_wide
    bs_real_quote = result_wide['real_quote'] & mask_wide

    # Invariant check: structural chars should be outside strings
    assert bs_structural & bs_str_mask == 0, \
        "INVARIANT FAIL: structural & str_mask != 0"

    # Spot-check character classification at random positions
    struct_chars = set('{}[]:,')
    positions = rng_wide.sample(range(W), 200)
    for pos in positions:
        code = sum(((b_wide[k] >> pos) & 1) << k for k in range(8))
        ch = chr(code) if code < 128 else '?'
        is_structural_char = ch in struct_chars
        pos_is_structural = (bs_structural >> pos) & 1
        pos_in_string = (bs_str_mask >> pos) & 1
        if is_structural_char and not pos_in_string:
            assert pos_is_structural, (
                f"spot-check pos {pos}: char '{ch}' outside string but not structural")
        elif pos_is_structural and not is_structural_char:
            assert False, (
                f"spot-check pos {pos}: char '{ch}' marked structural but is not")

    write_bsdata(
        os.path.join(TESTS_DIR, "mison_large_1m.bsdata"),
        W,
        inputs={"even_bits": even_bits_wide & mask_wide},
        params={"LOG_W": log_w_wide},
        input_arrays={"b": {k: v & mask_wide for k, v in b_wide.items()}},
        expected={
            "structural": bs_structural & mask_wide,
            "str_mask": bs_str_mask & mask_wide,
            "real_quote": bs_real_quote & mask_wide,
        },
    )
    tests.append({
        "name": "Large 1M-bit wide",
        "bitlength": W,
        "data_file": "mison_large_1m.bsdata",
    })

    # -- Real-data tests (optional, require numpy + .npz files) ---------------
    tests.append({
        "name": "GH Archive 1M",
        "category": "generated",
        "optional": True,
        "bitlength": 1000000,
        "generate": {
            "type": "file",
            "dataset": "gharchive_small",
        },
    })
    tests.append({
        "name": "GH Archive 10M",
        "category": "generated",
        "optional": True,
        "bitlength": 10000000,
        "generate": {
            "type": "file",
            "dataset": "gharchive_medium",
        },
    })
    tests.append({
        "name": "GH Archive 100M",
        "category": "generated",
        "optional": True,
        "bitlength": 100000000,
        "generate": {
            "type": "file",
            "dataset": "gharchive_large",
        },
    })

    # -- Preserve existing tier entries and write tests.json -------------------
    existing = load_tests_json()
    tier_data_files = {f"mison_tier_{t}.bsdata" for t in TIER_NAMES}
    for entry in existing:
        if entry.get("data_file") in tier_data_files:
            tests.append(entry)

    save_tests_json(tests)
    print(f"mison: generated {len(tests)} test entries")


if __name__ == "__main__":
    main()
