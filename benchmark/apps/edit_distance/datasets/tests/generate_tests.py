#!/usr/bin/env python3
"""Generate precomputed bitstream-level test data for Edit Distance (Myers).

Usage:
    python generate_tests.py                # unit tests only (existing behavior)
    python generate_tests.py --tier small   # unit tests + small tier (11.1M vectors)
    python generate_tests.py --tier medium  # unit tests + medium tier (111M vectors)
    python generate_tests.py --tier large   # unit tests + large tier (1.11B vectors)
    python generate_tests.py --describe     # print provenance info
    python generate_tests.py --verify       # verify SHA-256 of existing files
"""

import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter
from benchmark.apps.edit_distance.src.run import (
    myers_reference, encode_text_columns, encode_pattern,
    encode_dist_init, decode_distance, DNA,
)
from benchmark.bsdata import write_bsdata, read_bsdata
from benchmark.tier_generate import (
    parse_generate_args, file_sha256, make_provenance,
    tier_test_entry, print_describe, verify_files,
)
from benchmark.tier_config import get_tier_vectors

TESTS_DIR = os.path.dirname(__file__)
BS_PATH = os.path.join(TESTS_DIR, "../../src/edit_distance.bs")

DOMAIN = "edit_distance"

# Seed offsets for each tier
TIER_SEEDS = {"small": 1000, "medium": 1001, "large": 1002}

# Tier display names
TIER_NAMES = {
    "small":  "Tier small (11.1M texts)",
    "medium": "Tier medium (111M texts)",
    "large":  "Tier large (1.11B texts)",
}

# Fixed parameters for tiered data
TIER_N = 8       # text length
TIER_M = 4       # pattern length
TIER_B = 4       # max(N, M).bit_length()
TIER_PATTERN = "ACGT"


def run_bs_test(program, texts, pattern):
    """Run edit distance on texts vs pattern. Returns (distances, ops)."""
    K = len(texts)
    N = max(len(t) for t in texts)
    M = len(pattern)
    B = max(N, M).bit_length()

    # Pad texts to length N
    padded = [t.ljust(N, 'A') for t in texts]

    text_arrays = encode_text_columns(padded, N)
    pat_arrays = encode_pattern(pattern)
    dist_init = encode_dist_init(M, B)

    input_arrays = {}
    for key, val in text_arrays.items():
        input_arrays[key] = val
    for key, val in pat_arrays.items():
        input_arrays[key] = val
    input_arrays["dist_init"] = dist_init

    interp = Interpreter()
    result = interp.run(
        program,
        inputs={},
        params={"N": N, "M": M, "B": B},
        input_arrays=input_arrays,
    )

    distances = decode_distance(result, B, K)
    return distances, interp.op_count


def make_test(name, texts, pattern, program):
    """Build a single test entry: run BS, verify against reference."""
    K = len(texts)
    N = max(len(t) for t in texts)
    M = len(pattern)
    B = max(N, M).bit_length()

    # Pad texts to length N
    padded = [t.ljust(N, 'A') for t in texts]

    text_arrays = encode_text_columns(padded, N)
    pat_arrays = encode_pattern(pattern)
    dist_init = encode_dist_init(M, B)

    input_arrays = {}
    for key, val in text_arrays.items():
        input_arrays[key] = val
    for key, val in pat_arrays.items():
        input_arrays[key] = val
    input_arrays["dist_init"] = dist_init

    interp = Interpreter()
    result = interp.run(
        program,
        inputs={},
        params={"N": N, "M": M, "B": B},
        input_arrays=input_arrays,
    )

    distances = decode_distance(result, B, K)

    # Verify against reference
    for k in range(K):
        ref = myers_reference(padded[k], pattern)
        assert distances[k] == ref, (
            f"{name} text {k}: bs={distances[k]} ref={ref} "
            f"text={padded[k]} pattern={pattern}")

    # Build expected output (bitsliced distance)
    mask = (1 << K) - 1
    expected_dist = {}
    for b in range(B):
        val = result["dist"].get(b, 0) & mask
        expected_dist[b] = val

    # Sanitize name for filename
    safe = name.lower().replace(" ", "_").replace("-", "_")
    data_filename = f"edit_distance_{safe}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_filename)

    write_bsdata(data_path, K,
                 params={"N": N, "M": M, "B": B},
                 input_arrays=input_arrays,
                 expected={"dist": expected_dist})

    return {
        "name": name,
        "bitlength": K,
        "data_file": data_filename,
    }


def extract_text(pos_streams, N, idx):
    """Extract text at position idx from bitsliced position streams.

    pos_streams is a dict with keys like "pos_A", "pos_C", "pos_G", "pos_T",
    each mapping {j: stream} where bit idx of stream j indicates whether
    text[idx] has that nucleotide at position j.
    """
    text = []
    for p in range(N):
        for base in DNA:
            key = f"pos_{base}"
            if (pos_streams[key][p] >> idx) & 1:
                text.append(base)
                break
    return ''.join(text)


def generate_tier_data(program, tier, bitlength):
    """Generate a tier .bsdata file with random DNA text data.

    Uses direct bitstream generation: for each text column position p,
    generate two random streams r0_p, r1_p, then derive 4 nucleotide
    indicator streams (25% each base). This avoids generating K individual
    text strings.

    Returns (data_file, size_bytes).
    """
    seed = TIER_SEEDS[tier]
    N = TIER_N
    M = TIER_M
    B = TIER_B
    K = bitlength
    pattern = TIER_PATTERN

    data_file = f"edit_distance_tier_{tier}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_file)

    print(f"  Generating {tier} tier: {K:,} vectors (seed={seed})...")

    rng = random.Random(seed)

    # Step 1: Generate 32 position streams via 2-bit random encoding
    # For each text column position p, use two random streams to get
    # uniform 25% distribution across A, C, G, T
    t0 = time.time()
    pos_streams = {f"pos_{ch}": {} for ch in DNA}
    for p in range(N):
        r0 = rng.getrandbits(K)
        r1 = rng.getrandbits(K)
        not_r0 = ((1 << K) - 1) ^ r0
        not_r1 = ((1 << K) - 1) ^ r1
        pos_streams["pos_A"][p] = not_r0 & not_r1   # 00 -> A
        pos_streams["pos_C"][p] = not_r0 & r1        # 01 -> C
        pos_streams["pos_G"][p] = r0 & not_r1        # 10 -> G
        pos_streams["pos_T"][p] = r0 & r1            # 11 -> T
    t_gen = time.time() - t0
    print(f"    Input generation: {t_gen:.1f}s")

    # Step 2: Encode pattern (broadcast constants)
    pat_arrays = encode_pattern(pattern)

    # Step 3: Encode dist_init (broadcast of M=4 in B=4 bits)
    dist_init = encode_dist_init(M, B)

    # Merge all input arrays
    input_arrays = {}
    for key, val in pos_streams.items():
        input_arrays[key] = val
    for key, val in pat_arrays.items():
        input_arrays[key] = val
    input_arrays["dist_init"] = dist_init

    # Step 4: Run interpreter to compute expected outputs
    t0 = time.time()
    interp = Interpreter()
    result = interp.run(
        program,
        inputs={},
        params={"N": N, "M": M, "B": B},
        input_arrays=input_arrays,
    )
    t_interp = time.time() - t0
    print(f"    Interpreter: {t_interp:.1f}s ({interp.op_count} ops)")

    # Step 5: Mask outputs to K bits
    mask = (1 << K) - 1
    expected_dist = {}
    for b in range(B):
        expected_dist[b] = result["dist"].get(b, 0) & mask

    # Step 6: Spot-check random positions against scalar reference
    t0 = time.time()
    num_checks = min(500, K)
    check_rng = random.Random(seed + 5000)
    indices = sorted(check_rng.sample(range(K), num_checks))
    for idx in indices:
        text = extract_text(pos_streams, N, idx)
        got_dist = 0
        for b in range(B):
            if (expected_dist[b] >> idx) & 1:
                got_dist |= 1 << b
        ref_dist = myers_reference(text, pattern)
        assert got_dist == ref_dist, (
            f"Spot-check FAIL at index {idx}: got={got_dist} ref={ref_dist} "
            f"text={text} pattern={pattern}")
    t_check = time.time() - t0
    print(f"    Spot-check: {num_checks} positions verified in {t_check:.1f}s")

    # Step 7: Write .bsdata
    t0 = time.time()
    write_bsdata(
        data_path, K,
        params={"N": N, "M": M, "B": B},
        input_arrays=input_arrays,
        expected={"dist": expected_dist},
    )
    t_write = time.time() - t0

    size_bytes = os.path.getsize(data_path)
    print(f"    Written: {data_file} ({size_bytes:,} bytes, "
          f"{size_bytes / 1e6:.1f} MB, {t_write:.1f}s)")

    return data_file, size_bytes


def generate_unit_tests(program):
    """Generate the original unit tests (unchanged)."""
    tests = []

    # -- Fixed inline tests -----------------------------------------------
    # Test 1: Exact match (distance = 0)
    tests.append(make_test(
        "Exact match",
        ["ACGT"], "ACGT", program))

    # Test 2: Single substitution (distance = 1)
    tests.append(make_test(
        "Single substitution",
        ["ACGT"], "ACGA", program))

    # Test 3: Single insertion (distance = 1)
    # text="ACGTA" (len 5), pattern="ACGT" (len 4) -> dist 1
    tests.append(make_test(
        "Single insertion",
        ["ACGTA"], "ACGT", program))

    # Test 4: Two substitutions (distance = 2)
    # text="ACTA", pattern="ACGT" -> C/G and A/T substitutions.
    tests.append(make_test(
        "Two substitutions",
        ["ACTA"], "ACGT", program))

    # Test 5: Full mismatch (distance = max)
    tests.append(make_test(
        "Full mismatch",
        ["AAAA"], "CCCC", program))

    # Test 6: Multiple texts at once
    tests.append(make_test(
        "Multi-text batch",
        ["ACGT", "ACGA", "AAAA", "CCCC", "TGCA", "ACGT", "GGGC", "TTTT"],
        "ACGT", program))

    # -- Random 16-text x50 batches (data file) ---------------------------
    rng = random.Random(42)
    sub_cases = []
    for _ in range(50):
        M = rng.randint(3, 6)
        N = rng.randint(M, M + 3)
        K = 16
        pattern = ''.join(rng.choice(DNA) for _ in range(M))
        texts = [''.join(rng.choice(DNA) for _ in range(N)) for _ in range(K)]

        B = max(N, M).bit_length()
        text_arrays = encode_text_columns(texts, N)
        pat_arrays = encode_pattern(pattern)
        dist_init_arr = encode_dist_init(M, B)

        input_arrays = {}
        for key, val in text_arrays.items():
            input_arrays[key] = val
        for key, val in pat_arrays.items():
            input_arrays[key] = val
        input_arrays["dist_init"] = dist_init_arr

        interp = Interpreter()
        result = interp.run(
            program, inputs={}, params={"N": N, "M": M, "B": B},
            input_arrays=input_arrays)
        distances = decode_distance(result, B, K)

        # Verify
        for k in range(K):
            ref = myers_reference(texts[k], pattern)
            assert distances[k] == ref, f"Random test: bs={distances[k]} ref={ref}"

        mask = (1 << K) - 1
        expected_dist = {}
        for b in range(B):
            expected_dist[b] = result["dist"].get(b, 0) & mask

        sub_cases.append({
            "bitlength": K,
            "params": {"N": N, "M": M, "B": B},
            "input_arrays": input_arrays,
            "expected": {"dist": expected_dist},
        })

    data_filename = "edit_distance_random_16text_50.bsdata"
    write_bsdata(os.path.join(TESTS_DIR, data_filename), 50,
                 cases=sub_cases)
    tests.append({
        "name": "Random 16-text x50",
        "bitlength": 50,
        "data_file": data_filename,
    })

    # -- Wide generated tests ---------------------------------------------
    tests.append({
        "name": "Wide 1K texts N=8 M=4",
        "category": "generated",
        "bitlength": 1000,
        "generate": {"K": 1000, "N": 8, "M": 4, "seed": 42},
    })

    tests.append({
        "name": "Wide 10K texts N=6 M=4",
        "category": "generated",
        "bitlength": 10000,
        "generate": {"K": 10000, "N": 6, "M": 4, "seed": 43},
    })

    tests.append({
        "name": "Large 1M texts N=8 M=4",
        "category": "generated",
        "bitlength": 1000000,
        "generate": {"K": 1000000, "N": 8, "M": 4, "seed": 44},
    })

    return tests


def load_tests_json():
    """Load existing tests.json if it exists."""
    path = os.path.join(TESTS_DIR, "tests.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_tests_json(tests):
    """Write tests.json."""
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


def do_describe():
    """Print tier descriptions and estimated sizes."""
    print("Edit Distance (Myers) tier configurations:")
    print(f"  Fixed params: N={TIER_N}, M={TIER_M}, B={TIER_B}, "
          f"pattern=\"{TIER_PATTERN}\"")
    print(f"  Streams: 36 (4 nucleotides x {TIER_N} text positions "
          f"= 32 input + {TIER_B} output)")
    print()

    # Also print existing tests.json provenance
    tests_json_path = os.path.join(TESTS_DIR, "tests.json")
    if os.path.exists(tests_json_path):
        with open(tests_json_path) as f:
            tests = json.load(f)
        print_describe(TESTS_DIR, tests)
        print()

    for tier_name in ["small", "medium", "large"]:
        bitlength = get_tier_vectors(DOMAIN, tier_name)
        seed = TIER_SEEDS[tier_name]
        est_bytes = 36 * math.ceil(bitlength / 8)
        data_file = f"edit_distance_tier_{tier_name}.bsdata"
        data_path = os.path.join(TESTS_DIR, data_file)
        exists = os.path.exists(data_path)
        actual_size = os.path.getsize(data_path) if exists else None
        print(f"  {tier_name}:")
        print(f"    label:     {TIER_NAMES[tier_name]}")
        print(f"    bitlength: {bitlength:,}")
        print(f"    seed:      {seed}")
        print(f"    est. size: {est_bytes:,} bytes ({est_bytes / 1e6:.1f} MB)")
        print(f"    file:      {data_file}")
        if exists:
            print(f"    status:    EXISTS ({actual_size:,} bytes)")
        else:
            print(f"    status:    not generated")
        print()


def do_verify():
    """Verify existing .bsdata files by spot-checking against Myers reference."""
    tests_json_path = os.path.join(TESTS_DIR, "tests.json")
    if not os.path.exists(tests_json_path):
        print("ERROR: tests.json not found")
        sys.exit(1)

    with open(tests_json_path) as f:
        tests = json.load(f)

    # Use verify_files for SHA-256 checks on tier entries
    print("edit_distance: verifying SHA-256 checksums")
    ok = verify_files(TESTS_DIR, tests)

    # Additionally spot-check tier .bsdata files
    for entry in tests:
        data_file = entry.get("data_file", "")
        if not data_file or not data_file.startswith("edit_distance_tier_"):
            continue
        data_path = os.path.join(TESTS_DIR, data_file)
        name = entry["name"]
        if not os.path.exists(data_path):
            print(f"  SKIP {name}: {data_file} not found")
            continue

        print(f"  Spot-checking {name} ({entry['bitlength']:,} vectors)...")
        t0 = time.time()
        data = read_bsdata(data_path)

        bitlength = data["bitlength"]
        ia = data.get("input_arrays", {})
        exp = data.get("expected", {})

        # Reconstruct pos_streams dict for extract_text
        pos_streams = {}
        for ch in DNA:
            key = f"pos_{ch}"
            pos_streams[key] = ia.get(key, {})

        dist_arrays = exp.get("dist", {})
        N = data.get("params", {}).get("N", TIER_N)
        B = data.get("params", {}).get("B", TIER_B)

        # Extract pattern from eq_ arrays
        M = data.get("params", {}).get("M", TIER_M)
        pattern_chars = []
        for k in range(M):
            for ch in DNA:
                eq_key = f"eq_{ch}"
                val = ia.get(eq_key, {}).get(k, 0)
                if val != 0:  # -1 or all-ones means match
                    pattern_chars.append(ch)
                    break
        pattern = ''.join(pattern_chars) if pattern_chars else TIER_PATTERN

        num_checks = min(500, bitlength)
        check_rng = random.Random(99999)
        indices = sorted(check_rng.sample(range(bitlength), num_checks))
        fail = False
        for idx in indices:
            text = extract_text(pos_streams, N, idx)
            got_dist = 0
            for b in range(B):
                if (dist_arrays.get(b, 0) >> idx) & 1:
                    got_dist |= 1 << b
            ref_dist = myers_reference(text, pattern)
            if got_dist != ref_dist:
                print(f"    FAIL at index {idx}: got={got_dist} ref={ref_dist} "
                      f"text={text} pattern={pattern}")
                ok = False
                fail = True
                break
        if not fail:
            t_total = time.time() - t0
            print(f"    OK: {num_checks} spot-checks passed ({t_total:.1f}s)")

    sys.exit(0 if ok else 1)


def main():
    args = parse_generate_args(DOMAIN)

    # Handle --describe and --verify
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
        # Tier generation mode: generate specified tier, merge into tests.json
        tier = args.tier
        bitlength = get_tier_vectors(DOMAIN, tier)
        if bitlength is None:
            print(f"edit_distance: tier '{tier}' not applicable (skipped)")
            return

        # Load existing tests.json to preserve unit tests
        tests = load_tests_json()

        data_file, size_bytes = generate_tier_data(program, tier, bitlength)
        sha = file_sha256(os.path.join(TESTS_DIR, data_file))
        prov = make_provenance(
            source="synthetic",
            seed=TIER_SEEDS[tier],
            description=(
                f"Random DNA texts of length N={TIER_N} vs fixed pattern "
                f"\"{TIER_PATTERN}\" (M={TIER_M}). "
                f"32 position streams generated via 2-bit random encoding "
                f"(uniform 25% per nucleotide). "
                f"{bitlength:,} texts. "
                f"Expected via Python interpreter; spot-checked 500 positions."
            ),
            generated_by=f"generate_tests.py --tier {tier}",
        )
        prov["sha256"] = sha
        entry = tier_test_entry(
            name=TIER_NAMES[tier],
            bitlength=bitlength,
            data_file=data_file,
            size_bytes=size_bytes,
            provenance=prov,
        )
        tests = merge_tier_entry(tests, entry)
        save_tests_json(tests)
    else:
        # Default: generate unit tests (original behavior)
        tests = generate_unit_tests(program)

        # Preserve any existing tier entries in tests.json
        existing = load_tests_json()
        tier_data_files = {f"edit_distance_tier_{t}.bsdata"
                          for t in TIER_SEEDS}
        for entry in existing:
            if entry.get("data_file") in tier_data_files:
                tests.append(entry)

        save_tests_json(tests)
        print(f"edit_distance: generated {len(tests)} test entries")


if __name__ == "__main__":
    main()
