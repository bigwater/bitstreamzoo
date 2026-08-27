#!/usr/bin/env python3
"""Generate precomputed bitstream-level test data for Shift-And.

Usage:
    python generate_tests.py                # unit tests only (existing behavior)
    python generate_tests.py --tier small   # unit tests + small tier (80M vectors)
    python generate_tests.py --tier medium  # unit tests + medium tier (800M vectors)
    python generate_tests.py --tier large   # unit tests + large tier (8B vectors)
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
from benchmark.apps.shift_and.src.run import (
    shift_and_reference, random_dna, ALPHABET,
)
from benchmark.bsdata import write_bsdata
from benchmark.tier_generate import (
    parse_generate_args, file_sha256, make_provenance,
    tier_test_entry, print_describe, verify_files,
    getrandbits_large,
)
from benchmark.tier_config import get_tier_vectors

TESTS_DIR = os.path.dirname(__file__)
BS_PATH = os.path.join(TESTS_DIR, "../../src/shift_and.bs")

DOMAIN = "shift_and"

# Fixed pattern for tier tests
TIER_PATTERN = "GATTACA"

# Seed offsets for each tier
TIER_SEEDS = {"small": 1000, "medium": 1001, "large": 1002}

# Tier display names
TIER_NAMES = {
    "small":  "Tier small (80M bases)",
    "medium": "Tier medium (800M bases)",
    "large":  "Tier large (8B bases)",
}


def tier_description(bitlength):
    """Canonical provenance description for a tier data file.

    Single source of truth for both tier generation and the preserve
    path of full regeneration, so a stale description cannot survive.
    The data is uniform random per base (~25% in expectation, not
    exactly balanced): each base comes from independent random bit
    streams with no rebalancing step.
    """
    return (
        f"Uniform random DNA (each base ~25% in expectation, not "
        f"exactly balanced). "
        f"Basis streams from 2 uniform random streams: "
        f"is_A=~r0&~r1, is_C=~r0&r1, is_G=r0&~r1, is_T=r0&r1. "
        f"Pattern: {TIER_PATTERN} (M={len(TIER_PATTERN)}). "
        f"{bitlength:,} bases."
    )


def encode_text(text):
    """Encode DNA text to basis streams: is_A, is_C, is_G, is_T."""
    N = len(text)
    basis = {c: 0 for c in ALPHABET}
    for k in range(N):
        basis[text[k]] |= 1 << k
    return basis


def encode_pattern(pattern):
    """Encode pattern to mask arrays: mask_X[j] = -1 if pattern[j]==X, else 0."""
    M = len(pattern)
    mask_arrays = {}
    for c in ALPHABET:
        arr = {}
        for j in range(M):
            arr[j] = -1 if pattern[j] == c else 0
        mask_arrays[c] = arr
    return mask_arrays


def run_bs(program, text, pattern):
    """Run interpreter on text/pattern, return matches stream."""
    N = len(text)
    M = len(pattern)
    basis = encode_text(text)
    mask_arrays = encode_pattern(pattern)

    interp = Interpreter()
    result = interp.run(
        program,
        inputs={
            "is_A": basis["A"], "is_C": basis["C"],
            "is_G": basis["G"], "is_T": basis["T"],
        },
        params={"M": M},
        input_arrays={
            "mask_A": mask_arrays["A"], "mask_C": mask_arrays["C"],
            "mask_G": mask_arrays["G"], "mask_T": mask_arrays["T"],
        },
    )
    return result["matches"]


def matches_to_stream(text, pattern):
    """Convert set of match positions to a bitmask stream."""
    ref_set = shift_and_reference(text, pattern)
    stream = 0
    for pos in ref_set:
        stream |= 1 << pos
    return stream


def shift_and_bitstream_reference(basis, pattern, bitlength):
    """Compute Shift-And expected output directly on basis streams.

    This replicates the .bs program logic using pure Python bitwise ops
    on large integers, which is efficient for tier-scale bitlength where
    the interpreter would be too slow.

    Args:
        basis: dict with keys 'A','C','G','T' -> int bitstreams
        pattern: str, the DNA pattern to match
        bitlength: int, number of bit positions (text length)

    Returns:
        int, the matches bitstream
    """
    M = len(pattern)
    mask = (1 << bitlength) - 1

    # Build mask arrays: for each base c and pattern position j,
    # mask is all-ones if pattern[j]==c, else all-zeros
    # In the bitstream domain, all-ones = mask (bitlength bits set)
    # and all-zeros = 0
    mask_arrays = {}
    for c in ALPHABET:
        arr = {}
        for j in range(M):
            arr[j] = mask if pattern[j] == c else 0
        mask_arrays[c] = arr

    # Position 0: character match = OR of (basis[c] & mask_c[0]) for each base
    R = 0
    for c in ALPHABET:
        R |= basis[c] & mask_arrays[c][0]

    # Positions 1..M-1: shift-and accumulation
    for j in range(1, M):
        cm = 0
        for c in ALPHABET:
            cm |= basis[c] & mask_arrays[c][j]
        R = ((R << 1) & mask) & cm

    return R & mask


def generate_tier_data(tier, bitlength):
    """Generate a tier .bsdata file with random DNA basis streams.

    Uses 2 random streams (r0, r1) to construct 4 mutually exclusive
    basis streams, each base uniform random (~25% in expectation, not
    exactly balanced):
        is_A = ~r0 & ~r1
        is_C = ~r0 &  r1
        is_G =  r0 & ~r1
        is_T =  r0 &  r1

    Pattern: fixed "GATTACA" (M=7).

    Returns (data_file, size_bytes).
    """
    seed = TIER_SEEDS[tier]
    pattern = TIER_PATTERN
    M = len(pattern)
    mask = (1 << bitlength) - 1
    rng = random.Random(seed)

    print(f"  Generating {tier} tier: {bitlength:,} vectors (seed={seed})...")

    # Generate 2 random streams to construct 4 basis streams
    print(f"    Generating r0 ({bitlength:,} bits)...")
    r0 = getrandbits_large(rng, bitlength)
    print(f"    Generating r1 ({bitlength:,} bits)...")
    r1 = getrandbits_large(rng, bitlength)

    not_r0 = (~r0) & mask
    not_r1 = (~r1) & mask

    basis = {
        "A": not_r0 & not_r1,
        "C": not_r0 & r1,
        "G": r0 & not_r1,
        "T": r0 & r1,
    }

    # Sanity: basis streams are mutually exclusive and cover all positions
    union = basis["A"] | basis["C"] | basis["G"] | basis["T"]
    assert union == mask, "Basis streams do not cover all positions"
    assert (basis["A"] & basis["C"]) == 0, "A and C overlap"

    # Compute expected output via direct bitstream shift-and (no interpreter)
    print(f"    Computing expected output via bitstream shift-and...")
    expected_matches = shift_and_bitstream_reference(basis, pattern, bitlength)

    # Count matches for diagnostics
    n_matches = bin(expected_matches).count('1')
    print(f"    Expected matches: {n_matches:,} / {bitlength:,} "
          f"({100 * n_matches / bitlength:.4f}%)")

    # Build mask arrays for the pattern
    mask_arrays = encode_pattern(pattern)

    # Write .bsdata
    data_file = f"shift_and_tier_{tier}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_file)
    print(f"    Writing {data_file}...")
    write_bsdata(
        data_path,
        bitlength,
        inputs={
            "is_A": basis["A"], "is_C": basis["C"],
            "is_G": basis["G"], "is_T": basis["T"],
        },
        params={"M": M},
        input_arrays={
            "mask_A": mask_arrays["A"], "mask_C": mask_arrays["C"],
            "mask_G": mask_arrays["G"], "mask_T": mask_arrays["T"],
        },
        expected={"matches": expected_matches},
    )
    size_bytes = os.path.getsize(data_path)
    print(f"    Wrote {data_file} ({size_bytes:,} bytes, {size_bytes / 1e6:.1f} MB)")

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
            print(f"shift_and: {len(tests)} test entries")
            print_describe(TESTS_DIR, tests)
        if args.verify:
            print(f"shift_and: verifying SHA-256 checksums")
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
            print(f"shift_and: tier '{tier}' not applicable (skipped)")
            return
        data_file, size_bytes = generate_tier_data(tier, bitlength)
        sha = file_sha256(os.path.join(TESTS_DIR, data_file))
        prov = make_provenance(
            source="synthetic",
            seed=TIER_SEEDS[tier],
            description=tier_description(bitlength),
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
        print(f"shift_and: tier '{tier}' merged into tests.json")
        return

    tests = []

    # -- Fixed tests ----------------------------------------------------------
    fixed = [
        ("Single char", "ACGTACGT", "A"),
        ("Dinucleotide", "ACGTACGT", "CG"),
        ("4-mer match", "AACGATCGATCG", "ATCG"),
        ("No match", "AAAAAAAAAA", "CG"),
        ("Full text match", "ACGT", "ACGT"),
        ("Overlapping matches", "AAAAAAA", "AAA"),
    ]

    for name, text, pattern in fixed:
        N = len(text)
        M = len(pattern)
        mask = (1 << N) - 1

        bs_result = run_bs(program, text, pattern)
        ref_stream = matches_to_stream(text, pattern)
        expected = bs_result & mask
        assert expected == (ref_stream & mask), f"{name}: bs={bs_result:#x} ref={ref_stream:#x}"

        basis = encode_text(text)
        mask_arrays = encode_pattern(pattern)

        safe = name.lower().replace(" ", "_").replace("-", "_")
        data_filename = f"shift_and_{safe}.bsdata"
        write_bsdata(os.path.join(TESTS_DIR, data_filename), N,
                     inputs={
                         "is_A": basis["A"],
                         "is_C": basis["C"],
                         "is_G": basis["G"],
                         "is_T": basis["T"],
                     },
                     params={"M": M},
                     input_arrays={
                         "mask_A": mask_arrays["A"],
                         "mask_C": mask_arrays["C"],
                         "mask_G": mask_arrays["G"],
                         "mask_T": mask_arrays["T"],
                     },
                     expected={"matches": expected})

        tests.append({
            "name": name,
            "bitlength": N,
            "data_file": data_filename,
        })

    # -- Random: TATA box -----------------------------------------------------
    prefix = random_dna(200, 10)
    suffix = random_dna(200, 11)
    text = prefix + "TATAAAT" + suffix
    pattern = "TATAAAT"
    N = len(text)
    M = len(pattern)
    mask = (1 << N) - 1
    bs_result = run_bs(program, text, pattern)
    ref_stream = matches_to_stream(text, pattern)
    expected = bs_result & mask
    assert expected == (ref_stream & mask), f"TATA box mismatch"
    basis = encode_text(text)
    mask_arrays = encode_pattern(pattern)

    write_bsdata(os.path.join(TESTS_DIR, "shift_and_tata_box.bsdata"), N,
                 inputs={
                     "is_A": basis["A"], "is_C": basis["C"],
                     "is_G": basis["G"], "is_T": basis["T"],
                 },
                 params={"M": M},
                 input_arrays={
                     "mask_A": mask_arrays["A"], "mask_C": mask_arrays["C"],
                     "mask_G": mask_arrays["G"], "mask_T": mask_arrays["T"],
                 },
                 expected={"matches": expected})
    tests.append({
        "name": "TATA box",
        "bitlength": N,
        "data_file": "shift_and_tata_box.bsdata",
    })

    # -- Random: Kozak-like ---------------------------------------------------
    prefix = random_dna(100, 20)
    suffix = random_dna(100, 21)
    text = prefix + "ACCATGG" + suffix
    pattern = "ACCATGG"
    N = len(text)
    M = len(pattern)
    mask = (1 << N) - 1
    bs_result = run_bs(program, text, pattern)
    ref_stream = matches_to_stream(text, pattern)
    expected = bs_result & mask
    assert expected == (ref_stream & mask), f"Kozak-like mismatch"
    basis = encode_text(text)
    mask_arrays = encode_pattern(pattern)

    write_bsdata(os.path.join(TESTS_DIR, "shift_and_kozak_like.bsdata"), N,
                 inputs={
                     "is_A": basis["A"], "is_C": basis["C"],
                     "is_G": basis["G"], "is_T": basis["T"],
                 },
                 params={"M": M},
                 input_arrays={
                     "mask_A": mask_arrays["A"], "mask_C": mask_arrays["C"],
                     "mask_G": mask_arrays["G"], "mask_T": mask_arrays["T"],
                 },
                 expected={"matches": expected})
    tests.append({
        "name": "Kozak-like",
        "bitlength": N,
        "data_file": "shift_and_kozak_like.bsdata",
    })

    # -- Random: 4-mer 500bp --------------------------------------------------
    text = random_dna(500, 30)
    pattern = random_dna(4, 31)
    N = len(text)
    M = len(pattern)
    mask = (1 << N) - 1
    bs_result = run_bs(program, text, pattern)
    ref_stream = matches_to_stream(text, pattern)
    expected = bs_result & mask
    assert expected == (ref_stream & mask), f"Random 4-mer 500bp mismatch"
    basis = encode_text(text)
    mask_arrays = encode_pattern(pattern)

    write_bsdata(os.path.join(TESTS_DIR, "shift_and_random_4mer_500bp.bsdata"), N,
                 inputs={
                     "is_A": basis["A"], "is_C": basis["C"],
                     "is_G": basis["G"], "is_T": basis["T"],
                 },
                 params={"M": M},
                 input_arrays={
                     "mask_A": mask_arrays["A"], "mask_C": mask_arrays["C"],
                     "mask_G": mask_arrays["G"], "mask_T": mask_arrays["T"],
                 },
                 expected={"matches": expected})
    tests.append({
        "name": "Random 4-mer 500bp",
        "bitlength": N,
        "data_file": "shift_and_random_4mer_500bp.bsdata",
    })

    # -- Random: 8-mer 500bp --------------------------------------------------
    text = random_dna(500, 40)
    pattern = random_dna(8, 41)
    N = len(text)
    M = len(pattern)
    mask = (1 << N) - 1
    bs_result = run_bs(program, text, pattern)
    ref_stream = matches_to_stream(text, pattern)
    expected = bs_result & mask
    assert expected == (ref_stream & mask), f"Random 8-mer 500bp mismatch"
    basis = encode_text(text)
    mask_arrays = encode_pattern(pattern)

    write_bsdata(os.path.join(TESTS_DIR, "shift_and_random_8mer_500bp.bsdata"), N,
                 inputs={
                     "is_A": basis["A"], "is_C": basis["C"],
                     "is_G": basis["G"], "is_T": basis["T"],
                 },
                 params={"M": M},
                 input_arrays={
                     "mask_A": mask_arrays["A"], "mask_C": mask_arrays["C"],
                     "mask_G": mask_arrays["G"], "mask_T": mask_arrays["T"],
                 },
                 expected={"matches": expected})
    tests.append({
        "name": "Random 8-mer 500bp",
        "bitlength": N,
        "data_file": "shift_and_random_8mer_500bp.bsdata",
    })

    # -- Random: 16-mer 1kbp --------------------------------------------------
    text = random_dna(1000, 50)
    pattern = random_dna(16, 51)
    N = len(text)
    M = len(pattern)
    mask = (1 << N) - 1
    bs_result = run_bs(program, text, pattern)
    ref_stream = matches_to_stream(text, pattern)
    expected = bs_result & mask
    assert expected == (ref_stream & mask), f"Random 16-mer 1kbp mismatch"
    basis = encode_text(text)
    mask_arrays = encode_pattern(pattern)

    write_bsdata(os.path.join(TESTS_DIR, "shift_and_random_16mer_1kbp.bsdata"), N,
                 inputs={
                     "is_A": basis["A"], "is_C": basis["C"],
                     "is_G": basis["G"], "is_T": basis["T"],
                 },
                 params={"M": M},
                 input_arrays={
                     "mask_A": mask_arrays["A"], "mask_C": mask_arrays["C"],
                     "mask_G": mask_arrays["G"], "mask_T": mask_arrays["T"],
                 },
                 expected={"matches": expected})
    tests.append({
        "name": "Random 16-mer 1kbp",
        "bitlength": N,
        "data_file": "shift_and_random_16mer_1kbp.bsdata",
    })

    # -- Random: 4-mer 1kbp ---------------------------------------------------
    text = random_dna(1000, 60)
    pattern = random_dna(4, 61)
    N = len(text)
    M = len(pattern)
    mask = (1 << N) - 1
    bs_result = run_bs(program, text, pattern)
    ref_stream = matches_to_stream(text, pattern)
    expected = bs_result & mask
    assert expected == (ref_stream & mask), f"Random 4-mer 1kbp mismatch"
    basis = encode_text(text)
    mask_arrays = encode_pattern(pattern)

    write_bsdata(os.path.join(TESTS_DIR, "shift_and_random_4mer_1kbp.bsdata"), N,
                 inputs={
                     "is_A": basis["A"], "is_C": basis["C"],
                     "is_G": basis["G"], "is_T": basis["T"],
                 },
                 params={"M": M},
                 input_arrays={
                     "mask_A": mask_arrays["A"], "mask_C": mask_arrays["C"],
                     "mask_G": mask_arrays["G"], "mask_T": mask_arrays["T"],
                 },
                 expected={"matches": expected})
    tests.append({
        "name": "Random 4-mer 1kbp",
        "bitlength": N,
        "data_file": "shift_and_random_4mer_1kbp.bsdata",
    })

    # -- Wide test (generated at runtime) -------------------------------------
    tests.append({
        "name": "Large 10Mbp 32-mer",
        "category": "generated",
        "bitlength": 10000000,
        "generate": {
            "type": "wide",
            "N": 10000000,
            "pattern_len": 32,
            "seed_text": 70,
            "seed_pat": 71,
        },
    })

    # -- Real-data tests (optional, require numpy + .npz files) ---------------
    tests.append({
        "name": "SARS-CoV-2 TATAAAT",
        "category": "generated",
        "optional": True,
        "bitlength": 29903,
        "generate": {
            "type": "file",
            "dataset": "sars_cov2",
            "pattern": "TATAAAT",
        },
    })
    tests.append({
        "name": "E. coli GATTACA",
        "category": "generated",
        "optional": True,
        "bitlength": 4641652,
        "generate": {
            "type": "file",
            "dataset": "ecoli_k12",
            "pattern": "GATTACA",
        },
    })
    tests.append({
        "name": "Human chr1 2M GATTACA",
        "category": "generated",
        "optional": True,
        "bitlength": 2000000,
        "generate": {
            "type": "file",
            "dataset": "hg38_chr1_2m",
            "pattern": "GATTACA",
        },
    })
    tests.append({
        "name": "Human chr1 20M GATTACA",
        "category": "generated",
        "optional": True,
        "bitlength": 20000000,
        "generate": {
            "type": "file",
            "dataset": "hg38_chr1_20m",
            "pattern": "GATTACA",
        },
    })
    tests.append({
        "name": "Human chr1 200M GATTACA",
        "category": "generated",
        "optional": True,
        "bitlength": 200000000,
        "generate": {
            "type": "file",
            "dataset": "hg38_chr1_200m",
            "pattern": "GATTACA",
        },
    })

    # -- Preserve existing tier entries and write tests.json -------------------
    # Refresh each preserved entry's provenance description from the
    # canonical text so a stale equal-frequency claim cannot survive
    # regeneration. seed/sha256/bsim_md5 stay as-is: they describe the
    # actual (unchanged) .bsdata file.
    existing = load_tests_json()
    tier_data_files = {f"shift_and_tier_{t}.bsdata" for t in TIER_NAMES}
    for entry in existing:
        if entry.get("data_file") in tier_data_files:
            prov = entry.get("provenance")
            if prov is not None and "bitlength" in entry:
                prov["description"] = tier_description(entry["bitlength"])
            tests.append(entry)

    save_tests_json(tests)
    print(f"shift_and: generated {len(tests)} test entries")


if __name__ == "__main__":
    main()
