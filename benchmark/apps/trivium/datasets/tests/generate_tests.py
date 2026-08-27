#!/usr/bin/env python3
"""Generate precomputed bitstream-level test data for Trivium.

Usage:
    python generate_tests.py                # unit tests only (existing behavior)
    python generate_tests.py --tier small   # unit tests + small tier (1.79M vectors)
    python generate_tests.py --tier medium  # unit tests + medium tier (17.9M vectors)
    python generate_tests.py --tier large   # unit tests + large tier (179M vectors)
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
from benchmark.apps.trivium.src.run import (
    trivium_reference, hex_to_bits, bits_to_ecrypt_hex,
)
from benchmark.bsdata import write_bsdata
from benchmark.tier_generate import (
    parse_generate_args, file_sha256, make_provenance,
    tier_test_entry, print_describe, verify_files,
)
from benchmark.tier_config import get_tier_vectors

TESTS_DIR = os.path.dirname(__file__)
BS_PATH = os.path.join(TESTS_DIR, "../../src/trivium.bs")

DOMAIN = "trivium"

# Seed offsets for each tier
TIER_SEEDS = {"small": 1000, "medium": 1001, "large": 1002}

# Tier display names
TIER_NAMES = {
    "small":  "Tier small (1.79M key/IV pairs)",
    "medium": "Tier medium (17.9M key/IV pairs)",
    "large":  "Tier large (179M key/IV pairs)",
}

# Standard output length for Trivium
TRIVIUM_L = 64


def pack_key_iv_pairs(keys, ivs):
    """Bit-slice key/IV pairs into input_arrays.

    keys: list of K key bit-lists, each 80 bits.
    ivs: list of K IV bit-lists, each 80 bits.
    Returns: (key_arrays, iv_arrays) dicts.
    """
    K = len(keys)

    key_arrays = {}
    for i in range(80):
        bits = 0
        for j in range(K):
            if keys[j][i]:
                bits |= 1 << j
        key_arrays[i] = bits

    iv_arrays = {}
    for i in range(80):
        bits = 0
        for j in range(K):
            if ivs[j][i]:
                bits |= 1 << j
        iv_arrays[i] = bits

    return key_arrays, iv_arrays


def run_bs(program, length, key_arrays, iv_arrays):
    """Run interpreter, return (result dict, op_count)."""
    interp = Interpreter()
    result = interp.run(
        program,
        inputs={},
        params={"L": length},
        input_arrays={"key": key_arrays, "iv": iv_arrays},
    )
    return result, interp.op_count


def unpack_keystreams(z_array, K, length):
    """Extract K keystream bit-lists from the output array."""
    keystreams = []
    for j in range(K):
        ks = []
        for k in range(length):
            ks.append(1 if z_array.get(k, 0) & (1 << j) else 0)
        keystreams.append(ks)
    return keystreams


def make_test(name, key_hexes, iv_hexes, length, program):
    """Build a single test entry: encode, run BS, verify against reference.

    Returns (test_metadata_dict, fname) for tests.json, and writes a .bsdata file.
    """
    keys = [hex_to_bits(h, 80) for h in key_hexes]
    ivs = [hex_to_bits(h, 80) for h in iv_hexes]
    K = len(keys)

    key_arrays, iv_arrays = pack_key_iv_pairs(keys, ivs)
    result, ops = run_bs(program, length, key_arrays, iv_arrays)
    z_array = result["z"]

    # Verify against reference
    got_ks = unpack_keystreams(z_array, K, length)
    for j in range(K):
        ref_ks = trivium_reference(keys[j], ivs[j], length)
        assert got_ks[j] == ref_ks, f"{name} pair {j}: bitstream != reference"

    # Mask z_array to K bits (the precomputed runner masks actual outputs to
    # bitlength bits, so expected must also be masked)
    mask = (1 << K) - 1
    z_masked = {k: v & mask for k, v in z_array.items()}

    fname = name.lower().replace(" ", "_").replace("/", "_")
    write_bsdata(
        os.path.join(TESTS_DIR, f"{fname}.bsdata"),
        K,
        params={"L": length},
        input_arrays={"key": key_arrays, "iv": iv_arrays},
        expected={"z": z_masked},
    )

    return {
        "name": name,
        "bitlength": K,
        "data_file": f"{fname}.bsdata",
    }


def generate_tier_data(program, tier, bitlength):
    """Generate a tier .bsdata file with random key/IV streams.

    Each of the 80 key streams and 80 IV streams is a uniform random
    bitlength-bit integer (each vector is a different random key/IV pair).
    The interpreter runs Trivium with L=64, producing 64 output streams.
    We spot-check a few positions against the scalar reference.

    Returns (data_file, size_bytes).
    """
    seed = TIER_SEEDS[tier]
    rng = random.Random(seed)
    length = TRIVIUM_L

    print(f"  [{tier}] Generating {bitlength:,} vectors (seed={seed})...")

    # Generate 80 random key streams
    print(f"    Generating 80 key streams ({bitlength:,} bits each)...")
    key_arrays = {i: rng.getrandbits(bitlength) for i in range(80)}

    # Generate 80 random IV streams
    print(f"    Generating 80 IV streams ({bitlength:,} bits each)...")
    iv_arrays = {i: rng.getrandbits(bitlength) for i in range(80)}

    # Run interpreter to compute expected outputs
    print(f"    Running Trivium interpreter (L={length}, {bitlength:,} vectors)...")
    result, ops = run_bs(program, length, key_arrays, iv_arrays)
    z_array = result["z"]
    print(f"    Interpreter finished ({ops:,} ops)")

    # Mask z_array to bitlength bits
    mask = (1 << bitlength) - 1
    z_masked = {k: v & mask for k, v in z_array.items()}

    # Spot-check a few positions against the scalar reference
    n_verify = min(20, bitlength)
    verify_rng = random.Random(seed + 100)
    verify_indices = sorted(verify_rng.sample(range(bitlength), n_verify))
    print(f"    Spot-checking {n_verify} positions against scalar reference...")
    for j in verify_indices:
        key_bits = [(key_arrays[i] >> j) & 1 for i in range(80)]
        iv_bits = [(iv_arrays[i] >> j) & 1 for i in range(80)]
        got_bits = [(z_masked.get(k, 0) >> j) & 1 for k in range(length)]
        ref_bits = trivium_reference(key_bits, iv_bits, length)
        assert got_bits == ref_bits, \
            f"Tier {tier}: spot-check failed at position {j}"
    print(f"    All {n_verify} spot-checks passed")

    # Write .bsdata
    data_file = f"trivium_tier_{tier}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_file)
    print(f"    Writing {data_file}...")
    write_bsdata(
        data_path,
        bitlength,
        params={"L": length},
        input_arrays={"key": key_arrays, "iv": iv_arrays},
        expected={"z": z_masked},
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
            print(f"trivium: {len(tests)} test entries")
            print_describe(TESTS_DIR, tests)
        if args.verify:
            print(f"trivium: verifying SHA-256 checksums")
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
            print(f"trivium: tier '{tier}' not applicable (skipped)")
            return
        data_file, size_bytes = generate_tier_data(program, tier, bitlength)
        sha = file_sha256(os.path.join(TESTS_DIR, data_file))
        prov = make_provenance(
            source="synthetic",
            seed=TIER_SEEDS[tier],
            description=(
                f"Random key/IV pairs: 80 key streams + 80 IV streams "
                f"(uniform random), L={TRIVIUM_L} output streams. "
                f"{bitlength:,} vectors (key/IV pairs). "
                f"224 total binary streams."
            ),
            generated_by="generate_tests.py --tier " + tier,
            sha256=sha,
        )
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
        print(f"trivium: tier '{tier}' merged into tests.json")
        return

    tests = []

    # -- Zero key/IV 64b -----------------------------------------------
    tests.append(make_test(
        "Zero key/IV 64b",
        ["00000000000000000000"], ["00000000000000000000"],
        64, program,
    ))

    # -- Key bit-list index 79 64b -------------------------------------
    tests.append(make_test(
        "Key index 79 64b",
        ["80000000000000000000"], ["00000000000000000000"],
        64, program,
    ))

    # -- IV bit-list index 79 64b --------------------------------------
    tests.append(make_test(
        "IV index 79 64b",
        ["00000000000000000000"], ["80000000000000000000"],
        64, program,
    ))

    # -- 3 pairs x 64b -------------------------------------------------
    tests.append(make_test(
        "3 pairs x 64b",
        ["00000000000000000000", "80000000000000000000", "00000000000000000000"],
        ["00000000000000000000", "00000000000000000000", "80000000000000000000"],
        64, program,
    ))

    # -- Zero key/IV 128b ----------------------------------------------
    tests.append(make_test(
        "Zero key/IV 128b",
        ["00000000000000000000"], ["00000000000000000000"],
        128, program,
    ))

    # -- ECRYPT vector (Key=0, IV=0) -----------------------------------
    # Also verify against ECRYPT published test vector
    key_zero = [0] * 80
    iv_zero = [0] * 80
    key_arrays, iv_arrays = pack_key_iv_pairs([key_zero], [iv_zero])
    result, ops = run_bs(program, 64, key_arrays, iv_arrays)
    z_array = result["z"]
    got_ks = unpack_keystreams(z_array, 1, 64)
    ecrypt_hex = bits_to_ecrypt_hex(got_ks[0])
    assert ecrypt_hex == "FBE0BF265859051B", \
        f"ECRYPT vector: got {ecrypt_hex}, expected FBE0BF265859051B"
    # Also verify reference
    ref_ks = trivium_reference(key_zero, iv_zero, 64)
    ref_hex = bits_to_ecrypt_hex(ref_ks)
    assert ref_hex == "FBE0BF265859051B", \
        f"Reference ECRYPT: got {ref_hex}, expected FBE0BF265859051B"
    assert got_ks[0] == ref_ks, "ECRYPT vector: bitstream != reference"

    # Mask z_array to 1 bit (bitlength=1)
    z_masked = {k: v & 1 for k, v in z_array.items()}

    write_bsdata(
        os.path.join(TESTS_DIR, "ecrypt_vector_key=0_iv=0.bsdata"),
        1,
        params={"L": 64},
        input_arrays={"key": key_arrays, "iv": iv_arrays},
        expected={"z": z_masked},
    )
    tests.append({
        "name": "ECRYPT vector (Key=0, IV=0)",
        "bitlength": 1,
        "data_file": "ecrypt_vector_key=0_iv=0.bsdata",
    })

    # -- Wide test (generated at runtime) ------------------------------
    tests.append({
        "name": "Large 1M pairs x 64b",
        "category": "generated",
        "bitlength": 1000000,
        "generate": {"W": 1000000, "length": 64, "seed": 42},
    })

    # -- Preserve existing tier entries and write tests.json ----------
    existing = load_tests_json()
    tier_data_files = {f"trivium_tier_{t}.bsdata" for t in TIER_NAMES}
    for entry in existing:
        if entry.get("data_file") in tier_data_files:
            tests.append(entry)

    save_tests_json(tests)
    print(f"trivium: generated {len(tests)} test entries")


if __name__ == "__main__":
    main()
