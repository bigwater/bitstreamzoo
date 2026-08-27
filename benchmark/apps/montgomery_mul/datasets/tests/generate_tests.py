#!/usr/bin/env python3
"""Generate test data for Montgomery modular multiplication.

Usage:
    python generate_tests.py              # unit tests only
    python generate_tests.py --tier small
    python generate_tests.py --tier all
    python generate_tests.py --describe
    python generate_tests.py --verify
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from simulator.pythonsim import parse
from benchmark.apps.montgomery_mul.src.run import (
    montgomery_mul_reference, compute_np,
    encode_operands, encode_broadcast, decode_result,
    run_montgomery_bs,
)
from benchmark.bsdata import write_bsdata
from benchmark.tier_config import get_tier_vectors
from benchmark.tier_generate import (
    file_sha256, make_provenance, tier_test_entry,
    getrandbits_large,
)

TESTS_DIR = os.path.dirname(__file__)
BS_PATH = os.path.join(TESTS_DIR, "../../src/montgomery_mul.bs")
DOMAIN = "montgomery_mul"

# Kyber modulus
KYBER_Q = 3329
KYBER_K = 12

# Stable fixture filenames. The committed unit .bsdata files use these names;
# deriving a filename from the display name would emit different (and
# shell-hostile, e.g. "r2^12") names and orphan the committed fixtures.
FIXTURE_FILENAMES = {
    "q=3329 zero (R=2^12)": "montgomery_kyber_zero.bsdata",
    "q=3329 a=1 b=1 (R=2^12)": "montgomery_kyber_a1_b1.bsdata",
    "q=3329 Montgomery form of 1 squared (R=2^12)":
        "montgomery_kyber_mont_form_of_1_squared.bsdata",
    "q=3329 max inputs (R=2^12)": "montgomery_kyber_max_inputs.bsdata",
    "q=3329 random M=16 (R=2^12)": "montgomery_kyber_random_m16.bsdata",
    "K=16 n=65521 random M=32": "montgomery_k16_n65521_random_m32.bsdata",
}


def make_test(name, a_vals, b_vals, n_val, K, program):
    """Create a precomputed test entry with .bsdata file."""
    M = len(a_vals)
    np_val = compute_np(n_val, K)
    PK = 2 * K

    # Reference
    ref = [montgomery_mul_reference(a_vals[m], b_vals[m], n_val, np_val, K)
           for m in range(M)]

    # Bitstream execution
    results, op_count = run_montgomery_bs(program, a_vals, b_vals, n_val, K)
    assert results == ref, f"{name}: bs={results[:5]} ref={ref[:5]}"

    # Encode for .bsdata
    mask = (1 << M) - 1
    a_arr = encode_operands(a_vals, K)
    b_arr = encode_operands(b_vals, K)
    n_arr = encode_broadcast(n_val, K)
    np_arr = encode_broadcast(np_val, K)

    # Encode expected output
    r_streams = encode_operands(ref, K)
    expected = {"r": {k: v & mask for k, v in r_streams.items()}}

    if name in FIXTURE_FILENAMES:
        data_filename = FIXTURE_FILENAMES[name]
    else:
        safe = name.lower().replace(" ", "_").replace("=", "").replace(",", "")
        safe = safe.replace("-", "_").replace("(", "").replace(")", "")
        safe = safe.replace("^", "").replace("*", "").replace("'", "")
        data_filename = f"montgomery_{safe}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_filename)

    write_bsdata(data_path, M,
                 params={"K": K, "PK": PK},
                 input_arrays={"a": a_arr, "b": b_arr,
                               "n": n_arr, "np": np_arr},
                 expected=expected)

    print(f"  {name}: M={M}, ops={op_count}, file={data_filename}")
    return {"name": name, "bitlength": M, "data_file": data_filename}


def generate_unit_tests(program):
    """Generate all unit test entries."""
    tests = []
    K = KYBER_K
    n = KYBER_Q
    np_val = compute_np(n, K)
    R = 1 << K

    # Verify n' is correct
    assert (n * np_val + 1) % R == 0, f"Bad n': {np_val}"

    # R mod n = Montgomery form of 1
    R_mod_n = R % n
    print(f"  Kyber: q={n}, K={K}, R={R}, R mod n={R_mod_n}, n'={np_val}")

    # ── Test: a=0, b=0 -> 0 ────────────────────────────────
    tests.append(make_test("q=3329 zero (R=2^12)", [0], [0], n, K, program))

    # ── Test: a=1, b=1 -> R^{-1} mod n ─────────────────────
    R_inv = montgomery_mul_reference(1, 1, n, np_val, K)
    print(f"  R^{{-1}} mod {n} = {R_inv}")
    tests.append(make_test("q=3329 a=1 b=1 (R=2^12)", [1], [1], n, K, program))

    # ── Test: Montgomery form of 1 squared = Montgomery form of 1
    # MontMul(R mod n, R mod n) = (R mod n)^2 * R^{-1} mod n = R mod n
    mont_one = R_mod_n
    result = montgomery_mul_reference(mont_one, mont_one, n, np_val, K)
    print(f"  MontMul({mont_one}, {mont_one}) = {result}")
    tests.append(make_test("q=3329 Montgomery form of 1 squared (R=2^12)",
                           [mont_one], [mont_one], n, K, program))

    # ── Test: max inputs a=n-1, b=n-1 ──────────────────────
    tests.append(make_test("q=3329 max inputs (R=2^12)",
                           [n - 1], [n - 1], n, K, program))

    # ── Test: random batch M=16 at K=12 ────────────────────
    rng = random.Random(4001)
    M = 16
    a_vals = [rng.randint(0, n - 1) for _ in range(M)]
    b_vals = [rng.randint(0, n - 1) for _ in range(M)]
    tests.append(make_test("q=3329 random M=16 (R=2^12)", a_vals, b_vals, n, K, program))

    # ── Test: random batch M=32 at K=16, n=65521 ──────────
    K16 = 16
    n16 = 65521  # largest 16-bit prime
    rng16 = random.Random(4002)
    M16 = 32
    a16 = [rng16.randint(0, n16 - 1) for _ in range(M16)]
    b16 = [rng16.randint(0, n16 - 1) for _ in range(M16)]
    tests.append(make_test("K=16 n=65521 random M=32", a16, b16, n16, K16, program))

    # ── Generated: 1M items at K=12 ───────────────────────
    tests.append({
        "name": "q=3329 1M items K=12",
        "category": "generated",
        "bitlength": 1000000,
        "generate": {
            "type": "montgomery_mul",
            "K": KYBER_K, "n": KYBER_Q,
            "np": compute_np(KYBER_Q, KYBER_K),
            "seed": 4444,
        },
    })

    return tests


def spot_check_montgomery(a_arr, b_arr, r_streams, n, np_val, K,
                          bitlength, num_checks=500, seed=None):
    """Spot-check random positions in the bitsliced tier streams against the
    scalar reference. Extracts (a, b) at each position, runs
    montgomery_mul_reference, and compares against the decoded bs output.
    Mirrors gemv/edit_distance tier verification."""
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(bitlength), min(num_checks, bitlength)))
    for m in indices:
        a_val = 0
        b_val = 0
        for b in range(K):
            if (a_arr[b] >> m) & 1:
                a_val |= 1 << b
            if (b_arr[b] >> m) & 1:
                b_val |= 1 << b
        ref = montgomery_mul_reference(a_val, b_val, n, np_val, K)
        got = 0
        for b in range(K):
            if (r_streams.get(b, 0) >> m) & 1:
                got |= 1 << b
        assert got == ref, (
            f"spot-check position {m}: a={a_val} b={b_val} "
            f"got={got} ref={ref}")
    return len(indices)


TIER_SEEDS = {"small": 5000, "medium": 5001, "large": 5002}


def generate_tiers(program, tier_arg, unit_tests, tests_path):
    """Generate tier data for montgomery_mul."""
    import time

    tier_names = list(TIER_SEEDS.keys()) if tier_arg == "all" else [tier_arg]

    # Load existing tests.json to preserve unit tests + other tiers
    if os.path.exists(tests_path):
        with open(tests_path) as f:
            tests = json.load(f)
    else:
        tests = list(unit_tests)

    K = KYBER_K
    n = KYBER_Q
    np_val = compute_np(n, K)
    PK = 2 * K

    for tier in tier_names:
        bitlength = get_tier_vectors(DOMAIN, tier)
        if bitlength is None:
            print(f"  {tier}: not applicable (None in tier_config)")
            continue

        seed = TIER_SEEDS[tier]
        data_file = f"montgomery_mul_tier_{tier}.bsdata"
        data_path = os.path.join(TESTS_DIR, data_file)

        print(f"  Generating {tier} tier: bitlength={bitlength:,}, seed={seed}...")
        t0 = time.time()

        # Generate random operands as bitstreams directly
        rng = random.Random(seed)
        a_arr = {k: getrandbits_large(rng, bitlength) for k in range(K)}
        b_arr = {k: getrandbits_large(rng, bitlength) for k in range(K)}
        n_arr = encode_broadcast(n, K)
        np_arr = encode_broadcast(np_val, K)
        t_gen = time.time() - t0
        print(f"    Input gen: {t_gen:.1f}s")

        # Run interpreter
        from simulator.pythonsim.interpreter import Interpreter
        t0 = time.time()
        interp = Interpreter()
        result = interp.run(
            program, inputs={}, params={"K": K, "PK": PK},
            input_arrays={"a": a_arr, "b": b_arr, "n": n_arr, "np": np_arr},
        )
        t_interp = time.time() - t0
        print(f"    Interpreter: {t_interp:.1f}s ({interp.op_count} ops)")

        # Spot-check 500 random positions against the scalar reference
        t0 = time.time()
        n_checked = spot_check_montgomery(
            a_arr, b_arr, result["r"], n, np_val, K, bitlength,
            num_checks=500, seed=seed + 5000)
        print(f"    Spot-check: {n_checked} positions verified "
              f"in {time.time() - t0:.1f}s")

        # Mask and write
        mask = (1 << bitlength) - 1
        expected = {"r": {k: v & mask for k, v in result["r"].items()}}

        t0 = time.time()
        write_bsdata(
            data_path, bitlength,
            params={"K": K, "PK": PK},
            input_arrays={"a": a_arr, "b": b_arr, "n": n_arr, "np": np_arr},
            expected=expected,
        )
        t_write = time.time() - t0
        size_bytes = os.path.getsize(data_path)
        print(f"    Written: {data_file} ({size_bytes:,} bytes, {t_write:.1f}s)")

        sha = file_sha256(data_path)
        prov = make_provenance(
            source="synthetic", seed=seed,
            description=(f"Random Montgomery reduction over q={n} with R=2^{K}. "
                         f"Operands are unconstrained {K}-bit streams, not "
                         f"canonical Kyber-range inputs. {bitlength:,} "
                         f"independent multiply-mod instances."),
            generated_by=f"generate_tests.py --tier {tier}",
        )
        prov["sha256"] = sha

        entry = tier_test_entry(
            name=f"Tier {tier} ({bitlength:,} items)",
            bitlength=bitlength,
            data_file=data_file,
            size_bytes=size_bytes,
            provenance=prov,
        )

        # Merge into tests list
        replaced = False
        for i, t in enumerate(tests):
            if t.get("data_file") == data_file:
                tests[i] = entry
                replaced = True
                break
        if not replaced:
            tests.append(entry)

        # Save incrementally
        with open(tests_path, "w") as f:
            json.dump(tests, f, indent=2)
        print(f"    Saved tests.json ({len(tests)} entries)")

    print(f"\n{DOMAIN}: tier generation complete")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["small", "medium", "large", "all"])
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    tests_path = os.path.join(TESTS_DIR, "tests.json")

    if args.describe or args.verify:
        if not os.path.exists(tests_path):
            print("No tests.json found"); sys.exit(1)
        with open(tests_path) as f:
            tests = json.load(f)
        if args.describe:
            print(f"{DOMAIN}: {len(tests)} test entries")
            for t in tests:
                print(f"  - {t['name']} (bitlength={t['bitlength']})")
        if args.verify:
            from benchmark.tier_generate import verify_files
            ok = verify_files(TESTS_DIR, [t for t in tests if t.get("provenance")])
            sys.exit(0 if ok else 1)
        return

    with open(BS_PATH) as f:
        program = parse(f.read())

    print(f"Generating {DOMAIN} tests...")
    tests = generate_unit_tests(program)

    if args.tier:
        generate_tiers(program, args.tier, tests, tests_path)
    else:
        # Preserve existing tier entries (only if tier is not None in tier_config)
        if os.path.exists(tests_path):
            with open(tests_path) as f:
                existing = json.load(f)
            for tier in ["small", "medium", "large"]:
                bl = get_tier_vectors(DOMAIN, tier)
                if bl is None:
                    continue
                df = f"montgomery_mul_tier_{tier}.bsdata"
                for entry in existing:
                    if entry.get("data_file") == df:
                        tests.append(entry)

        with open(tests_path, "w") as f:
            json.dump(tests, f, indent=2)
        print(f"\n{DOMAIN}: {len(tests)} test entries written to tests.json")


if __name__ == "__main__":
    main()
