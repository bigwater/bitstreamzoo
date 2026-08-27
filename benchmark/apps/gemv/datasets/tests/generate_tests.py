#!/usr/bin/env python3
"""Generate precomputed test data for GEMV.

Usage:
    python generate_tests.py                # unit tests only (existing behavior)
    python generate_tests.py --tier small   # unit tests + small tier (3 configs)
    python generate_tests.py --tier medium  # unit tests + medium tier (3 configs)
    python generate_tests.py --tier large   # unit tests + large tier (3 configs)
    python generate_tests.py --tier all     # unit tests + all tiers
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
from benchmark.apps.gemv.src.run import (
    gemv_reference, encode_vector, encode_weights,
    decode_result, compute_B,
)
from benchmark.bsdata import write_bsdata
from benchmark.tier_generate import (
    file_sha256, make_provenance,
    tier_test_entry, print_describe, verify_files,
)
from benchmark.tier_config import get_tier_vectors

TESTS_DIR = os.path.dirname(__file__)
BS_PATH = os.path.join(TESTS_DIR, "../../src/gemv.bs")

DOMAIN = "gemv"

# 3 tier configurations varying shape and precision
TIER_CONFIGS = [
    {"name": "small_k2",  "L": 16, "N": 16,  "K": 2,
     "domain_key": "gemv_small_k2",  "seed_base": 3000},
    {"name": "medium_k4", "L": 64, "N": 64,  "K": 4,
     "domain_key": "gemv_medium_k4", "seed_base": 3100},
    {"name": "large_k8",  "L": 64, "N": 128, "K": 8,
     "domain_key": "gemv_large_k8",  "seed_base": 3200},
]

TIER_SEED_OFFSETS = {"small": 0, "medium": 1, "large": 2}

# All possible tier data files (for merge logic)
TIER_DATA_FILES = {
    f"gemv_{cfg['name']}_tier_{tier}.bsdata"
    for cfg in TIER_CONFIGS
    for tier in ["small", "medium", "large"]
}


def mask_stream(v, bitlength):
    if bitlength > 0:
        return v & ((1 << bitlength) - 1)
    return v



def run_bs(program, x_values, w_values, K):
    M = len(x_values)
    N = len(w_values[0])
    L = len(w_values)
    PK = 2 * K
    B = compute_B(N, K)

    x_arrays = encode_vector(x_values, N, K)
    w_arrays = encode_weights(w_values, L, N, K)

    interp = Interpreter()
    result = interp.run(
        program, inputs={},
        params={"L": L, "N": N, "K": K, "PK": PK, "B": B},
        input_arrays={"x": x_arrays, "w": w_arrays},
    )
    return result["y"], interp.op_count


def make_test(name, x_values, w_values, K, program):
    M = len(x_values)
    N = len(w_values[0])
    L = len(w_values)
    PK = 2 * K
    B = compute_B(N, K)

    result_streams, _ = run_bs(program, x_values, w_values, K)

    ref = gemv_reference(x_values, w_values)
    bs_results = decode_result(result_streams, L, B, M)
    assert bs_results == ref, f"{name}: bs={bs_results} ref={ref}"

    x_arrays = encode_vector(x_values, N, K)
    w_arrays = encode_weights(w_values, L, N, K)
    input_arrays = {
        "x": x_arrays,
        "w": w_arrays,
    }
    masked = {k: mask_stream(v, M) for k, v in result_streams.items()}
    expected = {"y": masked}

    # Sanitize name for filename
    safe = name.lower().replace(" ", "_").replace("=", "").replace("-", "_")
    data_filename = f"gemv_{safe}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_filename)

    write_bsdata(data_path, M,
                 params={"L": L, "N": N, "K": K, "PK": PK, "B": B},
                 input_arrays=input_arrays,
                 expected=expected)

    return {
        "name": name,
        "bitlength": M,
        "data_file": data_filename,
    }


def spot_check_gemv(x_arrays, w_values, y_streams, L, N, K, B,
                    bitlength, num_checks=500, seed=None):
    """Spot-check random positions in large bitsliced streams against reference.

    Extracts individual x_values from the bitstreams, runs gemv_reference,
    and compares against decoded output bits.
    """
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(bitlength), min(num_checks, bitlength)))
    for m in indices:
        # Extract x_values[m] from bitstreams
        x_vals = []
        for f in range(N):
            val = 0
            for b in range(K):
                if (x_arrays[f * K + b] >> m) & 1:
                    val |= 1 << b
            x_vals.append(val)

        # Compute reference for this single item
        ref = gemv_reference([x_vals], w_values)  # returns [[dp0, dp1, ..., dpL-1]]

        # Decode output from bitstreams for this position
        for l in range(L):
            ref_dp = ref[0][l]
            bs_dp = 0
            for b in range(B):
                if (y_streams.get(l * B + b, 0) >> m) & 1:
                    bs_dp |= 1 << b
            assert bs_dp == ref_dp, (
                f"spot-check position {m}: output[{l}] got={bs_dp} ref={ref_dp}")
    return len(indices)


def generate_tier_data(tier, cfg, bitlength, program):
    """Generate a tier .bsdata file for one GEMV config.

    Returns (data_file, size_bytes).
    """
    name = cfg["name"]
    L = cfg["L"]
    N = cfg["N"]
    K = cfg["K"]
    PK = 2 * K
    B = compute_B(N, K)
    seed = cfg["seed_base"] + TIER_SEED_OFFSETS[tier]
    M = bitlength
    mask = (1 << M) - 1
    rng = random.Random(seed)

    n_io = N * K + L * B
    n_w = L * N * K
    total_streams = n_io + n_w

    print(f"  Generating {name} {tier} tier: {M:,} vectors (seed={seed})...")
    print(f"    Architecture: L={L}, N={N}, K={K}, PK={PK}, B={B}")
    print(f"    Streams: {N*K} input + {L*B} output + {n_w} weight = {total_streams}")

    # Generate random K-bit weights: w[l][f] in [0, (1<<K)-1]
    max_val = (1 << K) - 1
    print(f"    Generating weights ({L}x{N}, K={K}-bit)...")
    w_values = [[rng.randint(0, max_val) for _ in range(N)] for _ in range(L)]

    # Generate random input bitstreams directly as large integers
    # x_arrays has N*K entries: x_arrays[f*K+b] = random M-bit integer
    print(f"    Generating x streams ({N * K} x {M:,} bits)...")
    t0 = time.time()
    x_arrays = {f * K + b: rng.getrandbits(M)
                for f in range(N) for b in range(K)}
    t_gen = time.time() - t0
    print(f"    Input generation: {t_gen:.1f}s")

    # Encode weights as broadcast constants
    w_arrays = encode_weights(w_values, L, N, K)

    # Run interpreter to compute expected outputs
    print(f"    Running interpreter ({M:,} vectors)...")
    t0 = time.time()
    interp = Interpreter()
    result = interp.run(
        program,
        inputs={},
        params={"L": L, "N": N, "K": K, "PK": PK, "B": B},
        input_arrays={"x": x_arrays, "w": w_arrays},
    )
    y_streams = result["y"]
    t_interp = time.time() - t0
    print(f"    Interpreter: {t_interp:.1f}s ({interp.op_count} ops)")

    # Spot-check 500 random positions against gemv_reference
    print(f"    Spot-checking 500 positions...")
    t0 = time.time()
    n_checked = spot_check_gemv(x_arrays, w_values, y_streams,
                                L, N, K, B, M,
                                num_checks=500, seed=seed + 5000)
    t_check = time.time() - t0
    print(f"    Spot-check: {n_checked} positions verified in {t_check:.1f}s")

    # Mask y outputs to M bits
    masked_y = {k: v & mask for k, v in y_streams.items()}

    # Build input_arrays dict
    input_arrays = {
        "x": x_arrays,
        "w": w_arrays,
    }

    # Write .bsdata
    data_file = f"gemv_{name}_tier_{tier}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_file)
    print(f"    Writing {data_file}...")
    t0 = time.time()
    write_bsdata(
        data_path,
        M,
        params={"L": L, "N": N, "K": K, "PK": PK, "B": B},
        input_arrays=input_arrays,
        expected={"y": masked_y},
    )
    t_write = time.time() - t0

    size_bytes = os.path.getsize(data_path)
    print(f"    Written: {data_file} ({size_bytes:,} bytes, "
          f"{size_bytes / 1e6:.1f} MB, {t_write:.1f}s)")

    return data_file, size_bytes


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


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate GEMV test data")
    parser.add_argument("--tier", choices=["small", "medium", "large", "all"],
                        help="Generate tier test data (3 configs per tier)")
    parser.add_argument("--describe", action="store_true",
                        help="Print provenance info for all generated files")
    parser.add_argument("--verify", action="store_true",
                        help="Verify SHA-256 of existing .bsdata files")
    args = parser.parse_args()

    # Handle --describe and --verify on existing tests.json
    if args.describe or args.verify:
        tests_json_path = os.path.join(TESTS_DIR, "tests.json")
        if not os.path.exists(tests_json_path):
            print(f"No tests.json found at {tests_json_path}")
            sys.exit(1)
        with open(tests_json_path) as f:
            tests = json.load(f)
        if args.describe:
            print(f"gemv: {len(tests)} test entries")
            print_describe(TESTS_DIR, tests)
        if args.verify:
            print(f"gemv: verifying SHA-256 checksums")
            ok = verify_files(TESTS_DIR, tests)
            sys.exit(0 if ok else 1)
        return

    with open(BS_PATH) as f:
        source = f.read()
    program = parse(source)

    if args.tier:
        # Tier generation mode: generate specified tier(s), merge into tests.json
        tier_names = ["small", "medium", "large"] if args.tier == "all" else [args.tier]

        # Load existing tests.json to preserve unit tests
        tests = load_tests_json()

        for tier in tier_names:
            for cfg in TIER_CONFIGS:
                bitlength = get_tier_vectors(cfg["domain_key"], tier)
                if bitlength is None:
                    print(f"  gemv {cfg['name']}: tier '{tier}' not applicable (skipped)")
                    continue

                data_file, size_bytes = generate_tier_data(
                    tier, cfg, bitlength, program)
                sha = file_sha256(os.path.join(TESTS_DIR, data_file))

                L, N, K = cfg["L"], cfg["N"], cfg["K"]
                PK = 2 * K
                B = compute_B(N, K)
                seed = cfg["seed_base"] + TIER_SEED_OFFSETS[tier]

                prov = make_provenance(
                    source="synthetic",
                    seed=seed,
                    description=(
                        f"Random {K}-bit input vectors and weight matrix. "
                        f"Architecture: L={L}, N={N}, K={K}, PK={PK}, B={B}. "
                        f"Input x: {N * K} random bitstreams "
                        f"({bitlength:,} bits each). "
                        f"Weights: {L}x{N} random {K}-bit. "
                        f"{bitlength:,} vectors."
                    ),
                    generated_by=f"generate_tests.py --tier {tier}",
                )
                prov["sha256"] = sha

                label = (f"GEMV {cfg['name']} tier {tier} "
                         f"(L={L} N={N} K={K}, {bitlength:,} vectors)")
                entry = tier_test_entry(
                    name=label,
                    bitlength=bitlength,
                    data_file=data_file,
                    size_bytes=size_bytes,
                    provenance=prov,
                )
                tests = merge_tier_entry(tests, entry)

        save_tests_json(tests)
    else:
        # Default: generate unit tests (original behavior)
        tests = []

        # ── Fixed: identity-like (L=1, N=1, K=2) ──────────────────
        tests.append(make_test("L=1 N=1 K=2 zero", [[0]], [[0]], 2, program))
        tests.append(make_test("L=1 N=1 K=2 one", [[1]], [[1]], 2, program))
        tests.append(make_test("L=1 N=1 K=2 max", [[3]], [[3]], 2, program))

        # ── L=2 N=2 K=2 multi-output ──────────────────────────────
        tests.append(make_test(
            "L=2 N=2 K=2",
            x_values=[[1, 2], [3, 0]],
            w_values=[[2, 1], [1, 3]],
            K=2, program=program,
        ))

        # ── L=4 N=4 K=2 ──────────────────────────────────────────
        rng = random.Random(42)
        L, N, K, M = 4, 4, 2, 8
        max_val = (1 << K) - 1
        x = [[rng.randint(0, max_val) for _ in range(N)] for _ in range(M)]
        w = [[rng.randint(0, max_val) for _ in range(N)] for _ in range(L)]
        tests.append(make_test("Random L=4 N=4 K=2 M=8", x, w, K, program))

        # ── L=4 N=8 K=4 ──────────────────────────────────────────
        rng = random.Random(43)
        L, N, K, M = 4, 8, 4, 16
        max_val = (1 << K) - 1
        x = [[rng.randint(0, max_val) for _ in range(N)] for _ in range(M)]
        w = [[rng.randint(0, max_val) for _ in range(N)] for _ in range(L)]
        tests.append(make_test("L=4 N=8 K=4 M=16", x, w, K, program))

        # ── DNN hidden layer: L=16 N=16 K=4 ──────────────────────
        rng = random.Random(44)
        L, N, K, M = 16, 16, 4, 32
        max_val = (1 << K) - 1
        x = [[rng.randint(0, max_val) for _ in range(N)] for _ in range(M)]
        w = [[rng.randint(0, max_val) for _ in range(N)] for _ in range(L)]
        tests.append(make_test("Hidden layer L=16 N=16 K=4", x, w, K, program))

        # ── DNN hidden layer: L=8 N=16 K=8 ───────────────────────
        rng = random.Random(45)
        L, N, K, M = 8, 16, 8, 16
        max_val = (1 << K) - 1
        x = [[rng.randint(0, max_val) for _ in range(N)] for _ in range(M)]
        w = [[rng.randint(0, max_val) for _ in range(N)] for _ in range(L)]
        tests.append(make_test("Hidden layer L=8 N=16 K=8", x, w, K, program))

        # ── Wide test (generated at runtime) ──────────────────────
        L, N, K = 4, 8, 4
        PK = 2 * K
        B = compute_B(N, K)
        tests.append({
            "name": "Large 1M items L=4 N=8 K=4",
            "category": "generated",
            "bitlength": 1000000,
            "generate": {"type": "wide", "L": L, "N": N, "K": K, "PK": PK, "B": B,
                         "W": 1000000, "seed": 42},
        })

        # Preserve any existing tier entries in tests.json
        existing = load_tests_json()
        for entry in existing:
            if entry.get("data_file") in TIER_DATA_FILES:
                tests.append(entry)

        save_tests_json(tests)
        print(f"gemv: generated {len(tests)} test entries")


if __name__ == "__main__":
    main()
