#!/usr/bin/env python3
"""Generate precomputed bitstream-level test data for polar code encoding.

Usage:
    python generate_tests.py                # unit tests; preserves existing tier entries
    python generate_tests.py --tier small   # small tier only, merged into tests.json
    python generate_tests.py --tier medium  # medium tier only, merged into tests.json
    python generate_tests.py --tier large   # large tier only, merged into tests.json
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
from benchmark.apps.polar.src.run import polar_encode_reference, unpack_codewords
from benchmark.bsdata import write_bsdata
from benchmark.tier_generate import (
    parse_generate_args, file_sha256, make_provenance,
    tier_test_entry, print_describe, verify_files,
)
from benchmark.tier_config import get_tier_vectors

TESTS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(TESTS_DIR, "../../src")

# Program variants: name -> (N, bs_file)
PROGRAMS = {
    "polar_small":  (64,   "polar_small.bs"),
    "polar_medium": (256,  "polar_medium.bs"),
    "polar_large":  (1024, "polar_large.bs"),
}

TIER_SEEDS = {"small": 5000, "medium": 5001, "large": 5002}

# All possible tier data files (for merge logic)
TIER_DATA_FILES = {
    f"{prog}_tier_{tier}.bsdata"
    for prog in PROGRAMS
    for tier in ["small", "medium", "large"]
}


def pack_messages(messages, N):
    """Bit-slice K messages of N bits into input_arrays dict.

    Returns: input_arrays dict {i: stream_bits}
    """
    K = len(messages)
    u_arrays = {}
    for i in range(N):
        bits = 0
        for j in range(K):
            if messages[j][i]:
                bits |= 1 << j
        u_arrays[i] = bits
    return u_arrays


def run_bs(program, u_arrays):
    """Run interpreter, return (result dict, op_count)."""
    interp = Interpreter()
    result = interp.run(program, inputs={}, params={},
                        input_arrays={"u": u_arrays})
    return result, interp.op_count


def make_test(name, messages, N, program, prog_name):
    """Create a unit test entry: run bs, verify against reference, write .bsdata."""
    K = len(messages)
    u_arrays = pack_messages(messages, N)
    result, ops = run_bs(program, u_arrays)
    x_array = result["x"]

    # Verify against reference
    codewords = unpack_codewords(x_array, K, N)
    for j in range(K):
        ref = polar_encode_reference(messages[j], N)
        assert codewords[j] == ref, (
            f"{name}: msg {j} mismatch: got={codewords[j][:8]}... "
            f"ref={ref[:8]}...")

    # Mask outputs to K bits
    mask = (1 << K) - 1
    masked_x = {k: v & mask for k, v in x_array.items()}

    # Write .bsdata
    safe = name.lower().replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
    data_filename = f"{prog_name}_{safe}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_filename)

    write_bsdata(data_path, K,
                 params={},
                 input_arrays={"u": u_arrays},
                 expected={"x": masked_x})

    return {
        "name": name,
        "bitlength": K,
        "data_file": data_filename,
    }


def generate_tier_data(program, prog_name, N, tier, bitlength):
    """Generate a tier .bsdata file with random N-bit messages.

    Returns (data_file, size_bytes).
    """
    seed = TIER_SEEDS[tier]
    rng = random.Random(seed)

    print(f"  [{prog_name} {tier}] Generating {bitlength:,} vectors "
          f"(N={N}, seed={seed})...")

    # Generate N input streams, each bitlength bits wide
    t0 = time.time()
    u_arrays = {i: rng.getrandbits(bitlength) for i in range(N)}
    t_gen = time.time() - t0
    print(f"    Input generation: {t_gen:.1f}s")

    # Run interpreter
    print(f"    Running polar encoder (N={N}, {bitlength:,} vectors)...")
    t0 = time.time()
    result, ops = run_bs(program, u_arrays)
    x_array = result["x"]
    t_interp = time.time() - t0
    print(f"    Interpreter: {ops:,} ops, {t_interp:.1f}s")

    # Mask outputs to bitlength bits
    mask = (1 << bitlength) - 1
    masked_x = {k: v & mask for k, v in x_array.items()}

    # Spot-check against scalar reference
    n_verify = min(20, bitlength)
    verify_rng = random.Random(seed + 100)
    verify_indices = sorted(verify_rng.sample(range(bitlength), n_verify))
    print(f"    Spot-checking {n_verify} positions...")
    t0 = time.time()
    for j in verify_indices:
        u_bits = [(u_arrays[i] >> j) & 1 for i in range(N)]
        x_bits = [(masked_x.get(i, 0) >> j) & 1 for i in range(N)]
        ref = polar_encode_reference(u_bits, N)
        assert x_bits == ref, (
            f"Tier {tier}: spot-check failed at position {j}")
    t_check = time.time() - t0
    print(f"    All {n_verify} spot-checks passed ({t_check:.1f}s)")

    # Write .bsdata
    data_file = f"{prog_name}_tier_{tier}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_file)
    print(f"    Writing {data_file}...")
    t0 = time.time()
    write_bsdata(
        data_path,
        bitlength,
        params={},
        input_arrays={"u": u_arrays},
        expected={"x": masked_x},
    )
    t_write = time.time() - t0
    size_bytes = os.path.getsize(data_path)
    print(f"    Wrote {data_file} ({size_bytes:,} bytes, "
          f"{size_bytes / 1e6:.1f} MB, {t_write:.1f}s)")

    return data_file, size_bytes


def load_tests_json():
    """Load existing tests.json if it exists."""
    path = os.path.join(TESTS_DIR, "tests.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_tests_json(tests):
    """Write tests.json (dict keyed by program name)."""
    path = os.path.join(TESTS_DIR, "tests.json")
    with open(path, "w") as f:
        json.dump(tests, f, indent=2)
    total = sum(len(v) for v in tests.values())
    print(f"  Wrote {path} ({total} entries across {len(tests)} programs)")


def main():
    args = parse_generate_args("polar")

    # Handle --describe and --verify
    if args.describe or args.verify:
        tests_json_path = os.path.join(TESTS_DIR, "tests.json")
        if not os.path.exists(tests_json_path):
            print(f"No tests.json found at {tests_json_path}")
            sys.exit(1)
        with open(tests_json_path) as f:
            tests = json.load(f)
        all_entries = []
        for prog_tests in tests.values():
            all_entries.extend(prog_tests)
        if args.describe:
            print(f"polar: {len(all_entries)} test entries")
            print_describe(TESTS_DIR, all_entries)
        if args.verify:
            print(f"polar: verifying SHA-256 checksums")
            ok = verify_files(TESTS_DIR, all_entries)
            sys.exit(0 if ok else 1)
        return

    # Load and parse all .bs programs
    programs = {}
    for prog_name, (N, bs_file) in PROGRAMS.items():
        bs_path = os.path.join(SRC_DIR, bs_file)
        with open(bs_path) as f:
            programs[prog_name] = (parse(f.read()), N)

    if args.tier:
        # Tier generation mode: merge into existing tests.json
        tests = load_tests_json()

        tier = args.tier
        for prog_name, (program, N) in programs.items():
            bitlength = get_tier_vectors(prog_name, tier)
            if bitlength is None:
                print(f"  {prog_name}: tier '{tier}' not applicable (skipped)")
                continue

            data_file, size_bytes = generate_tier_data(
                program, prog_name, N, tier, bitlength)
            sha = file_sha256(os.path.join(TESTS_DIR, data_file))

            n = int(math.log2(N))
            prov = make_provenance(
                source="synthetic",
                seed=TIER_SEEDS[tier],
                description=(
                    f"Random {N}-bit messages: {N} input + {N} output streams. "
                    f"Butterfly XOR network, {n} stages, "
                    f"{N * n // 2} XOR ops. "
                    f"{bitlength:,} vectors (messages). "
                    f"{2 * N} total binary streams."
                ),
                generated_by=f"generate_tests.py --tier {tier}",
                sha256=sha,
            )

            label = (f"Tier {tier} ({bitlength:,} vectors, N={N})")
            entry = tier_test_entry(
                name=label,
                bitlength=bitlength,
                data_file=data_file,
                size_bytes=size_bytes,
                provenance=prov,
            )

            # Merge into program's test list
            if prog_name not in tests:
                tests[prog_name] = []
            # Replace existing tier entry or append
            replaced = False
            for i, e in enumerate(tests[prog_name]):
                if e.get("data_file") == data_file:
                    tests[prog_name][i] = entry
                    replaced = True
                    break
            if not replaced:
                tests[prog_name].append(entry)

        save_tests_json(tests)
    else:
        # Default: generate unit tests
        tests = {}

        for prog_name, (program, N) in programs.items():
            prog_tests = []
            rng = random.Random(42)

            # 1. All-zero (1 message)
            msgs = [[0] * N]
            prog_tests.append(make_test("All-zero", msgs, N, program,
                                        prog_name))

            # 2. Single-bit one-hot
            if N <= 256:
                msgs = []
                for k in range(N):
                    m = [0] * N
                    m[k] = 1
                    msgs.append(m)
                prog_tests.append(make_test(
                    f"One-hot {N} msgs", msgs, N, program, prog_name))
            else:
                # For N=1024, use a subset to keep test small
                msgs = []
                for k in range(0, N, 16):
                    m = [0] * N
                    m[k] = 1
                    msgs.append(m)
                prog_tests.append(make_test(
                    f"One-hot {len(msgs)} msgs stride-16", msgs, N,
                    program, prog_name))

            # 3. Random messages
            n_random = 128 if N <= 256 else 64
            msgs = [[rng.randint(0, 1) for _ in range(N)]
                    for _ in range(n_random)]
            prog_tests.append(make_test(
                f"Random {n_random} msgs seed42", msgs, N, program,
                prog_name))

            # 4. Larger random (for medium)
            if N == 256:
                rng2 = random.Random(123)
                msgs = [[rng2.randint(0, 1) for _ in range(N)]
                        for _ in range(512)]
                prog_tests.append(make_test(
                    f"Random 512 msgs seed123", msgs, N, program,
                    prog_name))

            # 5. Wide test (generated at runtime)
            W = 100000 if N <= 256 else 10000
            prog_tests.append({
                "name": f"Wide {W:,} msgs",
                "category": "generated",
                "bitlength": W,
                "generate": {"type": "random", "N": N, "W": W, "seed": 42},
            })

            # Preserve existing tier entries
            existing = load_tests_json()
            if prog_name in existing:
                for entry in existing[prog_name]:
                    if entry.get("data_file") in TIER_DATA_FILES:
                        prog_tests.append(entry)

            tests[prog_name] = prog_tests

        save_tests_json(tests)
        total = sum(len(v) for v in tests.values())
        print(f"polar: generated {total} test entries")


if __name__ == "__main__":
    main()
