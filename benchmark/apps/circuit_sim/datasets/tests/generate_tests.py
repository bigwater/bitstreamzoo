#!/usr/bin/env python3
"""Generate precomputed bitstream-level test data for circuit simulation.

Each .bench netlist gets its own test entries in the dict-based tests.json.
c17 and adder4 have exhaustive tests; all others have random tests.
Wide (1M) tests remain as "generated" (require runtime .bench reference sim).

Usage:
    python generate_tests.py                 # unit tests only (default)
    python generate_tests.py --tier small    # add small tier for all 12 circuits
    python generate_tests.py --tier medium   # add medium tier for all 12 circuits
    python generate_tests.py --tier large    # add large tier for all 12 circuits
    python generate_tests.py --describe      # print provenance info
    python generate_tests.py --verify        # verify SHA-256 of existing files
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter
from benchmark.apps.circuit_sim.datasets.bench2bs import convert as bench2bs, parse_bench, simulate_bench
from benchmark.bsdata import write_bsdata
from benchmark.tier_config import get_tier_vectors, estimate_size_bytes
from benchmark.tier_generate import (
    parse_generate_args, file_sha256, make_provenance,
    tier_test_entry, print_describe, verify_files,
    getrandbits_large,
)

TESTS_DIR = os.path.dirname(__file__)
NETLISTS_DIR = os.path.join(TESTS_DIR, "../../datasets/raw")

# All 12 circuits in sorted order (matches sorted listdir of .bench files).
# Index in this list determines the seed offset.
CIRCUIT_NAMES = [
    "adder4", "c1355", "c17", "c1908", "c2670", "c3540",
    "c432", "c499", "c5315", "c6288", "c7552", "c880",
]

# Seed base per tier: seed = base + circuit_index
TIER_SEED_BASE = {"small": 1000, "medium": 2000, "large": 3000}

# Large tier: all 12 circuits (total ~55 GB)
LARGE_TIER_CIRCUITS = set(CIRCUIT_NAMES)


def generate_c17(bench_text, program):
    """Generate exhaustive test for c17 (5 inputs, 32 vectors)."""
    n_tests = 32
    input_names = ["G1", "G2", "G3", "G4", "G5"]
    input_streams = {}
    for bit, name in enumerate(input_names):
        stream = 0
        for t in range(n_tests):
            if t & (1 << bit):
                stream |= 1 << t
        input_streams[name] = stream

    interp = Interpreter()
    result = interp.run(program, inputs=input_streams)

    # Verify against reference
    for t in range(n_tests):
        vec = {name: (input_streams[name] >> t) & 1 for name in input_names}
        ref = simulate_bench(bench_text, vec)
        _, outputs_list, _ = parse_bench(bench_text)
        for out_name in outputs_list:
            got = (result[out_name] >> t) & 1
            assert got == ref[out_name], f"c17 t={t} {out_name}: {got} != {ref[out_name]}"

    # Build expected dict
    _, outputs_list, _ = parse_bench(bench_text)
    expected = {name: result[name] & ((1 << n_tests) - 1) for name in outputs_list}

    return {
        "name": "Exhaustive 32",
        "bitlength": 32,
        "inputs": input_streams,
        "expected": expected,
    }


def generate_adder4(bench_text, program):
    """Generate exhaustive test for adder4 (9 inputs, 512 vectors)."""
    test_inputs = []
    for a in range(16):
        for b in range(16):
            for cin in range(2):
                test_inputs.append((a, b, cin))
    n_tests = len(test_inputs)

    a_wires = {f"a{i}": 0 for i in range(4)}
    b_wires = {f"b{i}": 0 for i in range(4)}
    cin_stream = 0
    for t, (a, b, cin) in enumerate(test_inputs):
        for i in range(4):
            if a & (1 << i):
                a_wires[f"a{i}"] |= 1 << t
            if b & (1 << i):
                b_wires[f"b{i}"] |= 1 << t
        if cin:
            cin_stream |= 1 << t

    inputs = {**a_wires, **b_wires, "cin": cin_stream}
    interp = Interpreter()
    result = interp.run(program, inputs=inputs)

    # Verify against scalar addition
    mask = (1 << n_tests) - 1
    for t, (a, b, cin) in enumerate(test_inputs):
        expected_full = a + b + cin
        expected_s = expected_full & 0xF
        expected_cout = (expected_full >> 4) & 1
        got_s = 0
        for i in range(4):
            if result[f"s{i}"] & (1 << t):
                got_s |= 1 << i
        got_cout = (result["cout"] >> t) & 1
        assert got_s == expected_s and got_cout == expected_cout, \
            f"adder4 t={t}: {a}+{b}+{cin} got s={got_s} cout={got_cout}"

    _, outputs_list, _ = parse_bench(bench_text)
    expected = {name: result[name] & mask for name in outputs_list}

    return {
        "name": "Exhaustive 512",
        "bitlength": 512,
        "inputs": inputs,
        "expected": expected,
    }


def generate_generic_random(bench_text, program, seed=42):
    """Generate random test for a generic netlist."""
    inputs_list, outputs_list, _ = parse_bench(bench_text)
    n_inputs = len(inputs_list)
    if n_inputs <= 10:
        bitlength = min(1 << n_inputs, 1024)
    else:
        bitlength = 200

    rng = random.Random(seed)
    input_streams = {name: 0 for name in inputs_list}
    vector_inputs = []

    for t in range(bitlength):
        if n_inputs <= 10 and bitlength == (1 << n_inputs):
            bits = t
        else:
            bits = rng.getrandbits(n_inputs)
        vec = {}
        for bit_idx, name in enumerate(inputs_list):
            val = (bits >> bit_idx) & 1
            vec[name] = val
            if val:
                input_streams[name] |= 1 << t
        vector_inputs.append(vec)

    interp = Interpreter()
    result = interp.run(program, inputs=input_streams)

    # Verify against reference: every generated vector, every output
    mask = (1 << bitlength) - 1
    for t in range(bitlength):  # verify all vectors per-output
        ref = simulate_bench(bench_text, vector_inputs[t])
        for out_name in outputs_list:
            got = (result[out_name] >> t) & 1
            assert got == ref[out_name], \
                f"t={t} {out_name}: got={got} expected={ref[out_name]}"

    expected = {name: result[name] & mask for name in outputs_list}

    return {
        "name": f"Random {bitlength}",
        "bitlength": bitlength,
        "inputs": input_streams,
        "expected": expected,
    }


def spot_check_circuit(bench_text, input_streams, result, outputs_list,
                       bitlength, num_checks=500, seed=None):
    """Spot-check random positions in large bitsliced streams against scalar reference.

    Returns the number of positions checked.
    """
    rng = random.Random(seed)
    inputs_list, _, _ = parse_bench(bench_text)
    indices = sorted(rng.sample(range(bitlength), min(num_checks, bitlength)))
    for t in indices:
        vec = {name: (input_streams[name] >> t) & 1 for name in inputs_list}
        ref = simulate_bench(bench_text, vec)
        for out_name in outputs_list:
            got = (result[out_name] >> t) & 1
            assert got == ref[out_name], \
                f"spot-check t={t} {out_name}: got={got} expected={ref[out_name]}"
    return len(indices)


def generate_tier_for_circuit(circuit_name, circuit_index, tier, bench_text, program):
    """Generate a single tier test for one circuit.

    Args:
        circuit_name: e.g. "c17", "adder4", "c7552"
        circuit_index: index in CIRCUIT_NAMES (for seed computation)
        tier: "small", "medium", or "large"
        bench_text: raw .bench file contents
        program: parsed .bs program

    Returns:
        tier test entry dict for tests.json
    """
    bitlength = get_tier_vectors(circuit_name, tier)
    if bitlength is None:
        return None

    seed = TIER_SEED_BASE[tier] + circuit_index
    prog_name = "netlist_" + circuit_name
    data_file = f"{prog_name}_{tier}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_file)

    inputs_list, outputs_list, _ = parse_bench(bench_text)
    n_inputs = len(inputs_list)
    n_outputs = len(outputs_list)
    n_streams = n_inputs + n_outputs

    print(f"  Generating {prog_name} {tier} "
          f"(bitlength={bitlength:,}, seed={seed}, "
          f"{n_inputs} inputs + {n_outputs} outputs = {n_streams} streams)...")

    # Generate random input streams directly as big integers
    t0 = time.time()
    rng = random.Random(seed)
    input_streams = {name: getrandbits_large(rng, bitlength) for name in inputs_list}
    t_gen = time.time() - t0
    print(f"    Input generation: {t_gen:.1f}s")

    # Run interpreter
    t0 = time.time()
    interp = Interpreter()
    result = interp.run(program, inputs=input_streams)
    t_interp = time.time() - t0
    print(f"    Interpreter: {t_interp:.1f}s ({interp.op_count} ops)")

    # Mask outputs to bitlength bits
    mask = (1 << bitlength) - 1
    expected = {name: result[name] & mask for name in outputs_list}

    # Spot-check against scalar simulate_bench
    t0 = time.time()
    n_checked = spot_check_circuit(
        bench_text, input_streams, result, outputs_list,
        bitlength, num_checks=500, seed=seed + 5000)
    t_check = time.time() - t0
    print(f"    Spot-check: {n_checked} positions verified in {t_check:.1f}s")

    # Write .bsdata
    t0 = time.time()
    write_bsdata(
        data_path,
        bitlength,
        inputs=input_streams,
        expected=expected,
    )
    t_write = time.time() - t0

    file_size = os.path.getsize(data_path)
    print(f"    Written: {data_file} ({file_size:,} bytes, "
          f"{file_size / 1e6:.1f} MB, {t_write:.1f}s)")

    sha = file_sha256(data_path)

    prov = make_provenance(
        source="synthetic",
        seed=seed,
        description=(
            f"Random logic test vectors for {circuit_name}: "
            f"{n_inputs} input streams via getrandbits({bitlength}); "
            f"expected via Python interpreter ({interp.op_count} ops); "
            f"spot-checked 500 positions (seed={seed + 5000})"
        ),
        generated_by=f"generate_tests.py --tier {tier}",
    )
    prov["sha256"] = sha

    return tier_test_entry(
        name=f"Tier {tier} ({bitlength:,} vectors)",
        bitlength=bitlength,
        data_file=data_file,
        size_bytes=file_size,
        provenance=prov,
    )


def load_tests_json():
    """Load existing tests.json if it exists (dict keyed by program name)."""
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
    print(f"  Wrote {path} ({total} entries across {len(tests)} netlists)")


def merge_tier_entry(entries, new_entry):
    """Merge a tier entry into a program's entry list, replacing any existing
    entry with the same data_file."""
    data_file = new_entry["data_file"]
    for i, entry in enumerate(entries):
        if entry.get("data_file") == data_file:
            entries[i] = new_entry
            return entries
    entries.append(new_entry)
    return entries


def circuits_for_tier(tier):
    """Return the list of circuit names applicable for a given tier."""
    if tier == "large":
        return [c for c in CIRCUIT_NAMES if c in LARGE_TIER_CIRCUITS]
    else:
        return list(CIRCUIT_NAMES)


def main():
    args = parse_generate_args("circuit_sim")

    # Handle --describe and --verify on existing tests.json
    if args.describe or args.verify:
        tests = load_tests_json()
        if not tests:
            print("No tests.json found")
            sys.exit(1)
        # Flatten all entries for describe/verify
        all_entries = []
        for prog_name, entries in sorted(tests.items()):
            print(f"\n{prog_name}:")
            if args.describe:
                print_describe(TESTS_DIR, entries)
            if args.verify:
                ok = verify_files(TESTS_DIR, entries)
                if not ok:
                    sys.exit(1)
        return

    bench_files = sorted(f for f in os.listdir(NETLISTS_DIR) if f.endswith(".bench"))

    if args.tier:
        # Tier generation mode: generate specified tier(s), merge into tests.json
        tests = load_tests_json()

        tier = args.tier
        target_circuits = circuits_for_tier(tier)
        print(f"circuit_sim: generating {tier} tier for {len(target_circuits)} circuit(s)")

        for bench_file in bench_files:
            circuit_name = os.path.splitext(bench_file)[0]
            if circuit_name not in target_circuits:
                continue

            circuit_index = CIRCUIT_NAMES.index(circuit_name)
            bench_path = os.path.join(NETLISTS_DIR, bench_file)
            with open(bench_path) as f:
                bench_text = f.read()

            bs_text = bench2bs(bench_text)
            program = parse(bs_text)
            prog_name = "netlist_" + circuit_name

            entry = generate_tier_for_circuit(
                circuit_name, circuit_index, tier, bench_text, program)
            if entry is not None:
                if prog_name not in tests:
                    tests[prog_name] = []
                tests[prog_name] = merge_tier_entry(tests[prog_name], entry)

        save_tests_json(tests)
    else:
        # Default: generate unit tests (original behavior)
        tests = {}

        for bench_file in bench_files:
            bench_path = os.path.join(NETLISTS_DIR, bench_file)
            with open(bench_path) as f:
                bench_text = f.read()

            bs_text = bench2bs(bench_text)
            program = parse(bs_text)
            prog_name = "netlist_" + os.path.splitext(bench_file)[0]

            inputs_list, _, _ = parse_bench(bench_text)
            n_inputs = len(inputs_list)

            entries = []

            if bench_file == "c17.bench":
                entry = generate_c17(bench_text, program)
                data_filename = f"{prog_name}_exhaustive_32.bsdata"
                write_bsdata(
                    os.path.join(TESTS_DIR, data_filename),
                    entry["bitlength"],
                    inputs=entry["inputs"],
                    expected=entry["expected"],
                )
                entries.append({
                    "name": entry["name"],
                    "bitlength": entry["bitlength"],
                    "data_file": data_filename,
                })
            elif bench_file == "adder4.bench":
                entry = generate_adder4(bench_text, program)
                data_filename = f"{prog_name}_exhaustive.bsdata"
                write_bsdata(
                    os.path.join(TESTS_DIR, data_filename),
                    entry["bitlength"],
                    inputs=entry["inputs"],
                    expected=entry["expected"],
                )
                entries.append({
                    "name": entry["name"],
                    "bitlength": entry["bitlength"],
                    "data_file": data_filename,
                })
            else:
                # Generic netlist: random test + conditional wide test
                entry = generate_generic_random(bench_text, program, seed=42)
                data_filename = f"{prog_name}_random.bsdata"
                write_bsdata(
                    os.path.join(TESTS_DIR, data_filename),
                    entry["bitlength"],
                    inputs=entry["inputs"],
                    expected=entry["expected"],
                )
                entries.append({
                    "name": entry["name"],
                    "bitlength": entry["bitlength"],
                    "data_file": data_filename,
                })

            # Wide test for netlists with >= 20 inputs
            if n_inputs >= 20:
                entries.append({
                    "name": "Large 1M vectors (verify 200)",
                    "category": "generated",
                    "bitlength": 1000000,
                    "generate": {"seed": 99, "n_verify": 200},
                })

            # Preserve any existing tier entries for this program
            existing = load_tests_json()
            if prog_name in existing:
                tier_data_files = {
                    f"{prog_name}_{t}.bsdata"
                    for t in TIER_SEED_BASE
                }
                for existing_entry in existing[prog_name]:
                    if existing_entry.get("data_file") in tier_data_files:
                        entries.append(existing_entry)

            tests[prog_name] = entries
            print(f"  {prog_name}: {len(entries)} test(s) ({n_inputs} inputs)")

        save_tests_json(tests)
        total = sum(len(v) for v in tests.values())
        print(f"circuit_sim: generated {total} test entries for {len(tests)} netlists")


if __name__ == "__main__":
    main()
