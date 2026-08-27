#!/usr/bin/env python3
"""
Benchmark harness for circuit simulation bitstream programs.

Key idea: each bit position in a stream is a DIFFERENT test vector.
One execution of the .bs program evaluates the circuit on all test
vectors simultaneously.
"""

from __future__ import annotations

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from simulator.pythonsim.interpreter import Interpreter
from benchmark.apps.circuit_sim.datasets.bench2bs import parse_bench, simulate_bench


# ── Generated (wide) test runner ──────────────────────────────

def run_generic_netlist_wide(prog, bench_text, seed=99, n_verify=200,
                             backend=None):
    """Large-scale test: 1M random vectors, spot-check against reference. Returns (ok, ops)."""
    if backend is None:
        from benchmark.base import PythonBackend
        backend = PythonBackend()

    inputs_list, outputs_list, _ = parse_bench(bench_text)
    bitlength = 1_000_000

    rng = random.Random(seed)
    input_streams = {name: rng.getrandbits(bitlength) for name in inputs_list}

    result, ops, _exec_ms = backend.run(prog, inputs=input_streams, bitlength=bitlength)

    verify_rng = random.Random(seed + 1)
    verify_indices = sorted(verify_rng.sample(range(bitlength), n_verify))
    ok = True
    for t in verify_indices:
        vec = {name: (input_streams[name] >> t) & 1 for name in inputs_list}
        ref = simulate_bench(bench_text, vec)
        for out_name in outputs_list:
            got = (result[out_name] >> t) & 1
            expected = ref[out_name]
            if got != expected:
                print(f"  FAIL t={t} output={out_name}: got={got} expected={expected}")
                ok = False
                break
        if not ok:
            break

    return ok, ops, _exec_ms


NETLISTS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "raw")


def _get_bench_text(prog):
    """Load the .bench netlist corresponding to a netlist_<name> program."""
    # prog.name is "netlist_c1355" -> bench file is "c1355.bench"
    bench_name = prog.name.removeprefix("netlist_") + ".bench"
    bench_path = os.path.join(NETLISTS_DIR, bench_name)
    with open(bench_path) as f:
        return f.read()


def run_generated(case, prog, backend=None):
    """Handle generated (wide) test cases for the generic runner."""
    g = case["generate"]
    bench_text = _get_bench_text(prog)
    return run_generic_netlist_wide(prog, bench_text,
                                   seed=g["seed"],
                                   n_verify=g.get("n_verify", 200),
                                   backend=backend)


def main():
    from benchmark.base import GenericDomain

    _name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    domain = GenericDomain(_name)
    results = domain.run_all()

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print("Circuit Simulation Bitstream Benchmark")
    print()
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.program} {r.test_name}  ({r.bitlength} vectors, {r.op_count} ops)")

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
