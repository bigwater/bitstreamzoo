#!/usr/bin/env python3
"""Standalone test harness for epistasis (2x2 contingency table) bitstream program.

Verifies the bitstream contingency table against a Python reference.
Each bit position represents one individual (sample) in the GWAS study.

Source primitive: BOOST/GBOOST Boolean bitwise contingency-table
construction. EpiGPU is related GPU exhaustive-scan work.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from simulator.pythonsim.interpreter import Interpreter


# ── Reference implementation ──────────────────────────────────────

def epistasis_reference(snp_a: int, snp_b: int, valid: int, width: int
                        ) -> tuple[int, int, int, int]:
    """Reference: count each cell of the 2x2 contingency table by looping.

    Returns (n_11, n_10, n_01, n_00).
    """
    n_11 = n_10 = n_01 = n_00 = 0
    for i in range(width):
        if not ((valid >> i) & 1):
            continue
        a = (snp_a >> i) & 1
        b = (snp_b >> i) & 1
        if a and b:
            n_11 += 1
        elif a and not b:
            n_10 += 1
        elif not a and b:
            n_01 += 1
        else:
            n_00 += 1
    return n_11, n_10, n_01, n_00


# ── Bitstream execution ──────────────────────────────────────────

def run_epistasis_bs(program, snp_a: int, snp_b: int, valid: int
                     ) -> tuple[tuple[int, int, int, int], Interpreter]:
    """Run epistasis bitstream program.

    Returns ((n_11, n_10, n_01, n_00), interpreter).
    """
    interp = Interpreter()
    result = interp.run(
        program,
        inputs={"snp_a": snp_a, "snp_b": snp_b, "valid": valid},
    )
    return (result["n_11"], result["n_10"], result["n_01"], result["n_00"]), interp


# ── Test helpers ──────────────────────────────────────────────────

def run_test(name: str, snp_a: int, snp_b: int, valid: int, width: int,
             program, verbose: bool = False) -> tuple[bool, int]:
    """Run one epistasis test, comparing bitstream vs reference.

    Returns (ok, op_count).
    """
    ref = epistasis_reference(snp_a, snp_b, valid, width)
    bs_result, interp = run_epistasis_bs(program, snp_a, snp_b, valid)

    ok = bs_result == ref
    if not ok and verbose:
        print(f"  MISMATCH {name}:")
        print(f"    bitstream: {bs_result}")
        print(f"    reference: {ref}")

    return ok, interp.op_count


# ── Main ──────────────────────────────────────────────────────────

def main():
    from benchmark.base import GenericDomain

    _name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    domain = GenericDomain(_name)
    results = domain.run_all()

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print("Epistasis (2x2 Contingency Table) Bitstream Benchmark")
    print()
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.test_name}  ({r.bitlength} individuals, {r.op_count} ops)")

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
