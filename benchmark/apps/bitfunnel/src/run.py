#!/usr/bin/env python3
"""Standalone test harness for BitFunnel bit-sliced signature row intersection.

Core query matching kernel from BitFunnel (Goodwin et al., SIGIR 2017).
Each bit position is one document; each input stream is one row of the
bit-sliced index.  The program AND-intersects the rows selected by the query,
expanding higher-rank rows to their rank-0 equivalents first.

Reference: Goodwin et al., "BitFunnel: Revisiting Signatures for Search",
  SIGIR 2017.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from simulator.pythonsim.interpreter import Interpreter


# ── Reference implementation ──────────────────────────────────────

def bitfunnel_reference(rows_r0: dict, rows_r1: dict, rows_r2: dict,
                        K0: int, K1: int, K2: int,
                        N: int, HALF: int, QUARTER: int) -> tuple[int, int]:
    """Scalar reference for BitFunnel row intersection with rank expansion.

    rows_r{k}: dict mapping index → stream (Python int, N-bit bitvector).
    Returns (match_stream, count).
    """
    mask = (1 << N) - 1
    acc = mask  # ONES: all documents are candidates
    for i in range(K0):
        acc &= rows_r0[i]
    for i in range(K1):
        row = rows_r1[i]
        acc &= row | (row << HALF)
    for i in range(K2):
        row = rows_r2[i]
        stage = row | (row << QUARTER)
        acc &= stage | (stage << HALF)
    acc &= mask
    return acc, bin(acc).count('1')


# ── Bitstream execution ───────────────────────────────────────────

def run_bitfunnel_bs(program, rows_r0: dict, rows_r1: dict, rows_r2: dict,
                     K0: int, K1: int, K2: int,
                     N: int, HALF: int, QUARTER: int):
    """Run BitFunnel .bs program via interpreter.

    Returns (count, matches_stream, interpreter).
    """
    interp = Interpreter(bitlength=N)
    result = interp.run(
        program,
        inputs={},
        params={"K0": K0, "K1": K1, "K2": K2, "HALF": HALF, "QUARTER": QUARTER},
        input_arrays={
            "rows_r0": rows_r0,
            "rows_r1": rows_r1,
            "rows_r2": rows_r2,
        },
    )
    return result["count"], result.get("matches", 0), interp


# ── Main ──────────────────────────────────────────────────────────

def main():
    from benchmark.base import GenericDomain

    _name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    domain = GenericDomain(_name)
    results = domain.run_all()

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print("BitFunnel Bit-Sliced Signature Row Intersection Benchmark")
    print()
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.test_name}  ({r.bitlength} docs, {r.op_count} ops)")

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
