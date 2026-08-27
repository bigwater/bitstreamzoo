#!/usr/bin/env python3
"""Standalone test harness for BitWeaving-V range query bitstream program.

Evaluates: SELECT count(*) FROM T WHERE lo <= val <= hi
on a column of unsigned B-bit integers stored as B bit-planes.

Reference: Li & Patel, "BitWeaving: Fast Scans for Main Memory Data
Processing", SIGMOD 2013.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from simulator.pythonsim.interpreter import Interpreter


# ── Reference implementation ──────────────────────────────────────

def bitweaving_reference(values: list[int], lo: int, hi: int) -> int:
    """Count how many values satisfy lo <= val <= hi."""
    return sum(1 for v in values if lo <= v <= hi)


# ── Encoding helpers ──────────────────────────────────────────────

def encode_column(values: list[int], B: int) -> dict[int, int]:
    """Encode a list of B-bit unsigned integers into B bit-plane streams.

    Returns dict: bit_index -> bitstream (Python int).
    Bit-plane b has bit b of each value at the corresponding bit position.
    """
    streams = {}
    for b in range(B):
        bits = 0
        for i, v in enumerate(values):
            if (v >> b) & 1:
                bits |= 1 << i
        streams[b] = bits
    return streams


def encode_bound(val: int, B: int) -> dict[int, int]:
    """Encode a scalar bound into B broadcast streams.

    Each stream is either 0 (all-zeros) or -1 (all-ones, representing ONES).
    """
    streams = {}
    for b in range(B):
        streams[b] = -1 if ((val >> b) & 1) else 0
    return streams


def decode_result(result_stream: int, N: int) -> list[bool]:
    """Extract per-row match bits from result stream."""
    return [bool((result_stream >> i) & 1) for i in range(N)]


# ── Bitstream execution ──────────────────────────────────────────

def make_valid_mask(N: int) -> int:
    """Create a valid mask with bits 0..N-1 set."""
    return (1 << N) - 1


def run_bitweaving_bs(program, values: list[int], lo: int, hi: int, B: int):
    """Run bitweaving bitstream program.

    Returns (count, result_stream, interpreter).
    """
    N = len(values)
    interp = Interpreter()
    result = interp.run(
        program,
        inputs={"valid": make_valid_mask(N)},
        params={"B": B},
        input_arrays={
            "data": encode_column(values, B),
            "lo_bits": encode_bound(lo, B),
            "hi_bits": encode_bound(hi, B),
        },
    )
    return result["count"], result.get("result", 0), interp


# ── Main ──────────────────────────────────────────────────────────

def main():
    from benchmark.base import GenericDomain

    _name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    domain = GenericDomain(_name)
    results = domain.run_all()

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print("BitWeaving-V Range Query Bitstream Benchmark")
    print()
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.test_name}  ({r.bitlength} rows, {r.op_count} ops)")

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
