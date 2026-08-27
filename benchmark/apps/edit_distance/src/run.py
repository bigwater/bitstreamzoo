#!/usr/bin/env python3
"""Standalone test harness for Myers' bit-parallel edit distance.

Verifies the bitstream edit distance against a Python scalar reference.
Each bit position = a different text being compared against the same pattern.

DNA alphabet (A, C, G, T). Computes Levenshtein distance.

References:
  Myers, "A Fast Bit-Vector Algorithm for Approximate String Matching
  Based on Dynamic Programming", JACM 1999
"""

from __future__ import annotations

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))


# ── Reference implementation ──────────────────────────────────────

DNA = "ACGT"


def myers_reference(text: str, pattern: str) -> int:
    """Scalar Myers edit distance for DNA strings."""
    n = len(text)
    m = len(pattern)

    # Simple DP fallback (correct for any size)
    dp = list(range(m + 1))
    for j in range(1, n + 1):
        prev = dp[0]
        dp[0] = j
        for i in range(1, m + 1):
            cost = 0 if text[j - 1] == pattern[i - 1] else 1
            temp = dp[i]
            dp[i] = min(dp[i] + 1, dp[i - 1] + 1, prev + cost)
            prev = temp
    return dp[m]


# ── Encoding helpers ──────────────────────────────────────────────

def encode_text_columns(texts: list[str], N: int) -> dict[str, dict[int, int]]:
    """Pack K texts into positional indicator arrays.

    Returns dict with keys 'pos_A', 'pos_C', 'pos_G', 'pos_T',
    each mapping {j: stream} where bit k of stream j = 1 iff texts[k][j] == X.
    """
    K = len(texts)
    result = {}
    for ch in DNA:
        arrays = {}
        for j in range(N):
            bits = 0
            for k in range(K):
                if j < len(texts[k]) and texts[k][j] == ch:
                    bits |= 1 << k
            arrays[j] = bits
        result[f"pos_{ch}"] = arrays
    return result


def encode_pattern(pattern: str) -> dict[str, dict[int, int]]:
    """Encode pattern into equality mask arrays (broadcast).

    Returns dict with keys 'eq_A', 'eq_C', 'eq_G', 'eq_T',
    each mapping {k: stream} where stream is all-1s if pattern[k] == X, else 0.
    """
    M = len(pattern)
    result = {}
    for ch in DNA:
        arrays = {}
        for k in range(M):
            arrays[k] = -1 if pattern[k] == ch else 0
        result[f"eq_{ch}"] = arrays
    return result


def encode_dist_init(M: int, B: int) -> dict[int, int]:
    """Encode initial distance M as bitsliced broadcast value."""
    arrays = {}
    for b in range(B):
        arrays[b] = -1 if ((M >> b) & 1) else 0
    return arrays


def decode_distance(result: dict, B: int, K: int) -> list[int]:
    """Extract K distances from bitsliced output."""
    distances = []
    for k in range(K):
        d = 0
        for b in range(B):
            if (result["dist"].get(b, 0) >> k) & 1:
                d |= 1 << b
        distances.append(d)
    return distances


# ── Wide test ─────────────────────────────────────────────────────

def run_edit_distance_wide(prog, K: int, N: int, M: int, seed: int = 42,
                           n_verify: int = 200, verbose: bool = False,
                           backend=None) -> tuple[bool, int]:
    """Large-scale edit distance: K texts of length N vs one pattern of length M."""
    if backend is None:
        from benchmark.base import PythonBackend
        backend = PythonBackend()

    rng = random.Random(seed)
    B = max(N, M).bit_length()

    # Generate random pattern
    pattern = ''.join(rng.choice(DNA) for _ in range(M))

    # Generate K random texts of length N as wide streams
    pos_arrays = {f"pos_{ch}": {} for ch in DNA}
    for j in range(N):
        # For each position, randomly assign each of K texts a character
        # We need to generate K random characters and pack them
        a_bits = 0
        c_bits = 0
        g_bits = 0
        t_bits = 0
        for k in range(K):
            ch = rng.choice(DNA)
            if ch == 'A':
                a_bits |= 1 << k
            elif ch == 'C':
                c_bits |= 1 << k
            elif ch == 'G':
                g_bits |= 1 << k
            else:
                t_bits |= 1 << k
        pos_arrays["pos_A"][j] = a_bits
        pos_arrays["pos_C"][j] = c_bits
        pos_arrays["pos_G"][j] = g_bits
        pos_arrays["pos_T"][j] = t_bits

    # Encode pattern (broadcast)
    eq_arrays = encode_pattern(pattern)

    # Encode dist_init
    dist_init = encode_dist_init(M, B)

    # Merge all input arrays
    input_arrays = {}
    for key, val in pos_arrays.items():
        input_arrays[key] = val
    for key, val in eq_arrays.items():
        input_arrays[key] = val
    input_arrays["dist_init"] = dist_init

    result, ops, _exec_ms = backend.run(
        prog, inputs={}, params={"N": N, "M": M, "B": B},
        input_arrays=input_arrays, bitlength=K)

    # Spot-check
    positions = rng.sample(range(K), min(n_verify, K))
    ok = True
    for k in positions:
        # Extract text k
        text = []
        for j in range(N):
            for ch in DNA:
                if (pos_arrays[f"pos_{ch}"][j] >> k) & 1:
                    text.append(ch)
                    break
        text = ''.join(text)

        got_dist = 0
        for b in range(B):
            if (result["dist"].get(b, 0) >> k) & 1:
                got_dist |= 1 << b
        ref_dist = myers_reference(text, pattern)
        if got_dist != ref_dist:
            if verbose:
                print(f"  SPOT-CHECK FAIL text {k}: got={got_dist} ref={ref_dist} "
                      f"text={text[:20]}... pattern={pattern[:20]}...")
            ok = False
            break

    return ok, ops, _exec_ms


# ── Generated test handler ────────────────────────────────────────

def run_generated(case, prog, backend=None):
    """Handle generated (wide) test cases for the generic runner."""
    g = case["generate"]
    return run_edit_distance_wide(prog, K=g["K"], N=g["N"], M=g["M"],
                                  seed=g["seed"], verbose=True, backend=backend)


# ── Main ──────────────────────────────────────────────────────────

def main():
    from benchmark.base import GenericDomain

    _name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    domain = GenericDomain(_name)
    results = domain.run_all()

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print("Edit Distance (Myers) Benchmark")
    print()
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.test_name}  ({r.bitlength} texts, {r.op_count} ops)")

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
