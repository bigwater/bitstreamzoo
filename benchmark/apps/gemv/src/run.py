#!/usr/bin/env python3
"""Standalone test harness for gate-decomposed GEMV.

Computes y = W * x with K-bit precision, fully decomposed into
AND/XOR/OR gates.  M independent instances processed in parallel.
"""

from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from simulator.pythonsim.interpreter import Interpreter


# ── Reference ─────────────────────────────────────────────────────

def gemv_reference(x_values: list[list[int]],
                   w_values: list[list[int]]) -> list[list[int]]:
    """Scalar reference: y = W * x for each of M data items.

    Args:
        x_values: M x N (M items, N input features, each K-bit int)
        w_values: L x N weight matrix (K-bit ints, broadcast to all M)

    Returns:
        M x L result matrix
    """
    M = len(x_values)
    L = len(w_values)
    N = len(w_values[0])
    results = []
    for m in range(M):
        row = []
        for l in range(L):
            dp = sum(x_values[m][f] * w_values[l][f] for f in range(N))
            row.append(dp)
        results.append(row)
    return results


# ── Encoding ──────────────────────────────────────────────────────

def compute_B(N: int, K: int) -> int:
    if N <= 1:
        return 2 * K
    return 2 * K + math.ceil(math.log2(N))


def encode_vector(values: list[list[int]], N: int, K: int) -> dict[int, int]:
    """Pack M items' N features (K bits each) into N*K streams."""
    M = len(values)
    streams = {}
    for f in range(N):
        for b in range(K):
            bits = 0
            for m in range(M):
                if (values[m][f] >> b) & 1:
                    bits |= 1 << m
            streams[f * K + b] = bits
    return streams


def encode_weights(w_values: list[list[int]], L: int, N: int,
                   K: int) -> dict[int, int]:
    """Broadcast L*N weights (K bits each) to streams."""
    streams = {}
    for l in range(L):
        for f in range(N):
            for b in range(K):
                streams[l * N * K + f * K + b] = (
                    -1 if ((w_values[l][f] >> b) & 1) else 0)
    return streams


def decode_result(result_streams: dict, L: int, B: int,
                  M: int) -> list[list[int]]:
    """Extract M x L result matrix from L*B output streams."""
    results = []
    for m in range(M):
        row = []
        for l in range(L):
            val = 0
            for b in range(B):
                if (result_streams.get(l * B + b, 0) >> m) & 1:
                    val |= 1 << b
            row.append(val)
        results.append(row)
    return results


# ── Bitstream execution ──────────────────────────────────────────

def run_gemv_bs(program, x_values, w_values, K):
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
    return result, interp


# ── Generated test handler ────────────────────────────────────────

def run_generated(case, prog, backend=None):
    if backend is None:
        from benchmark.base import PythonBackend
        backend = PythonBackend()

    g = case["generate"]
    gen_type = g["type"]

    if gen_type == "wide":
        rng = random.Random(g["seed"])
        L, N, K, W = g["L"], g["N"], g["K"], g["W"]
        PK = 2 * K
        B = compute_B(N, K)
        max_val = (1 << K) - 1

        # Random weights (broadcast)
        w_values = [[rng.randint(0, max_val) for _ in range(N)]
                     for _ in range(L)]

        # Random input streams
        x_arrays = {f * K + b: rng.getrandbits(W)
                    for f in range(N) for b in range(K)}
        w_arrays = encode_weights(w_values, L, N, K)

        result, ops, exec_ms = backend.run(
            prog, inputs={},
            params={"L": L, "N": N, "K": K, "PK": PK, "B": B},
            input_arrays={"x": x_arrays, "w": w_arrays},
            bitlength=W)
        y_streams = result["y"]

        # Spot-check
        n_verify = min(200, W)
        positions = rng.sample(range(W), n_verify)
        ok = True
        for m in positions:
            x_vals = []
            for f in range(N):
                val = 0
                for b in range(K):
                    if (x_arrays[f * K + b] >> m) & 1:
                        val |= 1 << b
                x_vals.append(val)

            for l in range(L):
                ref_dp = sum(x_vals[f] * w_values[l][f] for f in range(N))
                bs_dp = 0
                for b in range(B):
                    if (y_streams.get(l * B + b, 0) >> m) & 1:
                        bs_dp |= 1 << b
                if bs_dp != ref_dp:
                    ok = False
                    break
            if not ok:
                break

        return ok, ops, exec_ms


# ── Main ──────────────────────────────────────────────────────────

def main():
    from benchmark.base import GenericDomain
    _name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    domain = GenericDomain(_name)
    results = domain.run_all()
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print(f"\nGEMV: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
