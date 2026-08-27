#!/usr/bin/env python3
"""Standalone test harness for Montgomery modular multiplication.

Computes r = a * b * R^{-1} mod n where R = 2^K.
Each bit position = a different (a, b) pair multiplied modulo n.
"""

from __future__ import annotations
import os, sys, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))


# ── Reference implementation ──────────────────────────────────

def montgomery_mul_reference(a: int, b: int, n: int, np_val: int, K: int) -> int:
    """Scalar reference: compute a*b*R^{-1} mod n, R=2^K."""
    R = 1 << K
    mask = R - 1
    t = a * b
    m = ((t & mask) * np_val) & mask
    u = m * n
    s = t + u
    assert s % R == 0, f"s={s} not divisible by R={R}"
    r = s >> K
    if r >= n:
        r -= n
    return r


def compute_np(n: int, K: int) -> int:
    """Compute n' = -n^{-1} mod 2^K via Newton's iteration."""
    R = 1 << K
    assert n & 1 == 1, f"n must be odd, got {n}"
    x = 1
    for _ in range(K):
        x = (x * (2 - n * x)) % R
    np_val = (-x) % R
    assert (n * np_val + 1) % R == 0, f"n'={np_val} invalid for n={n}"
    return np_val


# ── Encoding / Decoding ───────────────────────────────────────

def encode_operands(values: list[int], K: int) -> dict[int, int]:
    """Bitslice M K-bit integers into K streams."""
    M = len(values)
    streams = {}
    for b in range(K):
        bits = 0
        for m_idx in range(M):
            if (values[m_idx] >> b) & 1:
                bits |= 1 << m_idx
        streams[b] = bits
    return streams


def encode_broadcast(value: int, K: int) -> dict[int, int]:
    """Encode a K-bit constant as K broadcast streams (0 or -1)."""
    return {b: (-1 if ((value >> b) & 1) else 0) for b in range(K)}


def decode_result(result_streams: dict, K: int, M: int) -> list[int]:
    """Extract M K-bit integers from output streams."""
    values = []
    for m_idx in range(M):
        val = 0
        for b in range(K):
            if (result_streams.get(b, 0) >> m_idx) & 1:
                val |= 1 << b
        values.append(val)
    return values


# ── Bitstream execution ───────────────────────────────────────

def run_montgomery_bs(program, a_vals, b_vals, n_val, K):
    """Run montgomery_mul.bs and return (results, op_count)."""
    from simulator.pythonsim.interpreter import Interpreter

    M = len(a_vals)
    np_val = compute_np(n_val, K)
    PK = 2 * K

    a_arr = encode_operands(a_vals, K)
    b_arr = encode_operands(b_vals, K)
    n_arr = encode_broadcast(n_val, K)
    np_arr = encode_broadcast(np_val, K)

    interp = Interpreter()
    result = interp.run(
        program, inputs={},
        params={"K": K, "PK": PK},
        input_arrays={"a": a_arr, "b": b_arr, "n": n_arr, "np": np_arr},
    )
    r_streams = result["r"]
    results = decode_result(r_streams, K, M)
    return results, interp.op_count


# ── Generated test handler ────────────────────────────────────

def run_generated(case, prog, backend=None):
    """Handle generated (wide) test cases for the generic runner."""
    if backend is None:
        from benchmark.base import PythonBackend
        backend = PythonBackend()

    g = case["generate"]
    seed = g["seed"]
    K = g["K"]
    n_val = g["n"]
    np_val = g["np"]
    W = case["bitlength"]
    PK = 2 * K

    rng = random.Random(seed)
    max_val = n_val - 1

    # Generate random values in [0, n), bitslice them
    a_values = [rng.randint(0, max_val) for _ in range(W)]
    b_values = [rng.randint(0, max_val) for _ in range(W)]
    a_arr = encode_operands(a_values, K)
    b_arr = encode_operands(b_values, K)
    n_arr = encode_broadcast(n_val, K)
    np_arr = encode_broadcast(np_val, K)

    result, ops, exec_ms = backend.run(
        prog, inputs={}, params={"K": K, "PK": PK},
        input_arrays={"a": a_arr, "b": b_arr, "n": n_arr, "np": np_arr},
        bitlength=W,
    )
    r_streams = result["r"]

    # Spot-check
    n_verify = min(200, W)
    positions = rng.sample(range(W), n_verify)
    ok = True
    for m in positions:
        a_val = a_values[m]
        b_val = b_values[m]
        ref = montgomery_mul_reference(a_val, b_val, n_val, np_val, K)
        got = 0
        for b in range(K):
            if (r_streams.get(b, 0) >> m) & 1:
                got |= 1 << b
        if got != ref:
            ok = False
            break

    return ok, ops, exec_ms
