#!/usr/bin/env python3
"""Standalone test harness for polar code encoding bitstream programs.

Verifies the bitstream polar encoder against a Python reference implementation.
Each bit position in a stream is an independent message being encoded in parallel.

Polar code encoding computes x = u * F^{otimes n} where F = [[1,0],[1,1]].
This is a butterfly XOR network, the core operation in 5G NR polar codes.

Source: Arikan, "Channel Polarization: A Method for Constructing
Capacity-Achieving Codes for Symmetric Binary-Input Memoryless Channels",
IEEE Trans. Information Theory, 2009.
"""

from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from simulator.pythonsim.interpreter import Interpreter


# ── Reference polar encoder ──────────────────────────────────────

def polar_encode_reference(u: list[int], N: int) -> list[int]:
    """Scalar butterfly polar encoder.

    Args:
        u: list of N bits (0 or 1), the input message.
        N: code length (must be power of 2).

    Returns:
        list of N encoded bits.
    """
    x = list(u)
    n = int(math.log2(N))
    for s in range(n):
        stride = 1 << s
        for j in range(0, N, 2 * stride):
            for i in range(stride):
                x[j + i] ^= x[j + i + stride]
    return x


# ── Bitstream execution ──────────────────────────────────────────

def run_polar_bs(program, messages: list[list[int]], N: int,
                 backend=None, prog=None) -> tuple[dict[int, int], int, float]:
    """Run polar encoding on K messages simultaneously.

    Args:
        program: parsed Program (used if backend is None)
        messages: list of K messages, each a list of N bits.
        N: code length.
        backend: optional Backend instance.
        prog: optional ProgramInfo (required when backend is provided).

    Returns:
        (x_array dict, op_count, exec_ms)
    """
    K = len(messages)

    # Pack: u[i] is a stream where bit j = messages[j][i]
    u_arrays = {}
    for i in range(N):
        bits = 0
        for j in range(K):
            if messages[j][i]:
                bits |= 1 << j
        u_arrays[i] = bits

    if backend is not None and prog is not None:
        result, ops, exec_ms = backend.run(
            prog, inputs={}, params={},
            input_arrays={"u": u_arrays}, bitlength=K)
    else:
        interp = Interpreter()
        result = interp.run(
            program, inputs={}, params={},
            input_arrays={"u": u_arrays})
        ops = interp.op_count
        exec_ms = 0.0

    return result["x"], ops, exec_ms


def unpack_codewords(x_array: dict[int, int], K: int,
                     N: int) -> list[list[int]]:
    """Extract K individual codewords from the output array."""
    codewords = []
    for j in range(K):
        cw = []
        for i in range(N):
            cw.append((x_array.get(i, 0) >> j) & 1)
        codewords.append(cw)
    return codewords


# ── Wide (large-scale) test ──────────────────────────────────────

def run_polar_wide(prog, N: int, W: int, n_verify: int = 200,
                   seed: int = 42, backend=None) -> tuple[bool, int, float]:
    """Large-scale polar encoding: W messages of N bits.

    Generates N input streams each as a W-bit random integer,
    then spot-checks n_verify random messages against the scalar reference.
    """
    if backend is None:
        from benchmark.base import PythonBackend
        backend = PythonBackend()

    rng = random.Random(seed)

    # Generate N input streams, each W bits wide
    u_arrays = {i: rng.getrandbits(W) for i in range(N)}

    # Run bitstream on all W messages simultaneously
    result, ops, exec_ms = backend.run(
        prog, inputs={}, params={},
        input_arrays={"u": u_arrays}, bitlength=W)
    x_array = result["x"]

    # Spot-check n_verify random messages against reference
    verify_rng = random.Random(seed + 1)
    verify_indices = sorted(verify_rng.sample(range(W), min(n_verify, W)))
    ok = True
    for j in verify_indices:
        # Extract message j's input bits
        u_bits = [(u_arrays[i] >> j) & 1 for i in range(N)]
        # Extract message j's encoded output
        x_bits = [(x_array.get(i, 0) >> j) & 1 for i in range(N)]
        # Compare against scalar reference
        ref = polar_encode_reference(u_bits, N)
        if x_bits != ref:
            ok = False
            break

    return ok, ops, exec_ms


# ── Generated test handler ───────────────────────────────────────

def run_generated(case, prog, backend=None):
    """Handle generated test cases for the generic runner.

    Dispatches based on case["generate"]["type"]:
      - "random": random N-bit messages, parameterized by K, N, seed
    """
    if backend is None:
        from benchmark.base import PythonBackend
        backend = PythonBackend()

    g = case["generate"]
    gen_type = g["type"]

    if gen_type == "random":
        N = g["N"]
        W = g["W"]
        seed = g["seed"]
        return run_polar_wide(prog, N=N, W=W, seed=seed, backend=backend)
    else:
        raise ValueError(f"Unknown generate type: {gen_type}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    from benchmark.base import GenericDomain

    _name = os.path.basename(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    domain = GenericDomain(_name)
    results = domain.run_all()

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print("Polar Code Encoding Bitstream Benchmark")
    print()
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.program}: {r.test_name}  "
              f"({r.bitlength} vectors, {r.op_count} ops)")

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
