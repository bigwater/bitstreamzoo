#!/usr/bin/env python3
"""Standalone test harness for SIMON 32/64 block cipher.

Verifies the bitstream SIMON against a Python scalar reference.
Each bit position = a different plaintext being encrypted simultaneously.

SIMON 32/64: 32-bit block (2x16-bit halves), 64-bit key, 32 rounds.

References:
  Beaulieu et al., "The SIMON and SPECK Families of Lightweight
  Block Ciphers", IACR ePrint 2013/404
"""

from __future__ import annotations

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))


# ── Reference implementation ──────────────────────────────────────

def rotl16(x: int, n: int) -> int:
    """Left rotate 16-bit value."""
    return ((x << n) | (x >> (16 - n))) & 0xFFFF


def simon_key_schedule(key: list[int]) -> list[int]:
    """Expand 4x16-bit key words to 32 round keys for SIMON 32/64.

    key: [k0, k1, k2, k3] where k3 is MSW.
    Returns list of 32 round keys (each 16-bit).
    """
    # z0 sequence from the SIMON paper (z[0] is the leftmost character)
    _z0 = [int(c) for c in
           "11111010001001010110000111001101111101000100101011000011100110"]
    c = 0xFFFC  # 2^n - 4
    rk = list(key)  # k[0..3]
    for i in range(4, 32):
        tmp = rotl16(rk[i - 1], 16 - 3)  # right rotate by 3
        tmp ^= rk[i - 3]
        tmp ^= rotl16(tmp, 16 - 1)  # XOR with right rotate by 1
        tmp ^= rk[i - 4]
        tmp ^= c ^ _z0[(i - 4) % 62]
        rk.append(tmp & 0xFFFF)
    return rk


def simon_encrypt(L: int, R: int, round_keys: list[int]) -> tuple[int, int]:
    """Encrypt one block with SIMON 32/64.

    L, R: 16-bit halves. round_keys: 32 round keys.
    Returns (cipherL, cipherR).
    """
    for r in range(32):
        f = (rotl16(L, 1) & rotl16(L, 8)) ^ rotl16(L, 2)
        new_L = R ^ f ^ round_keys[r]
        R = L
        L = new_L
    return L, R


# ── Encoding helpers ──────────────────────────────────────────────

def bitslice_16bit(values: list[int]) -> dict[int, int]:
    """Pack K 16-bit values into 16 streams.

    Returns {b: stream} where bit k of stream b = bit b of values[k].
    """
    K = len(values)
    arrays = {}
    for b in range(16):
        bits = 0
        for k in range(K):
            if (values[k] >> b) & 1:
                bits |= 1 << k
        arrays[b] = bits
    return arrays


def encode_round_keys(round_keys: list[int]) -> dict[int, int]:
    """Encode 32 round keys (each 16-bit) as 512 broadcast streams.

    Returns {r*16+b: stream} where stream is all-1s if bit b of round_keys[r], else 0.
    """
    arrays = {}
    for r in range(32):
        for b in range(16):
            arrays[r * 16 + b] = -1 if ((round_keys[r] >> b) & 1) else 0
    return arrays


def decode_16bit(result: dict, name: str, K: int) -> list[int]:
    """Extract K 16-bit values from bitsliced output array."""
    values = []
    for k in range(K):
        v = 0
        for b in range(16):
            if (result[name].get(b, 0) >> k) & 1:
                v |= 1 << b
        values.append(v)
    return values


# ── Wide test ─────────────────────────────────────────────────────

def run_simon_wide(prog, W: int, seed: int = 42, n_verify: int = 200,
                   verbose: bool = False, backend=None) -> tuple[bool, int]:
    """Large-scale SIMON: W plaintexts with random key, spot-check."""
    if backend is None:
        from benchmark.base import PythonBackend
        backend = PythonBackend()

    rng = random.Random(seed)

    # Random key
    key = [rng.randint(0, 0xFFFF) for _ in range(4)]
    round_keys = simon_key_schedule(key)

    # Generate W-bit random streams for plaintext
    plainL_arrays = {b: rng.getrandbits(W) for b in range(16)}
    plainR_arrays = {b: rng.getrandbits(W) for b in range(16)}

    # Encode round keys
    rk_arrays = encode_round_keys(round_keys)

    result, ops, _exec_ms = backend.run(
        prog, inputs={}, params={},
        input_arrays={"plainL": plainL_arrays, "plainR": plainR_arrays,
                      "round_key": rk_arrays},
        bitlength=W)

    # Spot-check
    positions = rng.sample(range(W), min(n_verify, W))
    ok = True
    for k in positions:
        L = sum(((plainL_arrays[b] >> k) & 1) << b for b in range(16))
        R = sum(((plainR_arrays[b] >> k) & 1) << b for b in range(16))
        ref_L, ref_R = simon_encrypt(L, R, round_keys)

        got_L = sum(((result["cipherL"].get(b, 0) >> k) & 1) << b for b in range(16))
        got_R = sum(((result["cipherR"].get(b, 0) >> k) & 1) << b for b in range(16))

        if got_L != ref_L or got_R != ref_R:
            if verbose:
                print(f"  SPOT-CHECK FAIL block {k}: "
                      f"got=({got_L:04x},{got_R:04x}) "
                      f"ref=({ref_L:04x},{ref_R:04x})")
            ok = False
            break

    return ok, ops, _exec_ms


# ── Generated test handler ────────────────────────────────────────

def run_generated(case, prog, backend=None):
    """Handle generated (wide) test cases for the generic runner."""
    g = case["generate"]
    return run_simon_wide(prog, W=g["W"], seed=g["seed"],
                          verbose=True, backend=backend)


# ── Main ──────────────────────────────────────────────────────────

def main():
    from benchmark.base import GenericDomain

    _name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    domain = GenericDomain(_name)
    results = domain.run_all()

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print("SIMON 32/64 Cipher Benchmark")
    print()
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.test_name}  ({r.bitlength} blocks, {r.op_count} ops)")

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
