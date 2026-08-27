#!/usr/bin/env python3
"""Standalone test harness for Trivium stream cipher bitstream program.

Verifies the bitstream Trivium against a Python reference implementation.
Each bit position is a different (key, IV) pair being encrypted.

Reference test vectors from the ECRYPT Stream Cipher Project.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from simulator.pythonsim.interpreter import Interpreter


# ── Reference Trivium implementation ─────────────────────────────

def trivium_reference(key_bits: list[int], iv_bits: list[int], length: int) -> list[int]:
    """Trivium stream cipher reference implementation.

    Args:
        key_bits: 80 bits (key[0] = MSB, key[79] = LSB in ECRYPT convention;
                  here key[i] = bit i of the key register).
        iv_bits: 80 bits.
        length: number of keystream bits to generate.

    Returns:
        list of keystream bits.
    """
    # Initialize state (1-indexed internally, 0-indexed arrays)
    s = [0] * 288

    # Register 1: s[0..92], load key into s[0..79]
    for i in range(80):
        s[i] = key_bits[i]
    # s[80..92] = 0

    # Register 2: s[93..176], load IV into s[93..172]
    for i in range(80):
        s[93 + i] = iv_bits[i]
    # s[173..176] = 0

    # Register 3: s[177..287], set s[285..287] = 1
    s[285] = 1
    s[286] = 1
    s[287] = 1

    # Warmup: 1152 blank clocks
    for _ in range(1152):
        t1 = s[65] ^ s[92]
        t2 = s[161] ^ s[176]
        t3 = s[242] ^ s[287]

        fb1 = t1 ^ (s[90] & s[91]) ^ s[170]
        fb2 = t2 ^ (s[174] & s[175]) ^ s[263]
        fb3 = t3 ^ (s[285] & s[286]) ^ s[68]

        # Shift registers
        for j in range(92, 0, -1):
            s[j] = s[j - 1]
        s[0] = fb3

        for j in range(176, 93, -1):
            s[j] = s[j - 1]
        s[93] = fb1

        for j in range(287, 177, -1):
            s[j] = s[j - 1]
        s[177] = fb2

    # Generate keystream
    z = []
    for _ in range(length):
        t1 = s[65] ^ s[92]
        t2 = s[161] ^ s[176]
        t3 = s[242] ^ s[287]

        z.append(t1 ^ t2 ^ t3)

        fb1 = t1 ^ (s[90] & s[91]) ^ s[170]
        fb2 = t2 ^ (s[174] & s[175]) ^ s[263]
        fb3 = t3 ^ (s[285] & s[286]) ^ s[68]

        for j in range(92, 0, -1):
            s[j] = s[j - 1]
        s[0] = fb3

        for j in range(176, 93, -1):
            s[j] = s[j - 1]
        s[93] = fb1

        for j in range(287, 177, -1):
            s[j] = s[j - 1]
        s[177] = fb2

    return z


# ── Bitstream execution ──────────────────────────────────────────

def run_trivium_bs(program, keys: list[list[int]],
                   ivs: list[list[int]], length: int) -> tuple[list[list[int]], Interpreter]:
    """Run Trivium on multiple (key, IV) pairs simultaneously.

    Args:
        keys: list of K key bit-lists, each 80 bits.
        ivs: list of K IV bit-lists, each 80 bits.
        length: number of keystream bits to generate.

    Returns:
        (list of K keystream bit-lists each of `length` bits, interpreter)
    """
    K = len(keys)

    # Pack key arrays: key[i] = stream where bit j = keys[j][i]
    key_arrays = {}
    for i in range(80):
        bits = 0
        for j in range(K):
            if keys[j][i]:
                bits |= 1 << j
        key_arrays[i] = bits

    # Pack IV arrays
    iv_arrays = {}
    for i in range(80):
        bits = 0
        for j in range(K):
            if ivs[j][i]:
                bits |= 1 << j
        iv_arrays[i] = bits

    interp = Interpreter()
    result = interp.run(
        program,
        inputs={},
        params={"L": length},
        input_arrays={"key": key_arrays, "iv": iv_arrays},
    )

    # Unpack output keystream
    z_array = result["z"]
    keystreams = []
    for j in range(K):
        ks = []
        for k in range(length):
            ks.append(1 if z_array.get(k, 0) & (1 << j) else 0)
        keystreams.append(ks)

    return keystreams, interp


# ── Test helpers ─────────────────────────────────────────────────

def hex_to_bits(hex_str: str, nbits: int) -> list[int]:
    """Convert hex string to bit list (bit 0 first)."""
    val = int(hex_str, 16)
    return [(val >> i) & 1 for i in range(nbits)]


def bits_to_hex(bits: list[int]) -> str:
    """Convert bit list to hex string."""
    val = 0
    for i, b in enumerate(bits):
        val |= b << i
    nbytes = (len(bits) + 7) // 8
    return format(val, f"0{nbytes * 2}x")


def bits_to_ecrypt_hex(bits: list[int]) -> str:
    """Convert bit list to hex using ECRYPT convention (LSB-first within bytes).

    ECRYPT stores bit i at position (i mod 8) of byte (i // 8).
    """
    hex_str = ""
    for k in range((len(bits) + 7) // 8):
        byte = 0
        for j in range(8):
            idx = 8 * k + j
            if idx < len(bits) and bits[idx]:
                byte |= 1 << j
        hex_str += format(byte, "02X")
    return hex_str


def run_trivium_wide(prog, W: int, length: int = 64,
                     n_verify: int = 200, seed: int = 42,
                     verbose: bool = False, backend=None) -> tuple[bool, int]:
    """Large-scale Trivium: W key/IV pairs, generated as wide random bitstreams.

    Generates 80 key streams and 80 IV streams each as W-bit random integers.
    Spot-checks n_verify random pairs against the scalar reference.
    """
    if backend is None:
        from benchmark.base import PythonBackend
        backend = PythonBackend()

    import random
    rng = random.Random(seed)

    # Generate 160 input streams (80 key + 80 IV), each W bits wide — instant
    key_arrays = {i: rng.getrandbits(W) for i in range(80)}
    iv_arrays = {i: rng.getrandbits(W) for i in range(80)}

    # Run bitstream on all W pairs simultaneously
    result, ops, _exec_ms = backend.run(
        prog,
        inputs={},
        params={"L": length},
        input_arrays={"key": key_arrays, "iv": iv_arrays},
        bitlength=W,
    )
    z_array = result["z"]

    # Spot-check n_verify random pairs against reference
    verify_rng = random.Random(seed + 1)
    verify_indices = sorted(verify_rng.sample(range(W), min(n_verify, W)))
    ok = True
    for j in verify_indices:
        # Extract key/IV pair j
        key_bits = [(key_arrays[i] >> j) & 1 for i in range(80)]
        iv_bits = [(iv_arrays[i] >> j) & 1 for i in range(80)]
        # Extract keystream j
        got_bits = [(z_array.get(k, 0) >> j) & 1 for k in range(length)]
        # Compare against scalar reference
        ref_bits = trivium_reference(key_bits, iv_bits, length)
        if got_bits != ref_bits:
            ok = False
            if verbose:
                print(f"  MISMATCH pair {j}")
            break

    return ok, ops, _exec_ms


def run_test_ecrypt_vector(program, verbose=False) -> tuple[bool, int]:
    """Verify all-zero key/IV keystream against ECRYPT published test vector.

    ECRYPT Stream Cipher Project test vector for Key=0, IV=0:
      First 64 bits of keystream: FBE0BF265859051B
    This validates the bitstream program against a completely external source
    (not our own reference implementation).
    """
    key_zero = [0] * 80
    iv_zero = [0] * 80

    # Run bitstream
    keystreams, interp = run_trivium_bs(program, [key_zero], [iv_zero], 64)

    got = bits_to_ecrypt_hex(keystreams[0])
    expected = "FBE0BF265859051B"
    ok = got == expected
    if not ok and verbose:
        print(f"  ECRYPT vector mismatch: got {got}, expected {expected}")

    # Also verify our reference produces the same bits
    ref_bits = trivium_reference(key_zero, iv_zero, 64)
    ref_hex = bits_to_ecrypt_hex(ref_bits)
    if ref_hex != expected:
        ok = False
        if verbose:
            print(f"  Reference also wrong: got {ref_hex}, expected {expected}")

    return ok, interp.op_count


def run_test(name: str, keys: list[list[int]], ivs: list[list[int]], length: int,
             program, verbose: bool = False) -> tuple[bool, int]:
    """Run Trivium test comparing bitstream vs reference.
    Returns (ok, op_count)."""
    # Reference
    refs = [trivium_reference(k, iv, length) for k, iv in zip(keys, ivs)]

    # Bitstream
    keystreams, interp = run_trivium_bs(program, keys, ivs, length)

    ok = True
    for j in range(len(keys)):
        if keystreams[j] != refs[j]:
            ok = False
            if verbose:
                print(f"  MISMATCH key/iv pair {j}")
                print(f"    ref: {bits_to_hex(refs[j][:64])}")
                print(f"    bs:  {bits_to_hex(keystreams[j][:64])}")

    return ok, interp.op_count


# ── Generated test handler ──────────────────────────────────────

def run_generated(case, prog, backend=None):
    """Handle generated (wide) test cases for the generic runner."""
    g = case["generate"]
    return run_trivium_wide(prog, W=g["W"],
                            length=g["length"], seed=g["seed"],
                            verbose=True, backend=backend)


# ── Main ─────────────────────────────────────────────────────────

def main():
    from benchmark.base import GenericDomain

    _name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    domain = GenericDomain(_name)
    results = domain.run_all()

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print("Trivium Bitstream Benchmark")
    print()
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.test_name}  ({r.bitlength} vectors, {r.op_count} ops)")

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
