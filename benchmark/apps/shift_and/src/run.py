#!/usr/bin/env python3
"""Standalone test harness for DNA Shift-And bitstream program.

Verifies exact string matching against Python reference implementation.
Each bit position is one character position in a DNA text.
"""

from __future__ import annotations

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from simulator.pythonsim.interpreter import Interpreter


ALPHABET = "ACGT"


# ── Reference implementation ─────────────────────────────────────

def shift_and_reference(text: str, pattern: str) -> set[int]:
    """Shift-And exact string matching. Returns set of match-end positions."""
    m = len(pattern)
    if m == 0:
        return set()

    # Build character masks
    B = {}
    for c in ALPHABET:
        mask = 0
        for j in range(m):
            if pattern[j] == c:
                mask |= 1 << j
        B[c] = mask

    R = 0
    match_bit = 1 << (m - 1)
    matches = set()
    for i, c in enumerate(text):
        R = ((R << 1) | 1) & B.get(c, 0)
        if R & match_bit:
            matches.add(i)

    return matches


# ── Bitstream execution ──────────────────────────────────────────

def run_shift_and_bs(program, text: str, pattern: str) -> tuple[set[int], Interpreter]:
    """Run Shift-And on a single text/pattern pair using bitstreams.

    Each bit position k represents text character k.
    Returns (set of match-end positions, interpreter).
    """
    N = len(text)
    M = len(pattern)

    # Build basis streams: is_X[k] = 1 if text[k] == X
    basis = {}
    for c in ALPHABET:
        bits = 0
        for k in range(N):
            if text[k] == c:
                bits |= 1 << k
        basis[c] = bits

    # Build pattern mask arrays: mask_X[j] = ONES if pattern[j] == X, else ZERO
    mask_arrays = {}
    for c in ALPHABET:
        arr = {}
        for j in range(M):
            arr[j] = -1 if pattern[j] == c else 0  # ONES = -1, ZERO = 0
        mask_arrays[c] = arr

    interp = Interpreter()
    result = interp.run(
        program,
        inputs={
            "is_A": basis["A"],
            "is_C": basis["C"],
            "is_G": basis["G"],
            "is_T": basis["T"],
        },
        params={"M": M},
        input_arrays={
            "mask_A": mask_arrays["A"],
            "mask_C": mask_arrays["C"],
            "mask_G": mask_arrays["G"],
            "mask_T": mask_arrays["T"],
        },
    )

    # Extract match positions from result
    matches_bits = result["matches"]
    matches = set()
    for k in range(N):
        if matches_bits & (1 << k):
            matches.add(k)

    return matches, interp


# ── Test runner ──────────────────────────────────────────────────

def run_shift_and_wide(prog, N: int, pattern: str,
                       seed: int = 42,
                       verbose: bool = False,
                       backend=None) -> tuple[bool, int]:
    """Large-scale Shift-And: N-bp random text, generated as wide bitstreams.

    Instead of generating a string and transposing to basis streams (O(N^2/64)
    due to repeated big-int |= 1<<k), generates 4 basis streams directly from
    2 random N-bit integers:
      b0, b1 = getrandbits(N), getrandbits(N)
      is_A = ~b0 & ~b1,  is_C = ~b0 & b1,  is_G = b0 & ~b1,  is_T = b0 & b1
    This gives uniform random DNA in O(1) bitstream operations.
    """
    if backend is None:
        from benchmark.base import PythonBackend
        backend = PythonBackend()

    M = len(pattern)
    mask = (1 << N) - 1
    rng = random.Random(seed)

    # Generate random DNA text directly as basis bitstreams — instant
    b0 = rng.getrandbits(N)
    b1 = rng.getrandbits(N)
    basis = {
        "A": (~b0 & ~b1) & mask,
        "C": (~b0 & b1) & mask,
        "G": (b0 & ~b1) & mask,
        "T": (b0 & b1) & mask,
    }

    # Build pattern mask arrays (small, from the pattern string)
    mask_arrays = {}
    for c in ALPHABET:
        arr = {}
        for j in range(M):
            arr[j] = -1 if pattern[j] == c else 0
        mask_arrays[c] = arr

    # Run bitstream
    result, ops, _exec_ms = backend.run(
        prog,
        inputs={
            "is_A": basis["A"], "is_C": basis["C"],
            "is_G": basis["G"], "is_T": basis["T"],
        },
        params={"M": M},
        input_arrays={
            "mask_A": mask_arrays["A"], "mask_C": mask_arrays["C"],
            "mask_G": mask_arrays["G"], "mask_T": mask_arrays["T"],
        },
        bitlength=N,
    )
    matches_bits = result["matches"]

    # Reconstruct text from basis streams using to_bytes for O(N) efficiency
    # (avoids O(N^2/64) of shifting big ints character-by-character)
    n_bytes = (N + 7) // 8
    a_bytes = basis["A"].to_bytes(n_bytes, byteorder='little')
    c_bytes = basis["C"].to_bytes(n_bytes, byteorder='little')
    g_bytes = basis["G"].to_bytes(n_bytes, byteorder='little')

    text_buf = bytearray(N)
    for k in range(N):
        bi, bm = k >> 3, 1 << (k & 7)
        if a_bytes[bi] & bm:   text_buf[k] = 65   # 'A'
        elif c_bytes[bi] & bm: text_buf[k] = 67   # 'C'
        elif g_bytes[bi] & bm: text_buf[k] = 71   # 'G'
        else:                  text_buf[k] = 84   # 'T'
    text = text_buf.decode('ascii')

    # Run scalar reference on the reconstructed text
    ref_matches = shift_and_reference(text, pattern)

    # Extract match positions from bitstream result using to_bytes
    m_bytes = matches_bits.to_bytes(n_bytes, byteorder='little')
    bs_matches = set()
    for k in range(N):
        if m_bytes[k >> 3] & (1 << (k & 7)):
            bs_matches.add(k)

    ok = bs_matches == ref_matches
    if not ok and verbose:
        diff = bs_matches.symmetric_difference(ref_matches)
        print(f"  MISMATCH: {len(diff)} positions differ")

    return ok, ops, _exec_ms


def run_test(name: str, text: str, pattern: str, program,
             verbose: bool = False) -> tuple[bool, int]:
    """Run Shift-And test comparing bitstream vs reference.
    Returns (ok, op_count)."""
    ref_matches = shift_and_reference(text, pattern)
    bs_matches, interp = run_shift_and_bs(program, text, pattern)

    ok = bs_matches == ref_matches
    if not ok and verbose:
        print(f"  MISMATCH in {name}")
        print(f"    pattern: {pattern}")
        print(f"    text:    {text[:80]}...")
        print(f"    ref matches:  {sorted(ref_matches)}")
        print(f"    bs matches:   {sorted(bs_matches)}")

    return ok, interp.op_count


# ── Test data ────────────────────────────────────────────────────

def random_dna(length: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(ALPHABET) for _ in range(length))


# ── Real-data loading ────────────────────────────────────────────

def load_reallife_dna(dataset):
    """Load .npz DNA basis streams -> (basis dict, seq_len).

    Returns:
        basis: dict with keys 'A','C','G','T' -> int bitstreams
        seq_len: int, number of base pairs
    """
    import numpy as np
    npz_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")
    paths = {
        "sars_cov2": "small/sars_cov2.npz",
        "ecoli_k12": "medium/ecoli_k12.npz",
        "hg38_chr1_2m": "small/hg38_chr1_2m.npz",
        "hg38_chr1_20m": "medium/hg38_chr1_20m.npz",
        "hg38_chr1_200m": "large/hg38_chr1_200m.npz",
    }
    data = np.load(os.path.join(npz_dir, paths[dataset]))
    seq_len = int(data["sequence_length"])
    basis = {}
    for key, name in [("bA", "A"), ("bC", "C"), ("bG", "G"), ("bT", "T")]:
        basis[name] = int.from_bytes(data[key].tobytes(), 'little')
    return basis, seq_len


def _verify_spot_check(basis, matches_bits, seq_len, n_bytes, pattern,
                       op_count, n_windows=10, window_size=100000):
    """Spot-check large genome results by verifying random windows.

    Instead of reconstructing all 249M characters, sample random windows
    and run the scalar reference on each window independently.
    """
    M = len(pattern)
    a_bytes = basis["A"].to_bytes(n_bytes, byteorder='little')
    c_bytes = basis["C"].to_bytes(n_bytes, byteorder='little')
    g_bytes = basis["G"].to_bytes(n_bytes, byteorder='little')
    m_bytes = matches_bits.to_bytes(n_bytes, byteorder='little')

    rng = random.Random(42)
    max_start = seq_len - window_size - M
    if max_start < 0:
        max_start = 0
        window_size = seq_len

    for _ in range(n_windows):
        start = rng.randint(0, max_start)
        end = min(start + window_size, seq_len)

        # Reconstruct text for this window
        text_buf = bytearray(end - start)
        for k in range(start, end):
            bi, bm = k >> 3, 1 << (k & 7)
            if a_bytes[bi] & bm:   text_buf[k - start] = 65
            elif c_bytes[bi] & bm: text_buf[k - start] = 67
            elif g_bytes[bi] & bm: text_buf[k - start] = 71
            else:                  text_buf[k - start] = 84
        window_text = text_buf.decode('ascii')

        # Run scalar reference on window
        ref_matches = shift_and_reference(window_text, pattern)

        # Compare: ref match position p in window = global position start+p
        # Only check matches that are fully within the window (not near edges)
        for p in ref_matches:
            global_pos = start + p
            if m_bytes[global_pos >> 3] & (1 << (global_pos & 7)) == 0:
                return False, op_count

        # Check bitstream matches in interior of window are in ref
        for k in range(start + M - 1, end):
            bs_match = bool(m_bytes[k >> 3] & (1 << (k & 7)))
            ref_match = (k - start) in ref_matches
            if bs_match != ref_match:
                return False, op_count

    return True, op_count


def prepare_file_inputs(gen_config, datasets_dir):
    """Load NPZ data and return (inputs, params, input_arrays, bitlength).

    Used by run_generated() and other callers that load these datasets.
    """
    dataset = gen_config.get("dataset")
    pattern = gen_config.get("pattern", "GATTACA")
    if not dataset:
        return None
    import numpy as np
    npz_dir = datasets_dir
    paths = {
        "sars_cov2": "small/sars_cov2.npz",
        "ecoli_k12": "medium/ecoli_k12.npz",
        "hg38_chr1_2m": "small/hg38_chr1_2m.npz",
        "hg38_chr1_20m": "medium/hg38_chr1_20m.npz",
        "hg38_chr1_200m": "large/hg38_chr1_200m.npz",
    }
    if dataset not in paths:
        return None
    npz_path = os.path.join(npz_dir, paths[dataset])
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    seq_len = int(data["sequence_length"])
    inputs = {}
    for key, name in [("bA", "is_A"), ("bC", "is_C"), ("bG", "is_G"), ("bT", "is_T")]:
        inputs[name] = int.from_bytes(data[key].tobytes(), 'little')
    M = len(pattern)
    input_arrays = {}
    for c in ALPHABET:
        arr = {}
        for j in range(M):
            arr[j] = -1 if pattern[j] == c else 0
        input_arrays[f"mask_{c}"] = arr
    return inputs, {"M": M}, input_arrays, seq_len


# ── Main ─────────────────────────────────────────────────────────

def run_generated(case, prog, backend=None):
    """Handle generated test cases for the generic runner.

    Dispatches based on case["generate"]["type"]:
      - "wide": large-scale random DNA (existing)
      - "file": real-world DNA from .npz dataset
    """
    if backend is None:
        from benchmark.base import PythonBackend
        backend = PythonBackend()

    g = case["generate"]
    gen_type = g.get("type", "wide")

    if gen_type == "wide":
        pattern = random_dna(g["pattern_len"], g["seed_pat"])
        return run_shift_and_wide(prog, N=g["N"],
                                  pattern=pattern, seed=g["seed_text"],
                                  verbose=True, backend=backend)
    elif gen_type == "file":
        basis, seq_len = load_reallife_dna(g["dataset"])
        pattern = g["pattern"]
        M = len(pattern)

        # Build pattern mask arrays
        mask_arrays = {}
        for c in ALPHABET:
            arr = {}
            for j in range(M):
                arr[j] = -1 if pattern[j] == c else 0
            mask_arrays[c] = arr

        # Run bitstream program
        result, ops, _exec_ms = backend.run(
            prog,
            inputs={
                "is_A": basis["A"], "is_C": basis["C"],
                "is_G": basis["G"], "is_T": basis["T"],
            },
            params={"M": M},
            input_arrays={
                "mask_A": mask_arrays["A"], "mask_C": mask_arrays["C"],
                "mask_G": mask_arrays["G"], "mask_T": mask_arrays["T"],
            },
            bitlength=seq_len,
        )
        matches_bits = result["matches"]

        n_bytes = (seq_len + 7) // 8

        # For large genomes (>10M bp), use spot-check on random windows
        # to avoid O(N) Python loops on 249M characters.
        if seq_len > 10_000_000:
            ok, ops2 = _verify_spot_check(basis, matches_bits, seq_len,
                                           n_bytes, pattern, ops)
            return ok, ops2, _exec_ms

        # Full verification for small/medium datasets
        a_bytes = basis["A"].to_bytes(n_bytes, byteorder='little')
        c_bytes = basis["C"].to_bytes(n_bytes, byteorder='little')
        g_bytes = basis["G"].to_bytes(n_bytes, byteorder='little')
        m_bytes = matches_bits.to_bytes(n_bytes, byteorder='little')

        # Reconstruct text
        text_buf = bytearray(seq_len)
        for k in range(seq_len):
            bi, bm = k >> 3, 1 << (k & 7)
            if a_bytes[bi] & bm:   text_buf[k] = 65   # 'A'
            elif c_bytes[bi] & bm: text_buf[k] = 67   # 'C'
            elif g_bytes[bi] & bm: text_buf[k] = 71   # 'G'
            else:                  text_buf[k] = 84   # 'T'
        text = text_buf.decode('ascii')

        # Verify using scalar reference
        ref_matches = shift_and_reference(text, pattern)

        # Extract bitstream match positions
        bs_matches = set()
        for k in range(seq_len):
            if m_bytes[k >> 3] & (1 << (k & 7)):
                bs_matches.add(k)

        ok = bs_matches == ref_matches
        return ok, ops, _exec_ms


def main():
    from benchmark.base import GenericDomain

    _name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    domain = GenericDomain(_name)
    results = domain.run_all()

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print("DNA Shift-And Bitstream Benchmark")
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
