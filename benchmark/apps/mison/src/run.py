#!/usr/bin/env python3
"""Standalone test harness for Mison/simdjson JSON structural index pipeline.

Verifies the bitstream pipeline against a Python reference implementation.
Each bit position is one character position in the JSON text.

4-stage pipeline:
  1. Character classification from Parabix basis bit-planes
  2. Escape detection via carry-propagating backslash run analysis
  3. String mask computation via prefix XOR
  4. Structural character filtering

Sources: Mison (Li et al., VLDB 2017), simdjson (Langdale & Lemire, VLDBJ 2019)
"""

from __future__ import annotations

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from simulator.pythonsim.interpreter import Interpreter


# ── Helper functions ──────────────────────────────────────────────────

def text_to_basis_streams(text: str) -> dict[int, int]:
    """Convert ASCII text to 8 basis bit-planes.

    Returns dict {k: stream} where bit i of stream k = bit k of ord(text[i]).
    """
    b = {k: 0 for k in range(8)}
    for i, ch in enumerate(text):
        code = ord(ch)
        for k in range(8):
            if (code >> k) & 1:
                b[k] |= 1 << i
    return b


def make_even_bits(n: int) -> int:
    """Create alternating 010101... pattern of n bits (bit 0 = 1)."""
    if n <= 0:
        return 0
    # Build using doubling: start with 8-bit chunk, double until >= n
    result = 0x55
    width = 8
    while width < n:
        result = result | (result << width)
        width *= 2
    return result & ((1 << n) - 1)


def compute_log_w(n: int) -> int:
    """Compute LOG_W: smallest k such that 2^k >= n."""
    if n <= 1:
        return 0
    return (n - 1).bit_length()


# ── Reference implementation ──────────────────────────────────────────

def mison_reference(text: str) -> dict[str, int]:
    """Python reference for JSON structural index construction.

    Returns dict with: quote, backslash, real_quote, str_mask, structural
    """
    n = len(text)
    mask = (1 << n) - 1

    # Stage 1: Character classification
    quote = 0
    backslash = 0
    lbrace = 0
    rbrace = 0
    lbracket = 0
    rbracket = 0
    colon = 0
    comma = 0

    for i, ch in enumerate(text):
        bit = 1 << i
        if ch == '"':
            quote |= bit
        elif ch == '\\':
            backslash |= bit
        elif ch == '{':
            lbrace |= bit
        elif ch == '}':
            rbrace |= bit
        elif ch == '[':
            lbracket |= bit
        elif ch == ']':
            rbracket |= bit
        elif ch == ':':
            colon |= bit
        elif ch == ',':
            comma |= bit

    # Stage 2: Escape detection
    # Find positions preceded by odd-length backslash runs
    escaped = 0
    i = 0
    while i < n:
        if (backslash >> i) & 1:
            run_start = i
            while i < n and (backslash >> i) & 1:
                i += 1
            run_len = i - run_start
            if run_len % 2 == 1 and i < n:
                escaped |= 1 << i
        else:
            i += 1

    real_quote = quote & ~escaped & mask

    # Stage 3: String mask (running XOR parity of real quotes)
    str_mask = 0
    in_string = False
    for i in range(n):
        if (real_quote >> i) & 1:
            in_string = not in_string
        if in_string:
            str_mask |= 1 << i

    # Stage 4: Structural filtering
    struct_raw = lbrace | rbrace | lbracket | rbracket | colon | comma
    structural = struct_raw & ~str_mask & mask

    return {
        'real_quote': real_quote & mask,
        'str_mask': str_mask & mask,
        'structural': structural & mask,
    }


# ── Bitstream execution ──────────────────────────────────────────────

def run_mison_bs(program, text: str) -> tuple[dict[str, int], int]:
    """Run the Mison bitstream program on a text string.

    Returns (output_dict, op_count).
    """
    n = len(text)
    b = text_to_basis_streams(text)
    even_bits = make_even_bits(n)
    log_w = compute_log_w(n)

    result, ops, _exec_ms = run_mison_bs_raw(program, b, even_bits, log_w)
    return result, ops


def run_mison_bs_raw(program, b: dict[int, int], even_bits: int,
                     log_w: int, backend=None, prog=None,
                     bitlength: int = 0) -> tuple[dict[str, int], int]:
    """Run the Mison bitstream program on raw basis streams.

    Returns (output_dict, op_count).
    """
    _exec_ms = 0.0
    if backend is not None and prog is not None:
        result, ops, _exec_ms = backend.run(
            prog, inputs={"even_bits": even_bits},
            params={"LOG_W": log_w},
            input_arrays={"b": b}, bitlength=bitlength)
    else:
        interp = Interpreter()
        result = interp.run(
            program,
            inputs={"even_bits": even_bits},
            params={"LOG_W": log_w},
            input_arrays={"b": b},
        )
        ops = interp.op_count
    return result, ops, _exec_ms


# ── Test helpers ──────────────────────────────────────────────────────

def run_test(name: str, text: str, program,
             verbose: bool = False) -> tuple[bool, int]:
    """Run one Mison test case, comparing bitstream vs reference.

    Returns (ok, op_count).
    """
    n = len(text)
    mask = (1 << n) - 1

    ref = mison_reference(text)
    bs_result, ops = run_mison_bs(program, text)

    ok = True
    for key in ['real_quote', 'str_mask', 'structural']:
        bs_val = bs_result[key] & mask
        ref_val = ref[key]
        if bs_val != ref_val:
            ok = False
            if verbose:
                print(f"  MISMATCH {name} [{key}]:")
                print(f"    bitstream: {bs_val:#x}")
                print(f"    reference: {ref_val:#x}")
                print(f"    diff:      {(bs_val ^ ref_val):#x}")

    return ok, ops


def random_json(rng: random.Random, target_size: int) -> str:
    """Generate a random JSON-like string of approximately target_size bytes.

    ~20% of string values contain escape sequences (\\n, \\t, \\", \\\\)
    to exercise the escape detection stage.
    """
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    escape_seqs = ['\\\\', '\\"', '\\n', '\\t']
    parts = ['{']
    first = True

    while len(''.join(parts)) < target_size - 5:
        if not first:
            parts.append(',')
        first = False

        # Random key
        key_len = rng.randint(1, 4)
        key = ''.join(rng.choice(chars) for _ in range(key_len))

        # Random value
        val_type = rng.choice(['num', 'str', 'arr'])
        if val_type == 'num':
            val = str(rng.randint(0, 999))
        elif val_type == 'str':
            val_len = rng.randint(1, 6)
            val_str = ''.join(rng.choice(chars) for _ in range(val_len))
            # ~20% chance of inserting escape sequences
            if rng.random() < 0.2 and val_len >= 2:
                esc = rng.choice(escape_seqs)
                pos = rng.randint(0, len(val_str) - 1)
                val_str = val_str[:pos] + esc + val_str[pos:]
            val = f'"{val_str}"'
        else:
            val = f'[{rng.randint(0, 9)}]'

        pair = f'"{key}":{val}'
        if len(''.join(parts)) + len(pair) + 2 > target_size:
            break
        parts.append(pair)

    parts.append('}')
    return ''.join(parts)


# ── Real-data loading ────────────────────────────────────────────────

def load_reallife_json(dataset):
    """Load .npz JSON basis streams -> (b dict, text_length).

    Returns:
        b: dict {k: int} for k in 0..7, basis bit-planes as big ints
        text_length: int, number of characters
    """
    import numpy as np
    npz_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")
    paths = {
        "gharchive_small": "small/gharchive_1m.npz",
        "gharchive_medium": "medium/gharchive_10m.npz",
        "gharchive_large": "large/gharchive_100m.npz",
    }
    data = np.load(os.path.join(npz_dir, paths[dataset]))
    tl = data["text_length"]
    text_length = int(tl) if tl.ndim == 0 else int(tl[0])
    b = {}
    for k in range(8):
        b[k] = int.from_bytes(data[f"b{k}"].tobytes(), 'little')
    return b, text_length


def prepare_file_inputs(gen_config, datasets_dir):
    """Load NPZ data and return (inputs, params, input_arrays, bitlength).

    Used by run_generated() and other callers that load these datasets.
    """
    dataset = gen_config.get("dataset")
    if not dataset:
        return None
    import numpy as np
    npz_dir = datasets_dir
    paths = {
        "gharchive_small": "small/gharchive_1m.npz",
        "gharchive_medium": "medium/gharchive_10m.npz",
        "gharchive_large": "large/gharchive_100m.npz",
    }
    if dataset not in paths:
        return None
    npz_path = os.path.join(npz_dir, paths[dataset])
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    tl = data["text_length"]
    text_length = int(tl) if tl.ndim == 0 else int(tl[0])
    b = {}
    for k in range(8):
        b[k] = int.from_bytes(data[f"b{k}"].tobytes(), 'little')
    even_bits = make_even_bits(text_length)
    log_w = compute_log_w(text_length)
    return {"even_bits": even_bits}, {"LOG_W": log_w}, {"b": b}, text_length


# ── Generated test handler ───────────────────────────────────────────

def run_generated(case, prog, backend=None):
    """Handle generated test cases for the generic runner.

    Dispatches based on case["generate"]["type"]:
      - "file": real-world JSON from .npz dataset
    """
    if backend is None:
        from benchmark.base import PythonBackend
        backend = PythonBackend()

    g = case["generate"]
    b, text_length = load_reallife_json(g["dataset"])
    even_bits = make_even_bits(text_length)
    log_w = compute_log_w(text_length)
    mask = (1 << text_length) - 1

    result, ops, _exec_ms = run_mison_bs_raw(prog.program, b, even_bits, log_w,
                                              backend=backend, prog=prog,
                                              bitlength=text_length)

    structural = result['structural'] & mask
    str_mask = result['str_mask'] & mask

    # Invariant: structural chars must be outside strings
    ok = (structural & str_mask) == 0
    if not ok:
        return False, ops, _exec_ms

    # Spot-check: verify structural positions are actually structural chars
    struct_chars = set('{}[]:,')
    rng = random.Random(42)
    positions = rng.sample(range(text_length), min(200, text_length))
    for pos in positions:
        code = sum(((b[k] >> pos) & 1) << k for k in range(8))
        ch = chr(code) if code < 128 else '?'
        is_structural_char = ch in struct_chars
        pos_is_structural = (structural >> pos) & 1
        pos_in_string = (str_mask >> pos) & 1
        if is_structural_char and not pos_in_string:
            if not pos_is_structural:
                return False, ops, _exec_ms
        elif pos_is_structural and not is_structural_char:
            return False, ops, _exec_ms

    return True, ops, _exec_ms


# ── Main ──────────────────────────────────────────────────────────────

def main():
    from benchmark.base import GenericDomain

    _name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    domain = GenericDomain(_name)
    results = domain.run_all()

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print("Mison/simdjson JSON Structural Index Pipeline")
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
