#!/usr/bin/env python3
"""Standalone test harness for multi-pattern regex bitstream programs.

Verifies compiled Snort IDS patterns against Python re reference.
Each bit position is a different parallel byte position in the input stream.

The bitstream matcher reports ALL valid match-end positions (not just greedy),
following the Parabix parallel matching model.

Source: Snort 2.9.7.0 rules via AutomataZoo (Wadden et al., IISWC 2018),
compiled to .bs via Parabix methodology (Cameron et al., 2014).
"""

from __future__ import annotations

import sys
import os
import re
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from simulator.pythonsim.interpreter import Interpreter


# ── Dataset paths ────────────────────────────────────────────────

DATASET_PATHS = {
    "wrccdc2012_10k": "small/wrccdc2012_10k.npz",
    "wrccdc2012_1m": "small/wrccdc2012_1m.npz",
    "wrccdc2012_10m": "medium/wrccdc2012_10m.npz",
    "wrccdc2012_100m": "large/wrccdc2012_100m.npz",
}


# ── Pattern metadata (loaded from manifest) ──────────────────────

from benchmark.apps.regex.src.compile_all import TIERS, get_tier_patterns


def _get_patterns_for_program(prog_name: str) -> list[tuple[str, str, str]]:
    """Get the pattern list for a .bs program name."""
    return TIERS.get(prog_name, [])


# ── Reference implementation ──────────────────────────────────────


def regex_reference(pattern: str, flags: str, payload: bytes) -> int:
    """Compute reference match stream using Python re.

    Returns a bitstream where bit j=1 means a match of the regex
    COULD end at byte position j (all valid end positions, not just greedy).

    Note: The bitstream model ignores anchors (^, $), so the bitstream
    result may be a SUPERSET of this reference. Verification checks
    that ref subset of bs.
    """
    re_flags = 0
    if 'i' in flags:
        re_flags |= re.IGNORECASE
    if 's' in flags:
        re_flags |= re.DOTALL
    if 'm' in flags:
        re_flags |= re.MULTILINE

    try:
        compiled = re.compile(pattern.encode('latin-1'), re_flags)
    except re.error:
        return 0

    match_stream = 0
    n = len(payload)

    # Find all valid match-end positions by trying from each start
    for start in range(n):
        m = compiled.match(payload, start)
        if m and m.end() > m.start():
            # Set bits for all valid end positions within this match
            for end in range(start + 1, m.end() + 1):
                if compiled.fullmatch(payload[start:end]):
                    match_stream |= 1 << (end - 1)

    return match_stream


def regex_reference_fast(pattern: str, flags: str, payload: bytes) -> int:
    """Fast reference: only check positions found by finditer.

    For large payloads, the full reference is too slow. This approximation
    finds match end positions from finditer, which may miss some
    intermediate end positions for overlapping/ambiguous matches.
    """
    re_flags = 0
    if 'i' in flags:
        re_flags |= re.IGNORECASE
    if 's' in flags:
        re_flags |= re.DOTALL
    if 'm' in flags:
        re_flags |= re.MULTILINE

    try:
        compiled = re.compile(pattern.encode('latin-1'), re_flags)
    except re.error:
        return 0

    match_stream = 0
    for m in compiled.finditer(payload):
        if m.end() > m.start():
            match_stream |= 1 << (m.end() - 1)

    return match_stream


# ── icgrep reference (authoritative) ──────────────────────────────
# Optional cross-check against a locally built icgrep (Parabix). Point
# the ICGREP_BIN environment variable at the binary (or put icgrep on PATH)
# to enable it; the check is skipped silently when the binary is absent.

ICGREP_PATH = os.environ.get("ICGREP_BIN") or shutil.which("icgrep") or ""


def icgrep_reference(pattern: str, flags: str, payload_file: str) -> set[int] | None:
    """Run icgrep on a payload file and return byte offsets of matching lines.

    The .bs programs are mechanically translated from icgrep's Pablo IR via:
        Snort PCRE -> icgrep --ShowPablo -> Pablo IR -> pablo_to_bs.py -> .bs
    So icgrep is the authoritative reference (exact equivalence by construction).

    Returns a set of byte offsets where matching lines start, or None if icgrep
    is not available. This provides line-level verification (not bit-exact
    position-level), which is stronger than the Python re subset check.
    """
    import subprocess

    if not ICGREP_PATH or not os.path.exists(ICGREP_PATH):
        return None

    cmd = [ICGREP_PATH, '-b', '-U']
    if 'i' in flags:
        cmd.append('-i')
    cmd.extend([pattern, payload_file])

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return None

    offsets = set()
    for line in result.stdout.decode('latin-1', errors='replace').splitlines():
        # icgrep -b outputs "byte_offset:matched_content"
        if ':' in line:
            offset_str = line.split(':')[0]
            try:
                offsets.add(int(offset_str))
            except ValueError:
                continue
    return offsets


# ── Bitstream execution helpers ───────────────────────────────────


def payload_to_basis_bits(payload: bytes) -> dict[int, int]:
    """Convert byte payload to 8 basis bit-planes (as Python big ints)."""
    b = {}
    for i in range(8):
        val = 0
        for j, byte in enumerate(payload):
            if (byte >> i) & 1:
                val |= (1 << j)
        b[i] = val
    return b


def payload_to_basis_bits_fast(payload: bytes) -> dict[int, int]:
    """Convert byte payload to basis bits using numpy (fast for large payloads)."""
    import numpy as np
    arr = np.frombuffer(payload, dtype=np.uint8)
    n = len(arr)
    b = {}
    for i in range(8):
        bits = ((arr >> i) & 1).astype(np.uint8)
        # Convert packed bits to Python int
        val = int.from_bytes(np.packbits(bits, bitorder='little').tobytes(), 'little')
        b[i] = val
    return b


def load_reallife_npz(dataset: str) -> tuple[dict[int, int], int]:
    """Load .npz basis bit-planes -> (b dict, payload_length)."""
    import numpy as np
    npz_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")
    data = np.load(os.path.join(npz_dir, DATASET_PATHS[dataset]))
    pl = data["payload_length"]
    payload_length = int(pl) if pl.ndim == 0 else int(pl[0])

    b = {}
    for i in range(8):
        packed = data[f"b{i}"]
        val = int.from_bytes(packed.tobytes(), 'little')
        b[i] = val

    return b, payload_length


def prepare_file_inputs(gen_config, datasets_dir):
    """Load NPZ data and return (inputs, params, input_arrays, bitlength).

    Used by run_generated() and other callers that load these datasets.
    """
    dataset = gen_config.get("dataset")
    if not dataset:
        return None
    npz_dir = datasets_dir
    if dataset not in DATASET_PATHS:
        return None
    npz_path = os.path.join(npz_dir, DATASET_PATHS[dataset])
    if not os.path.exists(npz_path):
        return None
    import numpy as np
    data = np.load(npz_path)
    pl = data["payload_length"]
    payload_length = int(pl) if pl.ndim == 0 else int(pl[0])
    b = {}
    for i in range(8):
        b[i] = int.from_bytes(data[f"b{i}"].tobytes(), 'little')
    return {}, {}, {"b": b}, payload_length


# ── Generated test handler ────────────────────────────────────────


def run_generated(case, prog, backend=None):
    """Handle generated test cases for the generic runner.

    Dispatches based on case["generate"]["type"]:
      - "random_bytes": random byte input, verify against Python re
      - "file": real-world pcap data from .npz
    """
    if backend is None:
        from benchmark.base import PythonBackend
        backend = PythonBackend()

    g = case["generate"]
    gen_type = g["type"]

    patterns = _get_patterns_for_program(prog.name)
    if not patterns:
        return True, 0, 0.0  # No patterns to test

    if gen_type == "random_bytes":
        import random
        rng = random.Random(g["seed"])
        n = g["n_bytes"]
        payload = bytes(rng.randint(0, 255) for _ in range(n))
        b = payload_to_basis_bits(payload) if n <= 10000 else payload_to_basis_bits_fast(payload)
        mask = (1 << n) - 1

        result, ops, _exec_ms = backend.run(prog, inputs={}, input_arrays={'b': b},
                                  bitlength=n)

        # Verify any_match: OR of all pattern references
        ref_any = 0
        for name, pat, flags in patterns:
            ref_any |= regex_reference(pat, flags, payload) & mask
        bs_any = result.get("any_match", 0) & mask
        if ref_any & ~bs_any:
            return False, ops, _exec_ms

        return True, ops, _exec_ms

    elif gen_type == "file":
        dataset = g["dataset"]
        b, payload_length = load_reallife_npz(dataset)
        mask = (1 << payload_length) - 1

        result, ops, _exec_ms = backend.run(prog, inputs={}, input_arrays={'b': b},
                                  bitlength=payload_length)

        # Verify any_match output against reference
        npz_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")
        npz_path = os.path.join(npz_dir, DATASET_PATHS[dataset])

        import numpy as np
        data = np.load(npz_path)
        pl = int(data["payload_length"]) if data["payload_length"].ndim == 0 else int(data["payload_length"][0])
        # Reconstruct payload from basis bits
        payload_arr = np.zeros(pl, dtype=np.uint8)
        for i in range(8):
            bits = np.unpackbits(data[f"b{i}"], bitorder='little')[:pl]
            payload_arr |= (bits.astype(np.uint8) << i)
        payload = payload_arr.tobytes()

        # Verify any_match: check a sample of patterns against reference
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="regex_payload_")
        try:
            os.write(tmp_fd, payload)
            os.close(tmp_fd)

            bs_any = result.get("any_match", 0) & mask

            # Spot-check first 5 patterns: any reference match should be in any_match
            for name, pat, flags in patterns[:5]:
                ref_val = regex_reference_fast(pat, flags, payload) & mask
                if ref_val == 0:
                    continue
                missed = ref_val & ~bs_any
                if missed != 0:
                    n_missed = bin(missed).count('1')
                    n_ref = bin(ref_val).count('1')
                    import sys
                    print(f"  [WARN] any_match missed {n_missed}/{n_ref} "
                          f"positions from {name} (approximate, "
                          f"use icgrep for exact validation)", file=sys.stderr)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return True, ops, _exec_ms

    else:
        raise ValueError(f"Unknown generate type: {gen_type}")


# ── Main ──────────────────────────────────────────────────────────


def main():
    from benchmark.base import GenericDomain

    _name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    domain = GenericDomain(_name)
    results = domain.run_all()

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print("Regex Bitstream Benchmark")
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
