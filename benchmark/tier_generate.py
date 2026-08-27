"""Shared utilities for tiered test data generation.

Each domain's generate_tests.py imports from here for:
- CLI argument parsing (--tier, --describe, --verify)
- Provenance recording
- File size computation
- Common random stream generation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

from benchmark.tier_config import TIER_BITLENGTH, get_tier_bitlength, estimate_size_bytes

# Maximum bits for random.Random.getrandbits() (C int limit on some platforms)
_RANDBITS_CHUNK = (1 << 30)  # ~1 billion bits per chunk


def getrandbits_large(rng, n: int) -> int:
    """Generate n random bits, working around C int limit in getrandbits().

    Python's random.Random.getrandbits() passes the bit count as a C int,
    which overflows for n > 2^31 - 1.  This function generates in chunks
    of ~1 billion bits and assembles the result.
    """
    if n <= _RANDBITS_CHUNK:
        return rng.getrandbits(n)
    result = 0
    shift = 0
    remaining = n
    while remaining > 0:
        bits = min(remaining, _RANDBITS_CHUNK)
        result |= rng.getrandbits(bits) << shift
        shift += bits
        remaining -= bits
    return result


def parse_generate_args(domain: str) -> argparse.Namespace:
    """Parse common CLI arguments for generate_tests.py scripts."""
    parser = argparse.ArgumentParser(
        description=f"Generate test data for {domain}")
    parser.add_argument("--tier", choices=["small", "medium", "large"],
                        help="Generate only the specified tier (default: unit tests only)")
    parser.add_argument("--describe", action="store_true",
                        help="Print provenance info for all generated files")
    parser.add_argument("--verify", action="store_true",
                        help="Verify SHA-256 of existing files")
    return parser.parse_args()


def file_sha256(path: str) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def bsim_hash() -> str | None:
    """Get MD5 hash of bsim binary (for provenance staleness detection)."""
    import hashlib
    bsim = os.path.join(os.path.dirname(__file__), "..", "simulator", "csim", "build", "bsim")
    if not os.path.exists(bsim):
        return None
    h = hashlib.md5()
    with open(bsim, 'rb') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def make_provenance(source: str, seed: int | None = None,
                    description: str = "",
                    generated_by: str = "",
                    url: str | None = None,
                    sha256: str | None = None) -> dict:
    """Build a provenance dict for tests.json entries."""
    prov = {"source": source}
    if seed is not None:
        prov["seed"] = seed
    if description:
        prov["description"] = description
    if generated_by:
        prov["generated_by"] = generated_by
    if url:
        prov["url"] = url
    if sha256:
        prov["sha256"] = sha256
    # Track bsim binary hash for staleness detection
    bh = bsim_hash()
    if bh:
        prov["bsim_md5"] = bh
    return prov


def tier_test_entry(name: str, bitlength: int, data_file: str,
                    size_bytes: int | None = None,
                    provenance: dict | None = None) -> dict:
    """Build a tests.json entry for a tiered test."""
    entry = {
        "name": name,
        "bitlength": bitlength,
        "data_file": data_file,
    }
    if size_bytes is not None:
        entry["size_bytes"] = size_bytes
    if provenance is not None:
        entry["provenance"] = provenance
    return entry


def run_bsim_for_expected(bs_path: str, input_bsdata: str,
                          output_bsdata: str | None = None) -> dict:
    """Run the C++ bsim backend to compute expected outputs.

    Args:
        bs_path: path to .bs program
        input_bsdata: path to inputs-only .bsdata file
        output_bsdata: if provided, write combined .bsdata with expected outputs

    Returns:
        dict with outputs parsed from bsim JSON stdout
    """
    bsim = os.path.join(os.path.dirname(__file__), "..", "simulator", "csim", "build", "bsim")
    if not os.path.exists(bsim):
        raise FileNotFoundError(
            f"bsim not found at {bsim}. Build: cd simulator/csim && mkdir -p build && cd build && cmake .. && make -j")

    cmd = [bsim, bs_path, "--input", input_bsdata]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(f"bsim failed: {result.stderr.strip()}")

    output = json.loads(result.stdout)
    return output


def print_describe(tests_dir: str, tests: list[dict]):
    """Print provenance info for all test entries."""
    for t in tests:
        prov = t.get("provenance", {})
        data_file = t.get("data_file", "")
        path = os.path.join(tests_dir, data_file) if data_file else ""
        size = os.path.getsize(path) if path and os.path.exists(path) else 0
        print(f"  {t['name']}:")
        print(f"    data_file: {data_file}")
        print(f"    bitlength: {t.get('bitlength', t.get('n_vectors', '?'))}")
        print(f"    size: {size:,} bytes ({size / 1e6:.1f} MB)")
        if prov:
            print(f"    source: {prov.get('source', '?')}")
            if 'seed' in prov:
                print(f"    seed: {prov['seed']}")
            if 'description' in prov:
                print(f"    description: {prov['description']}")
            if 'sha256' in prov:
                print(f"    sha256: {prov['sha256']}")


def check_stale(tests: list[dict]) -> list[str]:
    """Check if any tier data was generated with a different bsim binary.
    Returns list of stale test names."""
    current = bsim_hash()
    if not current:
        return []
    stale = []
    for t in tests:
        prov = t.get("provenance", {})
        if not isinstance(prov, dict):
            continue
        old_hash = prov.get("bsim_md5")
        if old_hash and old_hash != current:
            stale.append(t.get("name", "?"))
    return stale


def verify_files(tests_dir: str, tests: list[dict]) -> bool:
    """Verify SHA-256 of existing .bsdata files. Returns True if all pass."""
    all_ok = True
    for t in tests:
        prov = t.get("provenance", {})
        expected_sha = prov.get("sha256")
        if not expected_sha:
            continue
        data_file = t.get("data_file", "")
        if not data_file:
            continue
        path = os.path.join(tests_dir, data_file)
        if not os.path.exists(path):
            print(f"  MISSING: {data_file}")
            all_ok = False
            continue
        actual_sha = file_sha256(path)
        if actual_sha == expected_sha:
            print(f"  OK: {data_file}")
        else:
            print(f"  MISMATCH: {data_file}")
            print(f"    expected: {expected_sha}")
            print(f"    actual:   {actual_sha}")
            all_ok = False
    return all_ok
