#!/usr/bin/env python3
"""Generate tiered real-life .npz datasets for mison.

Input format: 8 basis bit-planes (b0..b7) from JSON text characters.
Each plane is ceil(N_chars / 8) bytes, so the .npz is ~= N_chars bytes
total (8 planes plus small text_length/source metadata).

These are optional real-data inputs, separate from the canonical
synthetic benchmark tiers (see datasets/tests/generate_tests.py and
benchmark/tier_config.py).

Tiers (GH Archive character counts):
  small:  1M chars   (~1 MB .npz)
  medium: 10M chars  (~10 MB .npz)
  large:  100M chars (~100 MB .npz)

Source: GH Archive hourly event dump (gharchive_hourly.json.gz) in
        raw/. This is real GitHub API event JSON data.

Usage:
    python make_data.py                   # Generate all tiers
    python make_data.py --tier small      # Generate one tier
"""

import argparse
import gzip
import os
import sys

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(DIR, "raw", "gharchive_hourly.json.gz")

# Target N_chars so that bitstream data is approximately the tier size.
# 8 planes * N_chars / 8 = N_chars bytes of plane data.
# So target N_chars ~= tier_bytes.
TIERS = {
    "small":  1_000_000,     # ~1M chars -> ~1 MB
    "medium": 10_000_000,    # ~10M chars -> ~10 MB
    "large":  100_000_000,   # ~100M chars -> ~100 MB
}

OUTPUT_NAMES = {
    "small":  "gharchive_1m.npz",
    "medium": "gharchive_10m.npz",
    "large":  "gharchive_100m.npz",
}


def read_json_text(json_path: str, max_chars: int) -> str:
    """Read JSON text, supporting .gz compressed files."""
    if json_path.endswith(".gz"):
        with gzip.open(json_path, "rt", encoding="utf-8") as f:
            return f.read(max_chars)
    else:
        with open(json_path, encoding="utf-8") as f:
            return f.read(max_chars)


def text_to_basis_npz(text: str, npz_path: str, source: str):
    """Convert text to 8 basis bit-plane .npz (fast numpy path)."""
    n_bytes = (len(text) + 7) // 8
    codes = np.frombuffer(text.encode("ascii", errors="replace"),
                          dtype=np.uint8) & 0x7F
    arrays = {}
    for k in range(8):
        bits = ((codes >> k) & 1).astype(np.uint8)
        packed = np.packbits(bits, bitorder="little")
        arrays[f"b{k}"] = packed[:n_bytes]
    arrays["text_length"] = np.array([len(text)], dtype=np.int64)
    arrays["source"] = np.array(source)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(npz_path, **arrays)


def generate_tier(tier: str):
    """Generate one tier of mison data."""
    max_chars = TIERS[tier]
    out_dir = os.path.join(DIR, tier)
    out_path = os.path.join(out_dir, OUTPUT_NAMES[tier])

    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"  [{tier}] Already exists: {out_path} ({size:,} bytes) -- skipping")
        return

    if not os.path.isfile(RAW_FILE):
        print(f"  [{tier}] SKIP: raw file not found: {RAW_FILE}")
        print(f"         Source: GH Archive (https://www.gharchive.org/)")
        return

    print(f"  [{tier}] Reading up to {max_chars:,} chars from {os.path.basename(RAW_FILE)}...")
    text = read_json_text(RAW_FILE, max_chars)
    print(f"  [{tier}] Got {len(text):,} characters")

    if len(text) < max_chars:
        print(f"  [{tier}] WARNING: only {len(text):,} chars available (target: {max_chars:,})")

    source = f"GH Archive hourly dump, first {len(text):,} chars"
    text_to_basis_npz(text, out_path, source)
    size = os.path.getsize(out_path)
    print(f"  [{tier}] Wrote {out_path} ({size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Generate mison tiered datasets")
    parser.add_argument("--tier", choices=["small", "medium", "large"],
                        help="Generate only this tier (default: all)")
    args = parser.parse_args()

    print("mison: generating tiered datasets from GH Archive JSON")
    print(f"  Source: {RAW_FILE}")
    print()

    tiers = [args.tier] if args.tier else ["small", "medium", "large"]
    for tier in tiers:
        generate_tier(tier)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
