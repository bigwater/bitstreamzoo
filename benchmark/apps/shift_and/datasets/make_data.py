#!/usr/bin/env python3
"""Generate Human chr1 (hg38) prefix .npz datasets for shift_and.

These feed the OPTIONAL file-backed tests (Human chr1 2M/20M/200M
GATTACA). They are distinct from the canonical benchmark tiers, which
are synthetic 80M/800M/8B-base .bsdata defined in benchmark/tier_config.py
and produced by datasets/tests/generate_tests.py --tier. The small/
medium/large labels below name only these chr1 prefix sizes.

Input format: 4 basis streams (bA, bC, bG, bT) from DNA sequences.
Effective data = 4 * N_bases / 8 bytes.

Prefix sizes (Human chr1 .npz):
  small:  ~2M bases   (~1 MB)
  medium: ~20M bases  (~10 MB)
  large:  ~200M bases (~100 MB)

Source: Human chr1 (hg38) from UCSC Genome Browser.
        File: chr1.fa.gz (or gunzip'd chr1.fa) in raw/ (249M bases total).

Usage:
    python make_data.py                   # Generate all sizes
    python make_data.py --tier small      # Generate one size
"""

import argparse
import gzip
import os
import sys

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DIR, "raw")
# Accept either the gzipped UCSC download (chr1.fa.gz) or a gunzip'd
# chr1.fa. raw/readme.md documents `gunzip chr1.fa.gz`, which leaves
# chr1.fa; parse_fasta_truncated() handles both transparently.
RAW_CANDIDATES = ("chr1.fa.gz", "chr1.fa")


def resolve_raw_file() -> str:
    """First existing raw chr1 FASTA, or the .gz path for messages."""
    for name in RAW_CANDIDATES:
        path = os.path.join(RAW_DIR, name)
        if os.path.isfile(path):
            return path
    return os.path.join(RAW_DIR, RAW_CANDIDATES[0])

TIERS = {
    "small":  2_000_000,    # ~2M bases -> ~1 MB
    "medium": 20_000_000,   # ~20M bases -> ~10 MB
    "large":  200_000_000,  # ~200M bases -> ~100 MB
}

OUTPUT_NAMES = {
    "small":  "hg38_chr1_2m.npz",
    "medium": "hg38_chr1_20m.npz",
    "large":  "hg38_chr1_200m.npz",
}


def parse_fasta_truncated(fasta_path: str, max_bases: int) -> str:
    """Parse a FASTA file and return up to max_bases of sequence."""
    opener = gzip.open if fasta_path.endswith(".gz") else open
    parts = []
    total = 0
    with opener(fasta_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            chunk = line.upper()
            remaining = max_bases - total
            if len(chunk) >= remaining:
                parts.append(chunk[:remaining])
                total += remaining
                break
            parts.append(chunk)
            total += len(chunk)
    return "".join(parts)


def dna_to_npz(sequence: str, npz_path: str, source: str):
    """Convert DNA sequence to basis bitstream .npz (fast numpy path)."""
    n_bytes = (len(sequence) + 7) // 8
    codes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    arrays = {}
    for base in "ACGT":
        mask = (codes == ord(base)).view(np.uint8)
        packed = np.packbits(mask, bitorder="little")
        arrays[f"b{base}"] = packed[:n_bytes]
    arrays["sequence_length"] = np.int64(len(sequence))
    arrays["source"] = np.array(source)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(npz_path, **arrays)


def generate_tier(tier: str):
    """Generate one tier of shift_and data."""
    max_bases = TIERS[tier]
    out_dir = os.path.join(DIR, tier)
    out_path = os.path.join(out_dir, OUTPUT_NAMES[tier])

    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"  [{tier}] Already exists: {out_path} ({size:,} bytes) -- skipping")
        return

    raw_file = resolve_raw_file()
    if not os.path.isfile(raw_file):
        print(f"  [{tier}] SKIP: raw file not found in {RAW_DIR} (chr1.fa.gz or chr1.fa)")
        print(f"         Download from UCSC: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz")
        return

    print(f"  [{tier}] Parsing {max_bases:,} bases from {os.path.basename(raw_file)}...")
    seq = parse_fasta_truncated(raw_file, max_bases)
    print(f"  [{tier}] Got {len(seq):,} bases")

    source = f"Human chr1 (hg38), first {len(seq):,} bases"
    dna_to_npz(seq, out_path, source)
    size = os.path.getsize(out_path)
    print(f"  [{tier}] Wrote {out_path} ({size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Generate shift_and tiered datasets")
    parser.add_argument("--tier", choices=["small", "medium", "large"],
                        help="Generate only this tier (default: all)")
    args = parser.parse_args()

    print("shift_and: generating Human chr1 (hg38) prefix datasets")
    print(f"  Source: {resolve_raw_file()}")
    print()

    tiers = [args.tier] if args.tier else ["small", "medium", "large"]
    for tier in tiers:
        generate_tier(tier)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
