#!/usr/bin/env python3
"""Convert FASTA DNA sequences to Parabix-style basis bitstreams.

Parses FASTA files and produces 4 basis bitstreams {bA, bC, bG, bT}
where bX[i] = 1 if the nucleotide at position i equals X. Handles N
(unknown) bases by setting none of the basis bits.

Usage:
    python3 convert.py genome.fa                         # Print stats
    python3 convert.py genome.fa --output dna/           # Write .bin files
    python3 convert.py genome.fa --npz out.npz           # Write .npz file
    python3 convert.py genome.fa --npz out.npz --source "E. coli K-12"
"""

import os
import sys


BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def parse_fasta(fasta_path: str) -> str:
    """Parse a FASTA file and return the concatenated sequence.

    Args:
        fasta_path: Path to the .fa / .fasta file.

    Returns:
        Uppercase DNA string (may contain N for unknown bases).
    """
    import gzip
    opener = gzip.open if fasta_path.endswith(".gz") else open
    parts = []
    with opener(fasta_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            parts.append(line.upper())
    return "".join(parts)


def dna_to_basis_streams(sequence: str) -> dict[str, int]:
    """Convert a DNA string to 4 basis bitstreams.

    Args:
        sequence: DNA string over {A, C, G, T, N}.

    Returns:
        Dictionary with keys 'bA', 'bC', 'bG', 'bT', each an integer
        bitstream. Bit i is set if sequence[i] matches that base.
    """
    streams = {"bA": 0, "bC": 0, "bG": 0, "bT": 0}
    for i, ch in enumerate(sequence):
        if ch in BASE_MAP:
            key = f"b{ch}"
            streams[key] |= 1 << i
    return streams


def convert_fasta(fasta_path: str, output_dir: str | None = None,
                  npz_path: str | None = None,
                  source: str | None = None) -> dict[str, int] | None:
    """Convert a FASTA file to basis bitstreams.

    Args:
        fasta_path: Path to the FASTA file.
        output_dir: If provided, write raw .bin bitstream files.
        npz_path: If provided, write packed .npz file for benchmark tests.
        source: Optional source description for .npz metadata.

    Returns:
        Dictionary of basis bitstreams (Python ints), or None if npz-only
        fast path was used for large sequences.
    """
    seq = parse_fasta(fasta_path)
    n_count = seq.count("N")
    print(f"  {os.path.basename(fasta_path)}: {len(seq)} bp, "
          f"{n_count} N bases ({n_count/max(len(seq),1)*100:.2f}%)")

    n_bytes = (len(seq) + 7) // 8

    # Fast numpy path: go directly from sequence to packed arrays
    # (avoids building Python big integers for large sequences)
    if npz_path and not output_dir:
        import numpy as np
        codes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        arrays = {}
        for base in "ACGT":
            mask = (codes == ord(base)).view(np.uint8)
            packed = np.packbits(mask, bitorder="little")
            arrays[f"b{base}"] = packed[:n_bytes]
        arrays["sequence_length"] = np.int64(len(seq))
        if source:
            arrays["source"] = np.array(source)
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(npz_path, **arrays)
        print(f"  Wrote {npz_path} ({os.path.getsize(npz_path)} bytes)")
        return None

    # Slow Python path: build big integers (fine for small sequences)
    streams = dna_to_basis_streams(seq)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for name, val in streams.items():
            path = os.path.join(output_dir, f"{name}.bin")
            with open(path, "wb") as f:
                f.write(val.to_bytes(n_bytes, "little"))

    if npz_path:
        import numpy as np
        arrays = {}
        for name, val in streams.items():
            arrays[name] = np.frombuffer(
                val.to_bytes(n_bytes, "little"), dtype=np.uint8)
        arrays["sequence_length"] = np.int64(len(seq))
        if source:
            arrays["source"] = np.array(source)
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(npz_path, **arrays)
        print(f"  Wrote {npz_path} ({os.path.getsize(npz_path)} bytes)")

    return streams


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    out = None
    npz = None
    source_desc = None
    args = sys.argv[1:]
    positional = []
    i = 0
    while i < len(args):
        if args[i] == "--output":
            out = args[i + 1]; i += 2
        elif args[i] == "--npz":
            npz = args[i + 1]; i += 2
        elif args[i] == "--source":
            source_desc = args[i + 1]; i += 2
        else:
            positional.append(args[i]); i += 1
    if not positional:
        print(__doc__)
        sys.exit(1)
    convert_fasta(positional[0], output_dir=out, npz_path=npz,
                  source=source_desc)
