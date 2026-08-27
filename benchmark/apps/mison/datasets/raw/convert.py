#!/usr/bin/env python3
"""Convert JSON document files to character-basis bitstreams for Mison.

Reads JSON text and decomposes each character into 8 ASCII basis bitstreams
(b0..b7, Parabix-style bit planes). These bit planes are the input to the
Mison/simdjson structural index pipeline.

Usage:
    python3 convert.py events.json                         # Print stats
    python3 convert.py events.json --output out/           # Write .bin files
    python3 convert.py events.json --npz out.npz           # Write .npz file
    python3 convert.py events.json --npz out.npz --source "GH Archive"
"""

import os
import sys
import gzip


def read_json_text(json_path: str, max_chars: int = 1_000_000) -> str:
    """Read JSON text, supporting .gz compressed files.

    Args:
        json_path: Path to .json or .json.gz file.
        max_chars: Maximum characters to read.

    Returns:
        JSON text string.
    """
    if json_path.endswith(".gz"):
        with gzip.open(json_path, "rt", encoding="utf-8") as f:
            return f.read(max_chars)
    else:
        with open(json_path, encoding="utf-8") as f:
            return f.read(max_chars)


def text_to_basis_streams(text: str) -> dict[str, int]:
    """Convert text to 8 ASCII basis bitstreams (Parabix bit planes).

    Args:
        text: Input text string (ASCII characters).

    Returns:
        Dictionary with keys 'b0'..'b7', each an integer bitstream.
        b_k bit i = bit k of ord(text[i]).
    """
    streams = {f"b{k}": 0 for k in range(8)}
    for i, ch in enumerate(text):
        code = ord(ch) & 0x7F  # ASCII only
        for k in range(8):
            if (code >> k) & 1:
                streams[f"b{k}"] |= 1 << i
    return streams


def convert_json(json_path: str, output_dir: str | None = None,
                 max_chars: int = 1_000_000,
                 npz_path: str | None = None,
                 source: str | None = None) -> dict[str, int] | None:
    """Convert a JSON file to basis bitstreams.

    Args:
        json_path: Path to the JSON file.
        output_dir: If provided, write raw .bin bitstream files.
        max_chars: Maximum characters to process.
        npz_path: If provided, write packed .npz file for benchmark tests.
        source: Optional source description for .npz metadata.

    Returns:
        Dictionary of basis bitstreams (Python ints), or None if npz-only
        fast path was used for large texts.
    """
    text = read_json_text(json_path, max_chars)
    print(f"  {os.path.basename(json_path)}: {len(text)} characters")

    # Character class breakdown
    n_struct = sum(1 for c in text if c in '{}[]:,')
    n_quote = text.count('"')
    n_escape = text.count('\\')
    print(f"    Structural: {n_struct}, Quotes: {n_quote}, "
          f"Backslashes: {n_escape}")

    n_bytes = (len(text) + 7) // 8

    # Fast numpy path: go directly from text to packed arrays
    # (avoids building Python big integers for large texts)
    if npz_path and not output_dir:
        import numpy as np
        codes = np.frombuffer(text.encode("ascii", errors="replace"),
                              dtype=np.uint8) & 0x7F
        arrays = {}
        for k in range(8):
            bits = ((codes >> k) & 1).astype(np.uint8)
            packed = np.packbits(bits, bitorder="little")
            arrays[f"b{k}"] = packed[:n_bytes]
        arrays["text_length"] = np.array([len(text)], dtype=np.int64)
        if source:
            arrays["source"] = np.array(source)
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(npz_path, **arrays)
        print(f"  Wrote {npz_path} ({os.path.getsize(npz_path)} bytes)")
        return None

    # Slow Python path: build big integers (fine for small texts)
    streams = text_to_basis_streams(text)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for name, val in sorted(streams.items()):
            path = os.path.join(output_dir, f"{name}.bin")
            with open(path, "wb") as f:
                f.write(val.to_bytes(n_bytes, "little"))

    if npz_path:
        import numpy as np
        arrays = {}
        for name, val in sorted(streams.items()):
            arrays[name] = np.frombuffer(
                val.to_bytes(n_bytes, "little"), dtype=np.uint8)
        arrays["text_length"] = np.array([len(text)], dtype=np.int64)
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
    max_c = 1_000_000
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
        elif args[i] == "--max-chars":
            max_c = int(args[i + 1]); i += 2
        else:
            positional.append(args[i]); i += 1
    if not positional:
        print(__doc__)
        sys.exit(1)
    convert_json(positional[0], output_dir=out, max_chars=max_c,
                 npz_path=npz, source=source_desc)
