"""Read/write .bsdata binary format for precomputed test data.

Format: UTF-8 JSON header + b'\\0' + raw binary data.

The JSON header contains metadata (bitlength, params) and references to binary
sections via special markers:
  {"_bin": [offset, nbytes]}       — single stream (little-endian packed bits)
  {"_array_bin": [count, offset, stride]} — array of count streams
  {"_broadcast_pack": [count, offset, nbytes]} — bitpacked broadcast array
      (all elements are 0 or -1; bit=1 means -1, bit=0 means 0, LSB-first)

Small values (ints that fit in JSON) are stored inline. Binary sections store
little-endian packed bits with ceil(bitlength/8) bytes per stream.

Multi-case tests use {"cases": [...]} where each entry is a standalone case.
Legacy files with {"layers": [...]} are also supported for backward compat.
"""

from __future__ import annotations

import json
import math


def _stream_to_bytes(value: int, n_bytes: int) -> bytes:
    """Convert a Python int (possibly negative from NOT) to LE bytes."""
    # Mask to n_bytes * 8 bits (handles negative values from NOT and overflow)
    mask = (1 << (n_bytes * 8)) - 1
    return (value & mask).to_bytes(n_bytes, byteorder='little')


def _bytes_to_stream(data: bytes) -> int:
    """Convert LE bytes back to a Python int."""
    return int.from_bytes(data, byteorder='little')


def _should_store_binary(value, bitlength: int) -> bool:
    """Decide if a stream value should go in the binary section."""
    if not isinstance(value, int):
        return False
    if value == 0 or value == -1:
        return False
    # Store in binary if the value is large (more than ~52 bits for JSON safety)
    return value.bit_length() > 52 or value < 0


def _should_store_array_binary(arr: dict | list, bitlength: int) -> bool:
    """Decide if an array should go in the binary section."""
    if isinstance(arr, list):
        values = arr
    elif isinstance(arr, dict):
        values = list(arr.values())
    else:
        return False
    return any(_should_store_binary(v, bitlength) for v in values)


def write_bsdata(path: str, bitlength: int = 0, *,
                 n_vectors: int | None = None,
                 inputs: dict | None = None,
                 params: dict | None = None,
                 input_arrays: dict | None = None,
                 expected: dict | None = None,
                 cases: list[dict] | None = None,
                 iterations: int = 1,
                 feedback: dict | None = None) -> None:
    """Write a .bsdata file.

    For single-case tests, provide inputs/params/input_arrays/expected.
    For multi-case tests, provide cases=[{bitlength, params, ...}, ...].
    iterations/feedback are stored in the JSON header for feedback-loop tests.

    The n_vectors parameter is accepted for backward compatibility but
    bitlength is preferred. If both are provided, bitlength takes precedence.
    """
    # Backward compat: accept n_vectors as alias for bitlength
    if bitlength == 0 and n_vectors is not None:
        bitlength = n_vectors
    binary_parts: list[bytes] = []
    offset = 0

    def alloc(data: bytes) -> int:
        nonlocal offset
        start = offset
        binary_parts.append(data)
        offset += len(data)
        return start

    def _is_broadcast_array(arr_values, min_count=10000):
        """Check if array is large and all elements are 0 or -1 (broadcast)."""
        if len(arr_values) < min_count:
            return False
        return all(v == 0 or v == -1 for v in arr_values)

    def _broadcast_pack_bytes(arr_values, count):
        """Bitpack broadcast indicators: bit=1 if element is -1, bit=0 if 0."""
        nbytes = math.ceil(count / 8)
        packed = bytearray(nbytes)
        for i, v in enumerate(arr_values):
            if v == -1:
                packed[i >> 3] |= 1 << (i & 7)
        return bytes(packed)

    def encode_value(key: str, value, nv: int) -> object:
        """Encode a single value, possibly allocating binary space."""
        if isinstance(value, int):
            if _should_store_binary(value, nv):
                nbytes = math.ceil(nv / 8)
                off = alloc(_stream_to_bytes(value, nbytes))
                return {"_bin": [off, nbytes]}
            return value
        if isinstance(value, dict):
            # Array: {index: value}
            if not value:
                return {}
            count = max(value.keys()) + 1
            # Check for broadcast pack (large all-0/-1 arrays)
            arr_values = [value.get(i, 0) for i in range(count)]
            if _is_broadcast_array(arr_values):
                packed = _broadcast_pack_bytes(arr_values, count)
                nbytes = len(packed)
                off = alloc(packed)
                return {"_broadcast_pack": [count, off, nbytes]}
            if _should_store_array_binary(value, nv):
                nbytes = math.ceil(nv / 8)
                start = offset
                for i in range(count):
                    alloc(_stream_to_bytes(value.get(i, 0), nbytes))
                return {"_array_bin": [count, start, nbytes]}
            # Small array: store inline as list
            return [value.get(i, 0) for i in range(count)]
        if isinstance(value, list):
            # Already a list (from JSON parse)
            # Check for broadcast pack (large all-0/-1 arrays)
            if _is_broadcast_array(value):
                count = len(value)
                packed = _broadcast_pack_bytes(value, count)
                nbytes = len(packed)
                off = alloc(packed)
                return {"_broadcast_pack": [count, off, nbytes]}
            if _should_store_array_binary(value, nv):
                count = len(value)
                nbytes = math.ceil(nv / 8)
                start = offset
                for v in value:
                    alloc(_stream_to_bytes(v if isinstance(v, int) else 0, nbytes))
                return {"_array_bin": [count, start, nbytes]}
            return value
        return value

    def encode_section(section: dict | None, nv: int) -> dict:
        if not section:
            return {}
        return {k: encode_value(k, v, nv) for k, v in section.items()}

    header = {"format": "bsdata", "version": 1}

    if cases is not None:
        encoded_cases = []
        for case in cases:
            lnv = case.get("bitlength", case.get("n_vectors", bitlength))
            encoded = {"bitlength": lnv}
            if "params" in case:
                encoded["params"] = case["params"]
            if "inputs" in case:
                encoded["inputs"] = encode_section(case["inputs"], lnv)
            if "input_arrays" in case:
                encoded["input_arrays"] = encode_section(case["input_arrays"], lnv)
            if "expected" in case:
                encoded["expected"] = encode_section(case["expected"], lnv)
            encoded_cases.append(encoded)
        header["cases"] = encoded_cases
    else:
        header["bitlength"] = bitlength
        if params:
            header["params"] = params
        if inputs is not None:
            header["inputs"] = encode_section(inputs, bitlength)
        if input_arrays is not None:
            header["input_arrays"] = encode_section(input_arrays, bitlength)
        if expected is not None:
            header["expected"] = encode_section(expected, bitlength)
        if iterations > 1:
            header["iterations"] = iterations
        if feedback:
            header["feedback"] = feedback

    header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    with open(path, 'wb') as f:
        f.write(header_bytes)
        f.write(b'\0')
        for part in binary_parts:
            f.write(part)


def read_bsdata(path: str) -> dict:
    """Read a .bsdata file and return a dict compatible with the test runner.

    Returns dict with keys: bitlength, inputs, params, input_arrays, expected.
    For multi-case files, returns dict with key "cases" (list of dicts).
    """
    with open(path, 'rb') as f:
        raw = f.read()

    sep = raw.index(b'\0')
    header = json.loads(raw[:sep].decode('utf-8'))
    binary = raw[sep + 1:]

    def decode_value(v):
        if isinstance(v, dict):
            if "_bin" in v:
                off, nbytes = v["_bin"]
                return _bytes_to_stream(binary[off:off + nbytes])
            if "_broadcast_pack" in v:
                count, off, nbytes = v["_broadcast_pack"]
                result = {}
                for i in range(count):
                    byte_idx = off + (i >> 3)
                    bit_idx = i & 7
                    is_ones = (byte_idx < len(binary) and
                               (binary[byte_idx] >> bit_idx) & 1)
                    result[i] = -1 if is_ones else 0
                return result
            if "_array_bin" in v:
                count, off, stride = v["_array_bin"]
                result = {}
                for i in range(count):
                    start = off + i * stride
                    result[i] = _bytes_to_stream(binary[start:start + stride])
                return result
            # Regular dict (shouldn't normally appear in bsdata)
            return {k: decode_value(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return {i: decode_value(x) for i, x in enumerate(v)}
        return v

    def decode_section(section: dict) -> dict:
        return {k: decode_value(v) for k, v in section.items()}

    # Accept both "cases" and legacy "layers" key
    multi_key = "cases" if "cases" in header else ("layers" if "layers" in header else None)
    if multi_key:
        cases = []
        for entry in header[multi_key]:
            decoded = {"bitlength": entry.get("bitlength", entry.get("n_vectors", 0))}
            if "params" in entry:
                decoded["params"] = entry["params"]
            if "inputs" in entry:
                decoded["inputs"] = decode_section(entry["inputs"])
            if "input_arrays" in entry:
                decoded["input_arrays"] = decode_section(entry["input_arrays"])
            if "expected" in entry:
                decoded["expected"] = decode_section(entry["expected"])
            cases.append(decoded)
        return {"cases": cases}

    result = {"bitlength": header.get("bitlength", header.get("n_vectors", 0))}
    if "params" in header:
        result["params"] = header["params"]
    if "inputs" in header:
        result["inputs"] = decode_section(header["inputs"])
    if "input_arrays" in header:
        result["input_arrays"] = decode_section(header["input_arrays"])
    if "expected" in header:
        result["expected"] = decode_section(header["expected"])
    if "iterations" in header:
        result["iterations"] = header["iterations"]
    if "feedback" in header:
        result["feedback"] = header["feedback"]
    return result


# ── CLI: python -m benchmark.bsdata <file.bsdata> ──


def _format_value(v, max_hex=80):
    """Format a value for display: truncate large hex strings."""
    if isinstance(v, int):
        if v == 0:
            return "0"
        h = hex(v)
        if len(h) > max_hex:
            return f"{h[:40]}...{h[-20:]} ({v.bit_length()} bits)"
        return h
    if isinstance(v, dict):
        # Array: {index: value}
        items = sorted(v.items())
        if len(items) <= 6:
            return "{" + ", ".join(f"{k}: {_format_value(val, 40)}" for k, val in items) + "}"
        shown = items[:3] + items[-2:]
        return ("{" + ", ".join(f"{k}: {_format_value(val, 40)}" for k, val in shown[:3])
                + f", ... ({len(items)} entries), "
                + ", ".join(f"{k}: {_format_value(val, 40)}" for k, val in shown[3:])
                + "}")
    return str(v)


def _dump_section(name, section, indent="  "):
    """Pretty-print a section (inputs, expected, etc.)."""
    if not section:
        return
    print(f"{indent}{name}:")
    for k, v in sorted(section.items()):
        print(f"{indent}  {k}: {_format_value(v)}")


def _dump_bsdata(path):
    """Pretty-print a .bsdata file."""
    data = read_bsdata(path)
    print(f"File: {path}")

    if "cases" in data:
        print(f"Multi-case: {len(data['cases'])} cases")
        for i, case in enumerate(data["cases"]):
            print(f"\n  Case {i}:")
            print(f"    bitlength: {case.get('bitlength', case.get('n_vectors', 0))}")
            if "params" in case:
                print(f"    params: {case['params']}")
            _dump_section("inputs", case.get("inputs", {}), "    ")
            _dump_section("input_arrays", case.get("input_arrays", {}), "    ")
            _dump_section("expected", case.get("expected", {}), "    ")
            if case.get("iterations", 1) > 1:
                print(f"    iterations: {case['iterations']}")
            if "feedback" in case:
                print(f"    feedback: {case['feedback']}")
    else:
        print(f"bitlength: {data.get('bitlength', data.get('n_vectors', 0))}")
        if "params" in data:
            print(f"params: {data['params']}")
        _dump_section("inputs", data.get("inputs", {}))
        _dump_section("input_arrays", data.get("input_arrays", {}))
        _dump_section("expected", data.get("expected", {}))
        if data.get("iterations", 1) > 1:
            print(f"iterations: {data['iterations']}")
        if "feedback" in data:
            print(f"feedback: {data['feedback']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m benchmark.bsdata <file.bsdata> [file2.bsdata ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        _dump_bsdata(path)
        if len(sys.argv) > 2:
            print()
