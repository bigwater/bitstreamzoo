"""Shared test utilities for generate_tests.py files."""


def to_json_val(v, mask=None):
    """Convert value to JSON-safe format. Large ints -> hex strings.

    Dicts with int keys -> dense lists.
    Optional mask for NOT results (handles negative/unbounded Python ints).
    """
    if isinstance(v, dict):
        max_idx = max(v.keys()) if v else -1
        return [to_json_val(v.get(i, 0), mask) for i in range(max_idx + 1)]
    if isinstance(v, int) and mask is not None:
        v = v & mask
    if isinstance(v, int) and v >= 0 and v.bit_length() > 52:
        return hex(v)
    return v
