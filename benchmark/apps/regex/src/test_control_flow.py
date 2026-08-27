#!/usr/bin/env python3
"""Regression tests for Pablo if/while lowering in pablo_to_bs.py.

The converter used to drop every Pablo `if ... {` and `}` line and inline the
body.  For `If` that is semantics-preserving, because icgrep's If is a
scheduling decision: when the test stream is zero the block's results are zero
anyway.  For `While` it was wrong.  The `}` skip swallowed the loop's closing
brace, so a fixed-point loop was reduced to a single iteration and neither
errored nor looped.

The shipped payloads do not happen to contain a substring that forces two or
more iterations, so the suite's `REF subset of BS` spot-check passed anyway.
These tests force that case directly, so the bug cannot come back unnoticed.

Requires icgrep (ICGREP_BIN).  Skips when it is unavailable.

Usage:
    python benchmark/apps/regex/src/test_control_flow.py
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from benchmark.apps.regex.src.pablo_to_bs import (
    PabloToBs, compile_pattern_via_icgrep, run_icgrep,
)
from simulator.pythonsim import interpret, parse, validate_3addr


# A star over a multi-byte body cannot use MatchStar, so icgrep compiles it to
# a Pablo fixed-point `while`.  Each extra `ab` needs one more iteration, which
# is what single-iteration flattening got wrong.
WHILE_PATTERN = "x(ab)*y"
WHILE_CASES = ["xy", "xaby", "xababy", "xabababy", "xababababy", "zzz"]

# A trailing literal after a `+` puts the tail in a Pablo `if` block.
IF_PATTERN = "[0-9]+abc"
IF_CASES = ["1abc", "12abc", "123abc", "xabc", "abc"]


def _build(pattern: str) -> tuple[str, str]:
    """Compile one pattern into a standalone .bs program."""
    bs_lines, output_var = compile_pattern_via_icgrep("t", pattern, "")
    assert bs_lines and output_var, f"failed to compile /{pattern}/"
    source = ("input stream b[8]\noutput stream m\n"
              + "\n".join(bs_lines) + f"\nm = {output_var}\n")
    return source, output_var


def _run(source: str, text: str) -> int:
    """Run a program over `text` transposed into basis bit-planes."""
    basis = {i: 0 for i in range(8)}
    for pos, ch in enumerate(text):
        for bit in range(8):
            if (ord(ch) >> bit) & 1:
                basis[bit] |= 1 << pos
    return interpret(source, {}, None, {"b": basis})["m"]


def test_while_is_emitted_and_iterates() -> list[str]:
    """A fixed-point loop must survive lowering and run to convergence."""
    failures = []
    source, _ = _build(WHILE_PATTERN)

    if not re.search(r"^\s*while \(", source, re.M):
        failures.append(
            f"  /{WHILE_PATTERN}/: no `while` block in the generated .bs; "
            f"the loop was flattened")

    for text in WHILE_CASES:
        got = bool(_run(source, text))
        want = bool(re.search(WHILE_PATTERN, text))
        if got != want:
            failures.append(
                f"  /{WHILE_PATTERN}/ on {text!r}: got {got}, want {want}")
    return failures


def test_if_is_emitted() -> list[str]:
    """An `if` block must survive lowering and not change the match set."""
    failures = []
    source, _ = _build(IF_PATTERN)

    if not re.search(r"^\s*if \(", source, re.M):
        failures.append(
            f"  /{IF_PATTERN}/: no `if` block in the generated .bs")

    for text in IF_CASES:
        got = bool(_run(source, text))
        want = bool(re.search(IF_PATTERN, text))
        if got != want:
            failures.append(
                f"  /{IF_PATTERN}/ on {text!r}: got {got}, want {want}")
    return failures


def test_blocks_are_three_address() -> list[str]:
    """Generated control flow must satisfy the DSL's 3-address rules.

    `if`/`while` conditions have to be plain variables, which is what Pablo
    emits, so the validator is the check that lowering preserved that.
    """
    failures = []
    for pattern in (WHILE_PATTERN, IF_PATTERN):
        source, _ = _build(pattern)
        try:
            validate_3addr(parse(source))
        except Exception as e:
            failures.append(f"  /{pattern}/: not 3-address: {e}")
    return failures


def test_nesting_matches_pablo() -> list[str]:
    """Block nesting must reproduce the Pablo source, and braces must balance."""
    failures = []
    for pattern in (WHILE_PATTERN, IF_PATTERN):
        kernel = PabloToBs.extract_regex_kernel(run_icgrep(pattern, ""))
        pablo_max = depth = 0
        for line in kernel.split("\n"):
            line = line.strip()
            if re.match(r"(if|while)\s+\S+\s*\{$", line):
                depth += 1
                pablo_max = max(pablo_max, depth)
            elif line == "}":
                depth -= 1

        source, _ = _build(pattern)
        bs_max = depth = 0
        for line in source.split("\n"):
            line = line.strip()
            if re.match(r"(if|while)\s*\(.*\)\s*\{$", line):
                depth += 1
                bs_max = max(bs_max, depth)
            elif line == "}":
                depth -= 1

        if depth != 0:
            failures.append(f"  /{pattern}/: unbalanced braces (depth {depth})")
        if bs_max != pablo_max:
            failures.append(
                f"  /{pattern}/: nesting {bs_max} != Pablo nesting {pablo_max}")
    return failures


TESTS = [
    test_while_is_emitted_and_iterates,
    test_if_is_emitted,
    test_blocks_are_three_address,
    test_nesting_matches_pablo,
]


def main() -> int:
    try:
        run_icgrep("a", "")
    except RuntimeError as e:
        print(f"SKIP: icgrep unavailable ({e})")
        return 0

    print(f"Pablo control-flow lowering tests ({len(TESTS)})")
    print("-" * 60)
    all_failures = []
    n_pass = 0
    for test in TESTS:
        try:
            failures = test()
        except Exception as e:
            failures = [f"  {test.__name__}: EXCEPTION: {e}"]
        if failures:
            print(f"  FAIL  {test.__name__}")
            all_failures.extend(failures)
        else:
            print(f"  PASS  {test.__name__}")
            n_pass += 1
    print("-" * 60)
    print(f"{n_pass}/{len(TESTS)} passed")
    if all_failures:
        print("\nFailures:")
        for f in all_failures:
            print(f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
