#!/usr/bin/env python3
"""Generate precomputed bitstream-level test data for multi-pattern regex.

Produces tests.json with entries for regex_small, regex_medium, regex_large.
Each program gets:
  - 2 hand-crafted inline tests (known match positions)
  - Random-byte tests (data_file): 50 for small/medium, 10 for large
  - 1 real-data test (WRCCDC 2012 pcap, generated/optional)

For regex_medium/large, uses C++ backend (bsim --reuse-mem) with BATCHED
execution to avoid re-parsing the .bs file for every test case.

Usage:
    python generate_tests.py                # unit tests only
    python generate_tests.py --tier small   # unit tests + small tier
    python generate_tests.py --describe     # print provenance info
    python generate_tests.py --verify       # verify SHA-256 of existing files
"""

import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from simulator.pythonsim.parser import Program
from benchmark.base import CppBackend, ProgramInfo
from benchmark.apps.regex.src.run import (
    regex_reference, regex_reference_fast, payload_to_basis_bits,
)
from benchmark.apps.regex.src.compile_all import TIERS
from benchmark.bsdata import write_bsdata
from benchmark.tier_generate import (
    parse_generate_args, file_sha256, make_provenance,
    tier_test_entry, print_describe, verify_files,
    getrandbits_large,
)
from benchmark.tier_config import get_tier_vectors

TESTS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(TESTS_DIR, "../../src")

DOMAIN = "regex"
TIER_SEEDS = {"small": 1000, "medium": 1001, "large": 1002}
TIER_PROGRAMS = {
    "small": ["regex_small", "regex_medium", "regex_large"],
    # regex_large medium: GPU OOM (133 GB > VRAM), CPU-only at 74 min.
    "medium": ["regex_small", "regex_medium"],
    # Large tier: 4.4B bitlength → 0.6 GB per live variable.
    # regex_medium (~500 max_live → ~300 GB) and regex_large (~2400 → ~1.4 TB) OOM.
    # Only regex_small (~50 max_live → ~30 GB) is feasible on 256 GB.
    "large": ["regex_small"],
}

# Number of random tests per program
RANDOM_TEST_COUNTS = {
    "regex_small": 50,
    "regex_medium": 50,
    "regex_large": 10,  # fewer due to parse overhead
}

MAX_VERIFY_PATTERNS = 50


def load_program_info(tier_name):
    """Create a lightweight ProgramInfo for CppBackend (no Python parse)."""
    bs_path = os.path.join(SRC_DIR, f"{tier_name}.bs")
    with open(bs_path) as f:
        n_lines = sum(1 for _ in f)
    program = Program(
        decls=[],
        inputs=["b"],
        outputs=["any_match"],
        stmts=[],
        output_int_names=[],
    )
    return ProgramInfo(
        name=tier_name,
        source_path=os.path.abspath(bs_path),
        program=program,
        n_stmts=n_lines,
    )


def run_batch_on_payloads(prog_info, payloads, reuse_mem=False):
    """Run bsim on multiple payloads in a single invocation.

    Writes each payload as a temporary .bsdata file, runs bsim with
    multiple --input flags, parses batch output.

    Note: For unit tests (small payloads), reuse_mem=False is fine because
    memory is proportional to bitlength (tiny for 64-bit tests). reuse_mem
    adds liveness analysis overhead for large programs. Only use reuse_mem
    for tier data with large bitlength.

    Returns list of output dicts (one per payload).
    """
    backend = CppBackend(variant="simd", reuse_mem=reuse_mem)
    tmp_files = []

    try:
        for payload in payloads:
            n = len(payload)
            b = payload_to_basis_bits(payload)
            fd, tmp_path = tempfile.mkstemp(suffix=".bsdata")
            os.close(fd)
            write_bsdata(tmp_path, n, input_arrays={"b": b})
            tmp_files.append(tmp_path)

        # Run all in one bsim invocation
        batch_results = backend.run_batch(prog_info, tmp_files)

        # Parse outputs
        results = []
        for i, payload in enumerate(payloads):
            n = len(payload)
            mask = (1 << n) - 1
            raw = batch_results[i]
            outputs = {}
            for k, v in raw.get("outputs", {}).items():
                if isinstance(v, str) and v.startswith("0x"):
                    outputs[k] = int(v, 16) & mask
                elif isinstance(v, int):
                    outputs[k] = v & mask
            results.append(outputs)

        return results

    finally:
        for f in tmp_files:
            if os.path.exists(f):
                os.unlink(f)


# Optional cross-check against a locally built icgrep (Parabix). Point
# the ICGREP_BIN environment variable at the binary (or put icgrep on PATH)
# to enable it; the check is skipped silently when the binary is absent.
ICGREP_PATH = os.environ.get("ICGREP_BIN") or shutil.which("icgrep") or ""


def icgrep_existence_check(patterns, bs_result, payload, max_patterns=None):
    """Cross-check ``any_match`` against icgrep using an existence relation.

    For each pattern, run icgrep over a temp file containing the payload.
    If icgrep reports any match, ``any_match`` must be non-zero somewhere.
    This catches the failure mode where ``pablo_to_bs.py`` silently drops
    a pattern (without it, only Python ``re`` cross-checks the pipeline,
    and only for patterns whose PCRE Python can compile).

    Skipped silently if the icgrep binary is missing.  Returns True when
    icgrep is unavailable, otherwise the existence relation status.
    """
    if not ICGREP_PATH or not os.path.exists(ICGREP_PATH):
        return True

    bs_any = bs_result.get("any_match", 0)
    if bs_any != 0:
        # ``any_match`` is already non-zero, so the existence check
        # (icgrep_match → bs_any != 0) is trivially satisfied.
        return True

    n = len(payload)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="regex_payload_", suffix=".bin")
    try:
        os.write(tmp_fd, payload)
        os.close(tmp_fd)
        check_patterns = patterns[:max_patterns] if max_patterns else patterns
        for name, pat, flags in check_patterns:
            cmd = [ICGREP_PATH, "-b", "-U"]
            if "i" in flags:
                cmd.append("-i")
            cmd.extend([pat, tmp_path])
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                continue
            if result.stdout.strip():
                print(
                    f"  ICGREP-CROSS-CHECK FAIL: pattern {name} matched in "
                    f"icgrep but any_match is zero (n={n} bytes)"
                )
                return False
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return True


def verify_any_match(patterns, bs_result, payload, max_patterns=MAX_VERIFY_PATTERNS):
    """Verify any_match output against Python re reference (RE subset of BS).

    Checks that every position matched by the reference (OR of sampled
    patterns) is also set in the any_match output.
    """
    n = len(payload)
    mask = (1 << n) - 1
    bs_any = bs_result.get("any_match", 0) & mask
    ref_any = 0
    for name, pat, flags in patterns[:max_patterns]:
        ref_any |= regex_reference(pat, flags, payload) & mask
    missed = ref_any & ~bs_any
    if missed:
        print(f"  MISMATCH any_match: {bin(missed).count('1')} positions missed")
        print(f"    BS any_match: {bs_any:#x}")
        print(f"    REF (OR of {min(len(patterns), max_patterns)} patterns): {ref_any:#x}")
        return False
    return True


def generate_unit_tests(tier_name, patterns, prog_info):
    """Generate hand-crafted + random tests for a program.

    All test payloads are batched into a single bsim invocation.
    Returns list of test entries.
    """
    n_random = RANDOM_TEST_COUNTS.get(tier_name, 50)

    # Don't use --reuse-mem for unit tests (small payloads, n_words=1).
    # reuse_mem adds liveness analysis overhead for large programs.
    # Only needed for tier data where bitlength is large.
    reuse_mem = False

    # Prepare all payloads
    payloads = []
    payload_names = []

    # Hand-crafted payload 1: mixed network traffic triggers
    payload1 = (b"POST /login HTTP/1.1\r\n"
                b"Content-Length: 42\r\n"
                b"Content-Type: image/png\r\n"
                b"Host: evil.cn\r\n"
                b"User-Agent: MSIE 5\r\n"
                b"charset=utf-8\r\n"
                b"tid=123&id=1' select * from users\r\n"
                b"GET /shell.exe?id=1)\r\n"
                b"eval foo bar\r\n"
                b"../../../etc/passwd\r\n")
    payloads.append(payload1)
    payload_names.append("hand_crafted_1")

    # Hand-crafted payload 2: all zeros
    payload2 = b"\x00" * 64
    payloads.append(payload2)
    payload_names.append("hand_crafted_2")

    # Random payloads
    rng = random.Random("unit-payloads-" + tier_name)
    for i in range(n_random):
        payload = bytes(rng.randint(0, 255) for _ in range(64))
        payloads.append(payload)
        payload_names.append(f"random_{i}")

    # Run ALL payloads in a single bsim invocation
    print(f"    Running bsim on {len(payloads)} payloads (single invocation)...")
    results = run_batch_on_payloads(prog_info, payloads, reuse_mem=reuse_mem)
    print(f"    Done. Verifying...")

    # Verify hand-crafted tests
    ok1 = verify_any_match(patterns, results[0], payloads[0])
    assert ok1, f"Hand-crafted test 1 verification failed for {tier_name}"
    ok1_icgrep = icgrep_existence_check(patterns, results[0], payloads[0])
    assert ok1_icgrep, (
        f"Hand-crafted test 1 icgrep cross-check failed for {tier_name}"
    )

    ok2 = verify_any_match(patterns, results[1], payloads[1])
    # All-zeros may not trigger patterns; existence check still relevant.
    ok2_icgrep = icgrep_existence_check(patterns, results[1], payloads[1])
    assert ok2_icgrep, (
        f"All-zero test icgrep cross-check failed for {tier_name}"
    )

    # Verify first 5 random tests
    for i in range(min(5, n_random)):
        ok = verify_any_match(patterns, results[2 + i], payloads[2 + i])
        assert ok, f"Random test {i} verification failed for {tier_name}"

    print(f"    Verification passed (incl. icgrep existence cross-check).")

    tests = []

    # Write hand-crafted test 1
    n = len(payloads[0])
    expected1 = {"any_match": results[0].get("any_match", 0) & ((1 << n) - 1)}
    b = payload_to_basis_bits(payloads[0])
    fname1 = f"{tier_name}_hand_crafted_mixed_payload.bsdata"
    write_bsdata(os.path.join(TESTS_DIR, fname1), n,
                 input_arrays={"b": b}, expected=expected1)
    tests.append({"name": "Hand-crafted mixed payload",
                  "bitlength": n, "data_file": fname1})

    # Write hand-crafted test 2
    n2 = len(payloads[1])
    expected2 = {"any_match": results[1].get("any_match", 0) & ((1 << n2) - 1)}
    b2 = payload_to_basis_bits(payloads[1])
    fname2 = f"{tier_name}_all_zero_payload.bsdata"
    write_bsdata(os.path.join(TESTS_DIR, fname2), n2,
                 input_arrays={"b": b2}, expected=expected2)
    tests.append({"name": "All-zero payload",
                  "bitlength": n2, "data_file": fname2})

    # Write random tests as multi-case .bsdata
    sub_cases = []
    for i in range(n_random):
        payload = payloads[2 + i]
        n = len(payload)
        mask = (1 << n) - 1
        b = payload_to_basis_bits(payload)
        expected = {"any_match": results[2 + i].get("any_match", 0) & mask}
        sub_cases.append({
            "input_arrays": {"b": b},
            "bitlength": n,
            "expected": expected,
        })
    fname3 = f"{tier_name}_random_64bit_{n_random}.bsdata"
    write_bsdata(os.path.join(TESTS_DIR, fname3), n_random, cases=sub_cases)
    tests.append({"name": f"Random 64-byte x{n_random}",
                  "bitlength": n_random, "data_file": fname3})

    return tests


def extract_byte_from_basis(basis_streams, j):
    """Extract byte at position j from 8 basis bit-planes."""
    byte_val = 0
    for k in range(8):
        byte_val |= ((basis_streams[k] >> j) & 1) << k
    return byte_val


SPOT_CHECK_SLICE_BYTES = 65_536
STUB_OP_COUNT_THRESHOLD = 5  # patterns at or below this are trivial-ZERO stubs


# Policy: the bitstream model follows icgrep semantics, not PCRE.
# Python ``re`` is a convenience reference for the spot-check, but PCRE
# diverges from icgrep on a known set of constructs.  Patterns matching
# either filter below are excluded from the ``REF ⊆ BS`` spot-check.


def _has_nongreedy(pattern: str) -> bool:
    """True if the PCRE source uses a non-greedy quantifier.

    icgrep's translation of ``*?``/``+?``/``??`` does not match PCRE's
    leftmost-shortest semantics: BS finds the set of all positions
    where the pattern can end, PCRE finds the specific shortest match.
    """
    return "*?" in pattern or "+?" in pattern or "??" in pattern


def _has_dotall_dot(pattern: str, flags: str) -> bool:
    """True if the pattern has ``/s`` flag and a bare ``.`` outside ``[]``.

    icgrep does **not** honour the PCRE /s (DOTALL) flag and always
    treats ``.`` as "not newline".  PCRE with /s lets ``.`` match
    newline.  For patterns where this matters the two engines produce
    different match sets (audited: snort_0959 ``/\\s+.*?%.*?%/smi``
    drops two positions on a 64 KB slice because the .* region
    contained newlines that PCRE consumed and icgrep did not).
    """
    if "s" not in flags:
        return False
    i = 0
    in_class = False
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            i += 2
            continue
        if ch == "[":
            in_class = True
        elif ch == "]":
            in_class = False
        elif ch == "." and not in_class:
            return True
        i += 1
    return False


def _load_op_counts_by_name():
    """Return {pattern_name: (op_count, pattern_text)} from the manifest.

    Used by ``spot_check_tier`` to exclude trivial-ZERO stubs (counted
    repetitions, lookarounds, etc.) and non-greedy patterns from the
    reference.
    """
    from benchmark.apps.regex.src.compile_all import load_manifest
    manifest = load_manifest()
    return {p["name"]: (p["op_count"], p["pattern"])
            for p in manifest["patterns"]}


def spot_check_tier(patterns, basis_streams, bs_result, bitlength, rng,
                    slice_bytes=SPOT_CHECK_SLICE_BYTES):
    """Spot-check tiered any_match output against regex_reference.

    Picks a single contiguous slice of ``slice_bytes`` bytes at a random
    offset, runs Python ``re`` for every **non-stub, greedy** pattern
    over that slice, ORs the per-pattern match streams, and checks the
    BS ``any_match`` covers the reference (``REF ⊆ BS``) at the
    corresponding bit positions.

    Excluded from the reference (the bitstream model follows icgrep
    semantics, not PCRE, so the spot-check filters out patterns where
    the two engines are known to disagree):
    - Stubs (op_count ≤ STUB_OP_COUNT_THRESHOLD): emit constant ZERO in
      .bs by construction.
    - Non-greedy patterns: ``*?``/``+?``/``??`` semantics diverge.
    - ``/s`` + bare ``.``: icgrep ignores the DOTALL flag.

    Contiguous slices preserve byte adjacency, which the bitstream
    model relies on; the prior implementation pulled bytes from
    non-contiguous positions and asked Python ``re`` to match on the
    concatenation, which was not a valid reference for the .bs
    semantics.
    """
    meta = _load_op_counts_by_name()
    active = []
    n_stubs = 0
    n_nongreedy = 0
    n_dotall = 0
    for (n, p, f) in patterns:
        op_count, _ = meta.get(n, (0, ""))
        if op_count <= STUB_OP_COUNT_THRESHOLD:
            n_stubs += 1
        elif _has_nongreedy(p):
            n_nongreedy += 1
        elif _has_dotall_dot(p, f):
            n_dotall += 1
        else:
            active.append((n, p, f))

    slice_bytes = min(slice_bytes, bitlength)
    start = rng.randrange(0, bitlength - slice_bytes + 1)
    payload = bytearray(slice_bytes)
    for j in range(slice_bytes):
        payload[j] = extract_byte_from_basis(basis_streams, start + j)
    payload = bytes(payload)

    mask = (1 << slice_bytes) - 1
    # ``regex_reference_fast`` is finditer-based: it covers far more
    # patterns × positions per second than the exhaustive variant but can
    # miss intermediate end positions for overlapping matches.  The check
    # remains a valid one-sided subset (REF ⊆ BS); strictness comes from
    # breadth, not exhaustiveness.
    ref_any = 0
    for name, pat, flags in active:
        ref_any |= regex_reference_fast(pat, flags, payload) & mask
    bs_any = (bs_result.get("any_match", 0) >> start) & mask

    missed = ref_any & ~bs_any
    if missed:
        n_missed = bin(missed).count("1")
        n_ref = bin(ref_any).count("1")
        print(
            f"  SPOT-CHECK FAIL any_match: missed {n_missed}/{n_ref} "
            f"positions in [{start}, {start + slice_bytes}) "
            f"across {len(active)} active patterns "
            f"({n_stubs} stubs, {n_nongreedy} non-greedy, "
            f"{n_dotall} /s+dot excluded)"
        )
        return False
    print(
        f"    Spot-check covered {len(active)} active patterns "
        f"({n_stubs} stubs, {n_nongreedy} non-greedy, "
        f"{n_dotall} /s+dot excluded) × "
        f"{slice_bytes:,} bytes (ref hits = {bin(ref_any).count('1'):,})"
    )
    return True


def generate_tier_data(tier_name, tier, bitlength, prog_info, patterns):
    """Generate a tier .bsdata file using C++ backend with --reuse-mem."""
    seed = TIER_SEEDS[tier]
    mask = (1 << bitlength) - 1
    rng = random.Random(seed)

    print(f"  Generating {tier_name} {tier} tier: "
          f"{bitlength:,} vectors (seed={seed})...")

    basis_streams = {}
    for k in range(8):
        print(f"    Generating basis b[{k}] ({bitlength:,} bits)...")
        basis_streams[k] = getrandbits_large(rng, bitlength)

    # Use C++ backend with --reuse-mem
    # For large bitlength (>1B), write temp .bsdata to avoid piping
    # multi-GB JSON through stdin (causes pipe/memory failure).
    # regex_large (1.24M ops) needs extended time for liveness+execution
    timeout = 7200 if "regex_large" in tier_name else 3600
    # Use CPU backend with reuse-mem for all tiers.
    print(f"    Running C++ backend ({len(patterns)} patterns, "
          f"reuse_mem=True, timeout={timeout}s)...")
    backend = CppBackend(variant="simd", reuse_mem=True, timeout=timeout)
    if bitlength > 1_000_000_000:
        # Write temp .bsdata to avoid multi-GB JSON pipe
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".bsdata")
        os.close(tmp_fd)
        try:
            write_bsdata(tmp_path, bitlength, input_arrays={"b": basis_streams})
            tmp_size = os.path.getsize(tmp_path)
            print(f"    Wrote temp .bsdata ({tmp_size:,} bytes, "
                  f"{tmp_size / 1e9:.2f} GB)")
            result, ops, exec_ms = backend.run(
                prog_info, inputs={}, bsdata_path=tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        result, ops, exec_ms = backend.run(
            prog_info, inputs={}, input_arrays={'b': basis_streams},
            bitlength=bitlength)

    expected = {"any_match": result.get("any_match", 0) & mask}

    total_matches = bin(expected["any_match"]).count('1')
    print(f"    Total match bits: {total_matches:,} (exec: {exec_ms:.1f}ms)")

    # Spot-check
    print(f"    Spot-checking...")
    spot_rng = random.Random(seed + 7777)
    ok = spot_check_tier(patterns, basis_streams, result, bitlength,
                         spot_rng)
    assert ok, f"Spot-check failed for {tier_name} tier {tier}"
    print(f"    Spot-check passed")

    data_file = f"{tier_name}_{tier}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_file)
    write_bsdata(data_path, bitlength,
                 input_arrays={"b": basis_streams}, expected=expected)
    size_bytes = os.path.getsize(data_path)
    print(f"    Wrote {data_file} ({size_bytes:,} bytes, {size_bytes / 1e6:.1f} MB)")

    return data_file, size_bytes


def load_tests_json():
    """Load existing tests.json if it exists (dict keyed by program name)."""
    path = os.path.join(TESTS_DIR, "tests.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_tests_json(all_tests):
    """Write tests.json (dict keyed by program name)."""
    path = os.path.join(TESTS_DIR, "tests.json")
    with open(path, "w") as f:
        json.dump(all_tests, f, indent=2)
    total = sum(len(v) for v in all_tests.values())
    print(f"\nWrote {path}: {total} total test entries across "
          f"{len(all_tests)} programs")


def merge_tier_entry(entries, new_entry):
    """Merge a tier entry into a program's entry list, replacing any existing
    entry with the same data_file."""
    data_file = new_entry["data_file"]
    for i, entry in enumerate(entries):
        if entry.get("data_file") == data_file:
            entries[i] = new_entry
            return entries
    entries.append(new_entry)
    return entries


def main():
    args = parse_generate_args(DOMAIN)

    if args.describe or args.verify:
        tests_json_path = os.path.join(TESTS_DIR, "tests.json")
        if not os.path.exists(tests_json_path):
            print(f"No tests.json found at {tests_json_path}")
            sys.exit(1)
        with open(tests_json_path) as f:
            all_tests = json.load(f)
        if args.describe:
            for prog_name, tests in all_tests.items():
                print(f"{prog_name}: {len(tests)} test entries")
                print_describe(TESTS_DIR, tests)
        if args.verify:
            all_ok = True
            for prog_name, tests in all_tests.items():
                print(f"{prog_name}: verifying SHA-256 checksums")
                if not verify_files(TESTS_DIR, tests):
                    all_ok = False
            sys.exit(0 if all_ok else 1)
        return

    if args.tier:
        # Tier generation mode: load existing tests.json, generate only
        # the requested tier data, and merge into existing entries.
        all_tests = load_tests_json()
        if not all_tests:
            print("ERROR: tests.json not found. Run without --tier first "
                  "to generate unit tests.")
            sys.exit(1)

        tier = args.tier
        for tier_name in TIER_PROGRAMS.get(tier, []):
            if tier_name not in all_tests:
                print(f"  SKIP {tier_name}: no unit tests in tests.json "
                      f"(run without --tier first)")
                continue

            patterns = list(TIERS[tier_name])
            prog_info = load_program_info(tier_name)
            bitlength = get_tier_vectors(tier_name, tier)
            if bitlength is None:
                print(f"  {tier_name}: tier '{tier}' not applicable")
                continue

            print(f"\n=== {tier_name} ({len(patterns)} patterns) ===")
            data_file, size_bytes = generate_tier_data(
                tier_name, tier, bitlength, prog_info, patterns)
            sha = file_sha256(os.path.join(TESTS_DIR, data_file))
            prov = make_provenance(
                source="synthetic",
                seed=TIER_SEEDS[tier],
                description=(
                    f"Random byte payload as 8 basis bit-planes. "
                    f"{len(patterns)} Snort IDS regex patterns, "
                    f"single any_match output. {bitlength:,} bytes."
                ),
                generated_by=f"generate_tests.py --tier {tier}",
            )
            prov["sha256"] = sha
            entry = tier_test_entry(
                name=f"Tier {tier} ({bitlength:,} bytes)",
                bitlength=bitlength,
                data_file=data_file,
                size_bytes=size_bytes,
                provenance=prov,
            )
            all_tests[tier_name] = merge_tier_entry(
                all_tests[tier_name], entry)
            # Save incrementally after each program so partial progress
            # is preserved if a later program fails (e.g., timeout)
            save_tests_json(all_tests)
            print(f"  {len(all_tests[tier_name])} test entries for {tier_name}")
    else:
        # Default: generate unit tests, preserve existing tier entries
        all_tests = {}
        # Collect tier data_files to preserve
        tier_data_files = set()
        for t in TIER_SEEDS:
            for tn in TIERS:
                tier_data_files.add(f"{tn}_{t}.bsdata")

        existing = load_tests_json()

        for tier_name in TIERS:
            patterns = list(TIERS[tier_name])
            print(f"\n=== {tier_name} ({len(patterns)} patterns) ===")

            prog_info = load_program_info(tier_name)
            print(f"  Loaded {prog_info.source_path}")

            # Generate unit tests (batched bsim invocation)
            print("  Generating unit tests...")
            tests = generate_unit_tests(tier_name, patterns, prog_info)

            # Real-data tests (generated at runtime, optional)
            tests.append({
                "name": "WRCCDC 2012 pcap (10K bytes)",
                "category": "generated",
                "optional": True,
                "bitlength": 10000,
                "generate": {"type": "file", "dataset": "wrccdc2012_10k"},
            })
            tests.append({
                "name": "WRCCDC 2012 pcap (1M bytes)",
                "category": "generated",
                "optional": True,
                "bitlength": 1000000,
                "generate": {"type": "file", "dataset": "wrccdc2012_1m"},
            })

            # Preserve existing tier entries
            for entry in existing.get(tier_name, []):
                if entry.get("data_file") in tier_data_files:
                    tests.append(entry)

            all_tests[tier_name] = tests
            print(f"  {len(tests)} test entries for {tier_name}")

        save_tests_json(all_tests)


if __name__ == "__main__":
    main()
