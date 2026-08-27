#!/usr/bin/env python3
"""Curate Snort IDS patterns into a sorted manifest for regex benchmarks.

Reads all 2,396 patterns from snort.regex, compiles each via icgrep
Pablo IR pipeline, records success/failure and op count, sorts
successful patterns by op count ascending, and writes a manifest JSON.

Usage:
    python curate_patterns.py               # compile all, write manifest
    python curate_patterns.py --jobs 8      # parallel compilation (default: CPU count)
    python curate_patterns.py --describe    # print manifest summary
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from benchmark.apps.regex.src.pablo_to_bs import compile_pattern_via_icgrep

RAW_DIR = os.path.join(os.path.dirname(__file__), "../datasets/raw")
SNORT_REGEX = os.path.join(RAW_DIR, "snort.regex")
MANIFEST_PATH = os.path.join(RAW_DIR, "pattern_manifest.json")


def parse_snort_regex(path: str) -> list[tuple[int, str, str]]:
    """Parse snort.regex file. Returns [(line_number, pattern, flags)]."""
    results = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line or not line.startswith("/"):
                continue
            # Find last / to split pattern and flags
            last_slash = line.rfind("/")
            if last_slash <= 0:
                continue
            pattern = line[1:last_slash]
            flags = line[last_slash + 1:]
            results.append((i, pattern, flags))
    return results


def _compile_one(args):
    """Worker function for parallel compilation."""
    line_num, pattern, flags = args
    name = f"snort_{line_num:04d}"

    bs_lines, output_var = compile_pattern_via_icgrep(name, pattern, flags)
    if not bs_lines or output_var is None:
        return {
            "line_number": line_num,
            "name": name,
            "pattern": pattern,
            "flags": flags,
            "success": False,
            "op_count": 0,
            "n_lines": 0,
        }

    # Count ops by parsing the generated .bs code fragment.
    # Build a minimal parseable program with the lines.
    source_lines = ["input stream b[8]", f"output stream {name}", ""]
    source_lines.extend(bs_lines)
    source_lines.append(f"{name} = {output_var}")
    source = "\n".join(source_lines) + "\n"

    try:
        from simulator.pythonsim import parse
        from simulator.pythonsim.parser import count_stmts
        prog = parse(source)
        op_count = count_stmts(prog)
    except Exception:
        # Fallback: count lines with assignments (rough estimate)
        op_count = sum(1 for ln in bs_lines
                       if ln.strip() and "=" in ln and not ln.strip().startswith("//"))

    return {
        "line_number": line_num,
        "name": name,
        "pattern": pattern,
        "flags": flags,
        "success": True,
        "op_count": op_count,
        "n_lines": len(bs_lines),
    }


def compile_and_count_ops(pattern_entries, n_jobs=None):
    """Compile all patterns and return results list."""
    if n_jobs is None:
        n_jobs = min(multiprocessing.cpu_count(), 16)

    total = len(pattern_entries)
    print(f"Compiling {total} patterns with {n_jobs} workers...")
    t0 = time.time()

    results = []
    # Use multiprocessing for parallelism
    with multiprocessing.Pool(n_jobs) as pool:
        for i, result in enumerate(pool.imap(_compile_one, pattern_entries)):
            results.append(result)
            if (i + 1) % 100 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                status = "OK" if result["success"] else "FAIL"
                print(f"  [{i+1}/{total}] {elapsed:.1f}s "
                      f"({sum(1 for r in results if r['success'])} success)")

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r["success"])
    n_fail = total - n_ok
    print(f"Done in {elapsed:.1f}s: {n_ok} success, {n_fail} failed")
    return results


def build_manifest(results):
    """Build the manifest from compilation results.

    Sorts successful patterns by op count ascending, assigns tier membership.
    """
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    # Sort by op count ascending (stable sort preserves line order for ties)
    successful.sort(key=lambda r: (r["op_count"], r["line_number"]))

    manifest = {
        "description": (
            "Pattern manifest for Snort IDS regex benchmark. "
            "Compiled via icgrep Pablo IR pipeline. "
            "Sorted by op count ascending."
        ),
        "source": "AutomataZoo snort.regex (Snort 2.9.7.0, Wadden et al. 2018)",
        "total_patterns": len(results),
        "compiled_patterns": len(successful),
        "failed_patterns": len(failed),
        "tier_sizes": {
            "regex_small": 50,
            "regex_medium": 500,
            "regex_large": len(successful),
        },
        "patterns": [],
        "failed": [],
    }

    for i, r in enumerate(successful):
        entry = {
            "rank": i,
            "line_number": r["line_number"],
            "name": r["name"],
            "pattern": r["pattern"],
            "flags": r["flags"],
            "op_count": r["op_count"],
            "n_lines": r["n_lines"],
        }
        manifest["patterns"].append(entry)

    for r in failed:
        manifest["failed"].append({
            "line_number": r["line_number"],
            "name": r["name"],
            "pattern": r["pattern"],
            "flags": r["flags"],
        })

    return manifest


def describe_manifest(manifest):
    """Print summary of manifest."""
    patterns = manifest["patterns"]
    n = len(patterns)
    tiers = manifest["tier_sizes"]

    print(f"Pattern manifest: {manifest['compiled_patterns']}/{manifest['total_patterns']} compiled")
    print(f"  Source: {manifest['source']}")
    print()

    for tier_name, tier_size in tiers.items():
        subset = patterns[:tier_size]
        total_ops = sum(p["op_count"] for p in subset)
        min_ops = subset[0]["op_count"] if subset else 0
        max_ops = subset[-1]["op_count"] if subset else 0
        n_streams = 8 + len(subset)
        print(f"  {tier_name}: {len(subset)} patterns, {total_ops:,} ops "
              f"(range {min_ops}-{max_ops}), {n_streams} streams")


def main():
    parser = argparse.ArgumentParser(description="Curate Snort regex patterns")
    parser.add_argument("--jobs", "-j", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--describe", action="store_true",
                        help="Print existing manifest summary")
    args = parser.parse_args()

    if args.describe:
        if not os.path.exists(MANIFEST_PATH):
            print(f"No manifest at {MANIFEST_PATH}")
            sys.exit(1)
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        describe_manifest(manifest)
        return

    # Parse snort.regex
    entries = parse_snort_regex(SNORT_REGEX)
    print(f"Parsed {len(entries)} patterns from {SNORT_REGEX}")

    # Compile all patterns
    results = compile_and_count_ops(entries, n_jobs=args.jobs)

    # Build and write manifest
    manifest = build_manifest(results)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest to {MANIFEST_PATH}")

    describe_manifest(manifest)


if __name__ == "__main__":
    main()
