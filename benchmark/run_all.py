#!/usr/bin/env python3
"""Unified benchmark runner for all bitstream program domains.

Usage:
    python3 benchmark/run_all.py                         # run all domains (python)
    python3 benchmark/run_all.py circuit_sim             # run one domain
    python3 benchmark/run_all.py --backend cpp           # C++ only
    python3 benchmark/run_all.py --backend all           # run all backends (python, cpp, cuda)
    python3 benchmark/run_all.py --size small            # small tier only (~50MB per test)
    python3 benchmark/run_all.py --size medium           # medium tier only (~500MB per test)
    python3 benchmark/run_all.py --size large            # large tier only (~5GB per test)
    python3 benchmark/run_all.py --unit                  # unit fixtures only (clean checkout, no build)
    python3 benchmark/run_all.py --list                  # list all programs

Size filtering is by .bsdata file size (not bitlength). Without --size,
every test whose data is present on disk runs.

--unit runs only the self-contained unit-correctness fixtures: inline
tests and the small committed .bsdata (< 5 MB each). It needs no tier
data and passes from a clean checkout with no build step.

Without --unit, the synthetic small/medium/large tier datasets are also
run. Those .bsdata files are git-ignored and regenerated on demand with
    python build.py --generate-only --tier {small,medium,large}
A tier dataset that has not been generated is skipped, not an error:
the runner prints an up-front notice of what was skipped and how to
generate it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
import sys
import os
import subprocess
import tempfile

# Allow importing from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark.base import (
    BenchmarkDomain, BenchmarkResult, GenericDomain,
    PythonBackend, CppBackend, CudaBackend,
)
from benchmark.domains import discover_domains

DOMAIN_NAMES = discover_domains()


def make_domains(backend=None, size_filter=None,
                 unit_only=False) -> dict[str, BenchmarkDomain]:
    return {name: GenericDomain(name, backend=backend,
                                size_filter=size_filter,
                                unit_only=unit_only)
            for name in DOMAIN_NAMES}


def make_backend(name: str):
    """Create a Backend instance from a CLI name."""
    if name == "python":
        return PythonBackend()
    elif name == "cpp":
        return CppBackend(reuse_mem=True)
    elif name == "cuda":
        return CudaBackend(reuse_mem=True)
    elif name == "cuda32":
        return CudaBackend(word_bits=32)
    elif name == "cuda_reuse":
        return CudaBackend(reuse_mem=True)
    elif name == "cuda32_reuse":
        return CudaBackend(word_bits=32, reuse_mem=True)
    elif name == "cpp_scalar":
        return CppBackend(variant="scalar")
    elif name == "cpp_simd":
        return CppBackend(variant="simd")
    elif name == "cpp_simd_ks":
        return CppBackend(variant="simd", add_mode="kogge-stone")
    elif name == "all":
        return None  # sentinel: handled specially in main()
    else:
        print(f"Unknown backend: {name}", file=sys.stderr)
        print("Available: python, cpp, cpp_scalar, cpp_simd, cpp_simd_ks, "
              "cuda, cuda32, cuda_reuse, cuda32_reuse, all",
              file=sys.stderr)
        sys.exit(1)


def list_programs():
    """Print all discovered programs across all domains."""
    domains = make_domains()
    print(f"{'Domain':<16} {'Program':<24} {'Stmts':>6}  Source")
    print("-" * 80)
    for name, domain in domains.items():
        for prog in domain.discover_programs():
            print(f"{name:<16} {prog.name:<24} {prog.n_stmts:>6}  "
                  f"{os.path.relpath(prog.source_path)}")


def print_results(results: list[BenchmarkResult]):
    """Print a formatted summary table."""
    # Column widths
    w_dom = max(len("Domain"), max((len(r.domain) for r in results), default=0))
    w_prog = max(len("Program"), max((len(r.program) for r in results), default=0))
    w_test = max(len("Test"), max((len(r.test_name) for r in results), default=0))

    header = (f"{'Domain':<{w_dom}}  {'Program':<{w_prog}}  {'Test':<{w_test}}  "
              f"{'Bitlen':>8}  {'Stmts':>6}  {'Ops':>8}  {'Exec':>10}  {'Time':>8}  Status")
    sep = "-" * len(header)

    print()
    print(header)
    print(sep)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        time_str = f"{r.elapsed_s:.3f}s"
        exec_str = f"{r.exec_ms:.2f}ms"
        print(f"{r.domain:<{w_dom}}  {r.program:<{w_prog}}  "
              f"{r.test_name:<{w_test}}  {r.bitlength:>8}  "
              f"{r.n_stmts:>6}  {r.op_count:>8}  {exec_str:>10}  {time_str:>8}  "
              f"{status}")
    print(sep)

    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)
    total = len(results)
    print(f"Total: {n_pass} passed, {n_fail} failed, {total} tests")

    if total == 0:
        print("NO TESTS EXECUTED")
    elif n_fail == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {n_fail}")


def write_results_json(path: str, results: list[BenchmarkResult]):
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def read_results_json(path: str) -> list[BenchmarkResult]:
    with open(path) as f:
        data = json.load(f)
    return [BenchmarkResult(**row) for row in data]


def find_missing_tier_data(domain_names: list[str],
                           size_filter: str | None = None
                           ) -> list[tuple[str, str, str, int]]:
    """Scan tests.json for tier datasets whose .bsdata file is absent.

    Tier datasets are git-ignored and regenerated on demand, so a clean
    checkout has none of them. This reports which selected tests will be
    skipped for that reason. Returns (domain, test_name, data_file,
    size_bytes) tuples, restricted to the --size class when one is set.
    """
    apps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apps")
    size_range = (GenericDomain.SIZE_THRESHOLDS.get(size_filter)
                  if size_filter else None)
    missing: list[tuple[str, str, str, int]] = []
    for name in domain_names:
        tests_json = os.path.join(apps_dir, name, "datasets", "tests",
                                  "tests.json")
        if not os.path.exists(tests_json):
            continue
        try:
            with open(tests_json) as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, list):
            cases = data
        elif isinstance(data, dict):
            cases = [c for v in data.values() if isinstance(v, list) for c in v]
        else:
            cases = []
        tests_dir = os.path.dirname(tests_json)
        for case in cases:
            # Only regenerated tier data (carries a provenance block) is
            # expected to be absent; committed fixtures are not.
            if "provenance" not in case:
                continue
            data_file = case.get("data_file")
            if not data_file:
                continue
            if os.path.exists(os.path.join(tests_dir, data_file)):
                continue
            size_bytes = case.get("size_bytes", 0)
            if size_range is not None:
                lo, hi = size_range
                hi = hi if hi is not None else float("inf")
                if not (lo <= size_bytes <= hi):
                    continue
            missing.append((name, case.get("name", "?"), data_file, size_bytes))
    return missing


def print_tier_notice(domain_names: list[str], size_filter: str | None):
    """Print an up-front notice of tier datasets that will be skipped."""
    missing = find_missing_tier_data(domain_names, size_filter)
    if not missing:
        return
    total_bytes = sum(m[3] for m in missing)
    size_note = f", ~{total_bytes / 1e9:.1f} GB to generate" if total_bytes else ""
    tiers = size_filter if size_filter else "{small,medium,large}"
    print()
    print(f"Note: {len(missing)} tier dataset(s) not generated{size_note} "
          f"-- those tests will be skipped.")
    print(f"  Generate them:        python build.py --generate-only "
          f"--tier {tiers}")
    print(f"  Committed-only suite: python benchmark/run_all.py --unit")
    print()


def main():
    args = sys.argv[1:]

    if "--list" in args:
        list_programs()
        return

    json_out = None
    if "--json-out" in args:
        idx = args.index("--json-out")
        if idx + 1 >= len(args):
            print("--json-out requires a file path", file=sys.stderr)
            sys.exit(1)
        json_out = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    # Parse --backend flag
    backend_name = "python"
    if "--backend" in args:
        idx = args.index("--backend")
        if idx + 1 >= len(args):
            print("--backend requires a value: python, cpp, cuda, all", file=sys.stderr)
            sys.exit(1)
        backend_name = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    # Parse --size flag
    size_filter = None
    if "--size" in args:
        idx = args.index("--size")
        if idx + 1 >= len(args):
            print("--size requires a value: small, medium, large", file=sys.stderr)
            sys.exit(1)
        size_filter = args[idx + 1]
        args = args[:idx] + args[idx + 2:]
        valid_sizes = list(GenericDomain.SIZE_THRESHOLDS.keys())
        if size_filter not in valid_sizes:
            print(f"Unknown size: {size_filter}", file=sys.stderr)
            print(f"Available: {', '.join(valid_sizes)}", file=sys.stderr)
            sys.exit(1)

    # Parse --unit flag (clean-checkout unit-correctness suite only)
    unit_only = False
    if "--unit" in args:
        unit_only = True
        args = [a for a in args if a != "--unit"]

    if backend_name == "all":
        # Run each backend/domain in a subprocess to keep Python bigint-heavy
        # domains from retaining hundreds of GB of RSS across the full suite.
        all_backend_names = ["python", "cpp", "cuda"]
        all_results: list[BenchmarkResult] = []
        failed_children: list[str] = []
        # Validate domain names first
        test_domains = make_domains(size_filter=size_filter,
                                    unit_only=unit_only)
        if args:
            selected = []
            for name in args:
                if name not in test_domains:
                    print(f"Unknown domain: {name}", file=sys.stderr)
                    print(f"Available: {', '.join(test_domains)}", file=sys.stderr)
                    sys.exit(1)
                selected.append(name)
        else:
            selected = list(test_domains.keys())

        if not unit_only:
            print_tier_notice(selected, size_filter)

        for bname in all_backend_names:
            print(f"\n{'='*60}")
            print(f"Backend: {bname}")
            print(f"{'='*60}")
            for name in selected:
                print(f"Running {name} ...")
                with tempfile.NamedTemporaryFile(
                    prefix=f"run_all_{bname}_{name}_",
                    suffix=".json",
                    delete=False,
                ) as tmp:
                    result_path = tmp.name
                try:
                    cmd = [
                        sys.executable,
                        os.path.abspath(__file__),
                        name,
                        "--backend",
                        bname,
                        "--json-out",
                        result_path,
                    ]
                    if size_filter:
                        cmd.extend(["--size", size_filter])
                    if unit_only:
                        cmd.append("--unit")
                    # Per-domain timeout: small=10min, medium=2h, large=2h, no-filter=2h
                    domain_timeout = {"small": 600, "medium": 7200, "large": 7200}.get(size_filter, 7200)
                    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                    try:
                        child = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=domain_timeout,
                            env=env,
                        )
                    except subprocess.TimeoutExpired:
                        print(f"  TIMEOUT: {name}/{bname} exceeded {domain_timeout}s",
                              file=sys.stderr)
                        child = None
                    if child and os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                        all_results.extend(read_results_json(result_path))
                    if child is None or child.returncode != 0:
                        # A crashed or timed-out child may still have written
                        # partial results above; the run as a whole must fail.
                        failed_children.append(f"{name}/{bname}")
                    if child and child.returncode != 0 and child.stderr:
                        print(child.stderr, file=sys.stderr, end="")
                finally:
                    if os.path.exists(result_path):
                        os.unlink(result_path)

        print_results(all_results)
        if failed_children:
            print(f"FAILED backend runs: {', '.join(failed_children)}",
                  file=sys.stderr)
            sys.exit(1)
    else:
        backend = make_backend(backend_name)
        DOMAINS = make_domains(backend, size_filter=size_filter,
                               unit_only=unit_only)

        # Determine which domains to run
        if args:
            selected = []
            for name in args:
                if name not in DOMAINS:
                    print(f"Unknown domain: {name}", file=sys.stderr)
                    print(f"Available: {', '.join(DOMAINS)}", file=sys.stderr)
                    sys.exit(1)
                selected.append(name)
        else:
            selected = list(DOMAINS.keys())

        print(f"Backend: {backend_name}"
              + (f", size filter: {size_filter}" if size_filter else ""))

        if not unit_only:
            print_tier_notice(selected, size_filter)

        # Run benchmarks
        all_results: list[BenchmarkResult] = []
        for name in selected:
            domain = DOMAINS[name]
            print(f"Running {name} ...")
            results = domain.run_all()
            all_results.extend(results)

        print_results(all_results)

    if json_out:
        write_results_json(json_out, all_results)

    if not all_results and not json_out:
        # A per-domain child (--json-out) may legitimately have nothing to
        # run when its tier data is absent; the top-level invocation must
        # not report success after executing zero tests.
        print("No tests executed — nothing was selected or every test was "
              "skipped.", file=sys.stderr)
        sys.exit(1)
    if any(not r.passed for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
