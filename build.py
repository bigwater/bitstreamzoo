#!/usr/bin/env python3
"""Build script for the Bitstream Benchmark Suite.

Downloads, converts, and generates test data for all benchmark domains.
After running, `python benchmark/run_all.py` exercises all tests.

Usage:
    python build.py              # full build (all domains), unit fixtures only
    python build.py shift_and    # single domain
    python build.py --status     # show dataset availability
    python build.py --no-download  # skip downloads
    python build.py --generate-only  # only regenerate tests.json + unit fixtures
    python build.py --generate-only --tier small  # generate small tier data
    python build.py --generate-only --tier medium # generate medium tier data

The synthetic small/medium/large tier .bsdata files are NOT produced by
a plain `python build.py`: they are large (~50 MB / 500 MB / 5 GB per
test) and git-ignored. Generate them explicitly with
`--tier {small,medium,large}`. `python benchmark/run_all.py` skips any
tier dataset that has not been generated.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
APPS_DIR = os.path.join(ROOT, "benchmark", "apps")

sys.path.insert(0, ROOT)
from benchmark.domains import discover_domains

# All domains in the suite (auto-discovered from benchmark/apps/).
DOMAINS = discover_domains(APPS_DIR)

# Datasets that can be freely downloaded (no registration required).
# Each entry: url, dest (relative to ROOT), description.
DOWNLOADS = {
    "mison": [
        {
            "url": "https://data.gharchive.org/2024-01-01-0.json.gz",
            "dest": "benchmark/apps/mison/datasets/raw/gharchive_hourly.json.gz",
            "desc": "GH Archive hourly JSON dump",
        },
    ],
    "regex": [
        {
            "url": "https://archive.wrccdc.org/pcaps/2012/wrccdc2012.pcap.gz",
            "dest": "benchmark/apps/regex/datasets/raw/wrccdc2012.pcap.gz",
            "desc": "WRCCDC 2012 network traffic pcap",
        },
    ],
    "shift_and": [
        {
            "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz",
            "dest": "benchmark/apps/shift_and/datasets/raw/chr1.fa.gz",
            "desc": "Human chromosome 1 (UCSC hg38)",
        },
    ],
}

# Per-domain conversion recipes.
# Each recipe: raw input file (relative to datasets/), convert.py args, npz output.
# Only runs if raw file exists AND npz output doesn't.
CONVERSIONS = {
    # mison, shift_and: make_data.py reads raw files directly.
}

# Domains whose convert.py runs in batch mode (no per-file recipes).
BATCH_CONVERT = {
    "circuit_sim": ".bench netlists -> .bs bitstream programs",
}

# Domains whose convert.py uses flag-based training (no per-file recipes).
TRAINING_DOMAINS = {
    "bnn": [
        {"args": ["--mnist"], "output": "small/mnist_params.npz",
         "desc": "Train MNIST BNN"},
    ],
}

# Domains requiring manual data acquisition (no free direct-download URLs).
MANUAL_DOMAINS = {
    "epistasis": "PLINK genotype data: see benchmark/apps/epistasis/datasets/raw/readme.md",
}


def build_backends() -> bool:
    """Build C++ and CUDA simulator backends. Returns True if all succeeded."""
    ok = True

    src_dir = os.path.join(ROOT, "simulator", "csim")
    cmake_file = os.path.join(src_dir, "CMakeLists.txt")
    if not os.path.exists(cmake_file):
        print(f"  [skip] no CMakeLists.txt at {src_dir}")
        return False

    build_dir = os.path.join(src_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    print("  [build] C++ and CUDA backends")
    try:
        subprocess.run(
            ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=build_dir, check=True, capture_output=True, text=True,
        )
        nproc = os.cpu_count() or 4
        subprocess.run(
            ["make", f"-j{nproc}"],
            cwd=build_dir, check=True, capture_output=True, text=True,
        )
        print("    [ok] Build succeeded")
    except FileNotFoundError:
        print("    [error] cmake or make not found")
        ok = False
    except subprocess.CalledProcessError as e:
        print(f"    [error] Build failed: {e}")
        if e.stderr:
            for line in e.stderr.strip().splitlines()[-5:]:
                print(f"    {line}")
        ok = False

    # Verify binaries
    for binary, label in [
        (os.path.join(build_dir, "bsim"), "bsim"),
        (os.path.join(build_dir, "bsim_cuda"), "bsim_cuda"),
    ]:
        if os.path.exists(binary):
            print(f"  [ok] {label} found at {binary}")
        else:
            print(f"  [warn] {label} not found at {binary}")

    return ok


def check_dep(module_name: str) -> bool:
    """Check if a Python module is importable."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def download_if_missing(url: str, dest: str, desc: str) -> bool:
    """Download a file if it doesn't already exist. Returns True if downloaded."""
    full_dest = os.path.join(ROOT, dest)
    if os.path.exists(full_dest):
        print(f"    [skip] {desc} (already exists)")
        return False

    os.makedirs(os.path.dirname(full_dest), exist_ok=True)
    print(f"    [download] {desc}")
    print(f"      {url}")

    # Download to a temp path and rename on success, so an interrupted
    # download never leaves a truncated file at the final (idempotently
    # skipped) destination. Try wget first, then curl.
    part = full_dest + ".part"
    for cmd in (["wget", "-q", "--show-progress", "-O", part, url],
                ["curl", "-fL", "-o", part, url]):
        try:
            subprocess.run(cmd, check=True)
            os.replace(part, full_dest)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            err = e
    if os.path.exists(part):
        os.unlink(part)
    print(f"    [error] Failed to download: {err}")
    return False


def convert_domain(name: str) -> bool:
    """Run convert.py for a domain.

    Two modes:
    - Recipe-based (CONVERSIONS dict): each recipe converts a raw file to .npz.
      Skips if raw input missing or output already exists (idempotent).
    - Batch mode (BATCH_CONVERT set): runs convert.py with no args.
      Used for circuit_sim (.bench -> .bs conversion).

    Returns True if any conversion was run.
    """
    convert_py = os.path.join(APPS_DIR, name, "datasets", "raw", "convert.py")
    if not os.path.exists(convert_py):
        return False

    ran_any = False

    # Batch mode: run convert.py with no args
    if name in BATCH_CONVERT:
        print(f"    [convert] {name}: {BATCH_CONVERT[name]}")
        try:
            subprocess.run([sys.executable, convert_py], cwd=ROOT, check=True,
                           capture_output=True, text=True)
            ran_any = True
        except subprocess.CalledProcessError as e:
            print(f"    [error] Conversion failed: {e}")
            if e.stderr:
                print(f"    stderr: {e.stderr[:200]}")

    # Training mode: run convert.py with flags (e.g., BNN training)
    if name in TRAINING_DOMAINS:
        datasets_dir = os.path.join(APPS_DIR, name, "datasets")
        for recipe in TRAINING_DOMAINS[name]:
            out_path = os.path.join(datasets_dir, recipe["output"])
            if os.path.exists(out_path):
                print(f"    [skip] {recipe['output']} (already exists)")
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cmd = [sys.executable, convert_py] + recipe["args"]
            print(f"    [train] {recipe['desc']}")
            try:
                subprocess.run(cmd, cwd=os.path.dirname(convert_py),
                               check=True)
                ran_any = True
            except subprocess.CalledProcessError as e:
                print(f"    [error] Training failed: {e}")

    # Recipe-based mode
    if name in CONVERSIONS:
        datasets_dir = os.path.join(APPS_DIR, name, "datasets")
        for recipe in CONVERSIONS[name]:
            raw_path = os.path.join(datasets_dir, recipe["raw"])
            npz_path = os.path.join(datasets_dir, recipe["npz"])

            if not os.path.exists(raw_path):
                print(f"    [skip] {recipe['raw']} (not downloaded)")
                continue
            if os.path.exists(npz_path):
                print(f"    [skip] {recipe['npz']} (already exists)")
                continue

            os.makedirs(os.path.dirname(npz_path), exist_ok=True)
            cmd = [sys.executable, convert_py, raw_path, "--npz", npz_path]
            if recipe.get("source"):
                cmd += ["--source", recipe["source"]]
            if recipe.get("extra_args"):
                cmd += recipe["extra_args"]

            print(f"    [convert] {os.path.basename(raw_path)} -> {recipe['npz']}")
            try:
                subprocess.run(cmd, cwd=ROOT, check=True)
                ran_any = True
            except subprocess.CalledProcessError as e:
                print(f"    [error] Conversion failed: {e}")

    return ran_any


def generate_tests(name: str, tier: str | None = None) -> bool:
    """Run generate_tests.py for a domain. Returns True if run."""
    gen_py = os.path.join(APPS_DIR, name, "datasets", "tests", "generate_tests.py")
    if not os.path.exists(gen_py):
        print(f"    [skip] No generate_tests.py for {name}")
        return False

    cmd = [sys.executable, gen_py]
    if tier:
        cmd.extend(["--tier", tier])
        print(f"    [generate] {name} tests.json + {tier} tier data")
    else:
        print(f"    [generate] {name} tests.json")
    try:
        subprocess.run(
            cmd,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"    [error] Test generation failed: {e}")
        if e.stdout:
            print(f"    stdout: {e.stdout[:200]}")
        if e.stderr:
            print(f"    stderr: {e.stderr[:200]}")
        return None


def get_npz_files(name: str) -> dict[str, list[str]]:
    """Get available .npz files for a domain, organized by size tier."""
    result = {"small": [], "medium": [], "large": []}
    datasets_dir = os.path.join(APPS_DIR, name, "datasets")
    for tier in result:
        tier_dir = os.path.join(datasets_dir, tier)
        if os.path.isdir(tier_dir):
            for f in sorted(os.listdir(tier_dir)):
                if f.endswith(".npz"):
                    result[tier].append(f)
    return result


def get_test_count(name: str) -> int:
    """Count tests in tests.json for a domain."""
    import json
    tests_json = os.path.join(APPS_DIR, name, "datasets", "tests", "tests.json")
    if not os.path.exists(tests_json):
        return 0
    try:
        with open(tests_json) as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            return sum(len(v) for v in data.values() if isinstance(v, list))
        return 0
    except (json.JSONDecodeError, ValueError):
        return 0


def count_real_tests(name: str) -> int:
    """Count optional real-data tests in tests.json."""
    import json
    tests_json = os.path.join(APPS_DIR, name, "datasets", "tests", "tests.json")
    if not os.path.exists(tests_json):
        return 0
    try:
        with open(tests_json) as f:
            data = json.load(f)
        if isinstance(data, list):
            return sum(1 for t in data if t.get("optional"))
        return 0
    except (json.JSONDecodeError, ValueError):
        return 0


def print_status():
    """Print dataset availability status for all domains."""
    print("\nBitstream Benchmark Suite — Dataset Status\n")
    print(f"{'Domain':<16} {'Small':<14} {'Medium':<14} {'Large':<14} {'Tests'}")
    print("-" * 72)

    for name in sorted(DOMAINS):
        npz = get_npz_files(name)
        total = get_test_count(name)
        n_real = count_real_tests(name)

        def fmt_tier(files):
            if files:
                short = files[0].replace(".npz", "")
                if len(short) > 10:
                    short = short[:10]
                return f"\u2713 {short}"
            return "\u2014"

        # Check if large download.sh exists and has actual URLs
        large_dir = os.path.join(APPS_DIR, name, "datasets", "large")
        has_large_placeholder = os.path.exists(os.path.join(large_dir, "download.sh"))
        large_str = fmt_tier(npz["large"])
        if not npz["large"] and has_large_placeholder:
            large_str = "\u2717 (download)"

        test_str = str(total)
        if n_real > 0:
            test_str += f" + {n_real} real"

        print(f"{name:<16} {fmt_tier(npz['small']):<14} "
              f"{fmt_tier(npz['medium']):<14} {large_str:<14} {test_str}")

    print()
    print("\u2713 = available    \u2717 = not downloaded    \u2014 = N/A")
    print()


def precompute_tests(name: str) -> bool:
    """Run precompute.py for a domain to generate .bsdata files. Returns True if run."""
    precompute_py = os.path.join(APPS_DIR, name, "datasets", "tests", "precompute.py")
    if not os.path.exists(precompute_py):
        return False

    print(f"    [precompute] {name} .bsdata files")
    try:
        subprocess.run(
            [sys.executable, precompute_py],
            cwd=ROOT,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"    [error] Precompute failed: {e}")
        return None


def make_data_domain(name: str) -> bool:
    """Run make_data.py for a domain to generate small/medium/large .npz files."""
    make_data_py = os.path.join(APPS_DIR, name, "datasets", "make_data.py")
    if not os.path.exists(make_data_py):
        return False

    print(f"    [make_data] {name} small/medium/large datasets")
    try:
        subprocess.run(
            [sys.executable, make_data_py],
            cwd=ROOT,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"    [error] make_data failed: {e}")
        return False


def build_domain(name: str, skip_download: bool = False,
                 generate_only: bool = False, tier: str | None = None) -> list[str]:
    """Build one domain: download + convert + make_data + generate tests + precompute.

    Returns the list of failed steps (empty on success). Missing optional
    downloads are not failures; a generator or precompute step that ran and
    exited nonzero is.
    """
    errors: list[str] = []
    print(f"\n  [{name}]")

    if not generate_only:
        # Download (if applicable)
        if not skip_download and name in DOWNLOADS:
            for dl in DOWNLOADS[name]:
                download_if_missing(dl["url"], dl["dest"], dl["desc"])
        elif name in MANUAL_DOMAINS:
            print(f"    [info] {MANUAL_DOMAINS[name]}")

        # Convert (idempotent)
        convert_domain(name)

        # Generate benchmark datasets (small/medium/large)
        make_data_domain(name)

    # Generate tests.json (and tier data if --tier specified)
    if generate_tests(name, tier=tier) is None:
        errors.append(f"{name}: generate_tests")

    # Precompute .bsdata files
    if not generate_only:
        if precompute_tests(name) is None:
            errors.append(f"{name}: precompute")
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Build script for the Bitstream Benchmark Suite",
    )
    parser.add_argument("domains", nargs="*",
                        help="Domains to build (default: all)")
    parser.add_argument("--status", action="store_true",
                        help="Show dataset availability")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip downloads")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only regenerate tests.json files")
    parser.add_argument("--build-backends", action="store_true",
                        help="Build C++ and CUDA simulator backends")
    parser.add_argument("--tier", choices=["small", "medium", "large"],
                        help="Generate tier data (small ~50MB, medium ~500MB, large ~5GB)")
    args = parser.parse_args()

    if args.build_backends:
        print("Bitstream Benchmark Suite — Build Backends\n")
        build_backends()
        if not args.status and not args.domains and not args.generate_only:
            # If only --build-backends was specified, also do the full build
            pass  # fall through to normal build
        elif args.status:
            pass  # fall through to status
        # Otherwise fall through to normal domain build

    if args.status:
        print_status()
        return

    # Determine which domains to build
    if args.domains:
        selected = []
        for name in args.domains:
            if name not in DOMAINS:
                print(f"Unknown domain: {name}", file=sys.stderr)
                print(f"Available: {', '.join(sorted(DOMAINS))}", file=sys.stderr)
                sys.exit(1)
            selected.append(name)
    else:
        selected = list(DOMAINS)

    print("Bitstream Benchmark Suite — Build")
    print(f"Domains: {', '.join(selected)}")

    # Check numpy (required for real-data tests)
    has_numpy = check_dep("numpy")
    if not has_numpy:
        print("\n  [warning] numpy not installed — real-data tests will be skipped")
        print("  Install with: pip install numpy")

    failures: list[str] = []
    for name in selected:
        failures.extend(build_domain(name, skip_download=args.no_download,
                                     generate_only=args.generate_only,
                                     tier=args.tier))

    # Summary
    print("\n" + "=" * 50)
    if failures:
        print("Build FAILED for: " + "; ".join(failures))
        sys.exit(1)
    print("Build complete. Run tests with:")
    print("  python benchmark/run_all.py")
    if args.domains:
        print(f"  python benchmark/run_all.py {' '.join(args.domains)}")
    print()


if __name__ == "__main__":
    main()
