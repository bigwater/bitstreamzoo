"""Base classes for the unified benchmark framework."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

from benchmark.testutil import to_json_val

from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter
from simulator.pythonsim.parser import Program, count_stmts


# ── Backend abstraction ──────────────────────────────────────


class Backend:
    """Abstract backend for running .bs programs."""

    def run(self, prog: 'ProgramInfo', inputs: dict, params: dict | None = None,
            input_arrays: dict | None = None, bitlength: int = 0,
            iterations: int = 1, feedback: dict | None = None,
            bsdata_path: str | None = None,
            ) -> tuple[dict, int, float]:
        """Execute program. Returns (outputs_dict, op_count, exec_ms).

        If bsdata_path is provided and the backend supports it, the binary
        reads the .bsdata file directly instead of receiving JSON via stdin.
        """
        raise NotImplementedError


class PythonBackend(Backend):
    """Run programs via the Python interpreter."""

    TIMEOUT = 3600  # seconds, matches C++/CUDA backends

    def run(self, prog, inputs, params=None, input_arrays=None, bitlength=0,
            iterations=1, feedback=None, bsdata_path=None):
        def _timeout_handler(signum, frame):
            raise subprocess.TimeoutExpired("python-interpreter", self.TIMEOUT)

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(self.TIMEOUT)
        try:
            total_ops = 0
            result = None
            cur_inputs = dict(inputs)
            t0 = time.perf_counter()
            for _ in range(iterations):
                interp = Interpreter(bitlength=bitlength)
                result = interp.run(prog.program, cur_inputs, params, input_arrays)
                total_ops += interp.op_count
                if feedback:
                    for out_name, in_name in feedback.items():
                        cur_inputs[in_name] = result[out_name]
            exec_ms = (time.perf_counter() - t0) * 1000.0
            return result or {}, total_ops, exec_ms
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


# ── AMD Zen4 `rep stos` mitigation ────────────────────────────────────
# On AMD Zen4, `rep stosb` silently fails to write the last partial cache line
# of a large buffer once the working set passes roughly 400 MiB. glibc's
# memset dispatches to `rep stosb` above a size threshold, so the corruption
# lands in whichever large-size test happens to run under enough cache
# pressure, with no fault and no pattern that points back at a domain.
# Raising the threshold past any buffer we allocate keeps memset on the
# vectorized path. `rep movsb` (memcpy) is unaffected by the erratum, so this
# leaves it alone rather than disabling ERMS wholesale.
#
# alloc_stream() also allocates stream buffers cache-line aligned, which
# removes the partial tail line from the buffers we own. That is defense in
# depth, not a substitute: std::vector, std::string and the JSON writer
# allocate large buffers too, and their memset calls are not ours to change.
_REP_STOSB_THRESHOLD = 2_000_000_000


def backend_env() -> dict:
    """Environment for bsim / bsim_cuda subprocesses."""
    env = os.environ.copy()
    tunable = f"glibc.cpu.x86_rep_stosb_threshold={_REP_STOSB_THRESHOLD}"
    existing = env.get("GLIBC_TUNABLES", "")
    if "x86_rep_stosb_threshold" in existing:
        return env                      # caller pinned it; respect that
    env["GLIBC_TUNABLES"] = f"{existing}:{tunable}" if existing else tunable
    return env


class CppBackend(Backend):
    """Run programs via the C++ bsim binary."""

    def __init__(self, bsim_path: str | None = None, variant: str | None = None,
                 add_mode: str | None = None, reuse_mem: bool = True,
                 timeout: int = 7200):
        if bsim_path is None:
            bsim_path = os.path.join(
                os.path.dirname(__file__), "..", "simulator", "csim", "build", "bsim")
        self.bsim_path = bsim_path
        self.variant = variant      # None/"simd" (default) or "scalar"
        self.add_mode = add_mode    # None/"ripple", "kogge-stone"
        self.reuse_mem = reuse_mem
        self.timeout = timeout      # subprocess timeout in seconds
        if not os.path.exists(self.bsim_path):
            raise FileNotFoundError(
                f"bsim binary not found at {self.bsim_path}\n"
                "Build it: cd simulator/csim && mkdir -p build && cd build && cmake .. && make -j")

    def run(self, prog, inputs, params=None, input_arrays=None, bitlength=0,
            iterations=1, feedback=None, bsdata_path=None):
        cmd = [self.bsim_path, prog.source_path]
        if self.variant and self.variant not in (None, "simd"):
            cmd.extend(["--backend", self.variant])
        if self.add_mode:
            cmd.extend(["--add", self.add_mode])
        if self.reuse_mem:
            cmd.append("--reuse-mem")

        if bsdata_path:
            cmd.extend(["--input", bsdata_path])
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout,
                env=backend_env(),
            )
        else:
            input_json = build_input_json(inputs, params, input_arrays, bitlength,
                                          iterations, feedback)
            result = subprocess.run(
                cmd, input=json.dumps(input_json),
                capture_output=True, text=True, timeout=self.timeout,
                env=backend_env(),
            )
        if result.returncode != 0:
            raise RuntimeError(f"bsim failed: {result.stderr.strip()}")
        output = json.loads(result.stdout)
        actual = _parse_csim_outputs(output["outputs"])
        return actual, output["op_count"], output.get("exec_ms", 0.0)

    def run_batch(self, prog, bsdata_paths):
        """Run program on each .bsdata file individually.

        Returns list of result dicts, each with 'outputs', 'op_count', 'exec_ms'.
        Multi-case .bsdata files produce multiple results per file.

        Files are run in separate processes to avoid a bsim batch bug where
        outputs get corrupted when cases with very different bitlengths are
        combined (e.g. multi-case unit tests followed by large tier data).
        The .bs file is still parsed only once per process.
        """
        results = []
        for p in bsdata_paths:
            cmd = [self.bsim_path, prog.source_path]
            if self.variant and self.variant not in (None, "simd"):
                cmd.extend(["--backend", self.variant])
            if self.add_mode:
                cmd.extend(["--add", self.add_mode])
            if self.reuse_mem:
                cmd.append("--reuse-mem")
            cmd.extend(["--input", p])
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout,
                env=backend_env())
            if result.returncode != 0:
                raise RuntimeError(f"bsim failed: {result.stderr.strip()}")
            output = json.loads(result.stdout)
            if isinstance(output, list):
                results.extend(output)
            else:
                results.append(output)
        return results


class CudaBackend(Backend):
    """Run programs via the CUDA bsim_cuda binary (one-kernel-one-inst)."""

    def __init__(self, bsim_path: str | None = None, word_bits: int = 64,
                 reuse_mem: bool = True, timeout: int = 7200):
        if bsim_path is None:
            bsim_path = os.path.join(
                os.path.dirname(__file__), "..", "simulator", "csim", "build", "bsim_cuda")
        self.bsim_path = bsim_path
        self.word_bits = word_bits
        self.reuse_mem = reuse_mem
        self.timeout = timeout
        if not os.path.exists(self.bsim_path):
            raise FileNotFoundError(
                f"bsim_cuda binary not found at {self.bsim_path}\n"
                "Build it: cd simulator/csim && mkdir -p build && cd build && cmake .. && make -j")

    def run(self, prog, inputs, params=None, input_arrays=None, bitlength=0,
            iterations=1, feedback=None, bsdata_path=None):
        cmd = [self.bsim_path, prog.source_path,
               f"--word-bits={self.word_bits}"]
        if self.reuse_mem:
            cmd.append("--reuse-mem")

        if bsdata_path:
            cmd.extend(["--input", bsdata_path])
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout,
                env=backend_env(),
            )
        else:
            input_json = build_input_json(inputs, params, input_arrays, bitlength,
                                          iterations, feedback)
            result = subprocess.run(
                cmd, input=json.dumps(input_json),
                capture_output=True, text=True, timeout=self.timeout,
                env=backend_env(),
            )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "out of memory" in stderr:
                raise MemoryError(f"CUDA out of memory: {stderr}")
            raise RuntimeError(f"bsim_cuda failed: {stderr}")
        output = json.loads(result.stdout)
        actual = _parse_csim_outputs(output["outputs"])
        return actual, output["op_count"], output.get("exec_ms", 0.0)

    def run_batch(self, prog, bsdata_paths):
        """Run program on each .bsdata file individually.

        bsim_cuda does not support multi-file batch mode (multiple --input
        flags concatenate vectors instead of producing per-file results).
        """
        results = []
        for p in bsdata_paths:
            cmd = [self.bsim_path, prog.source_path,
                   f"--word-bits={self.word_bits}"]
            if self.reuse_mem:
                cmd.append("--reuse-mem")
            cmd.extend(["--input", p])
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout,
                env=backend_env())
            if result.returncode != 0:
                stderr = result.stderr.strip()
                if "out of memory" in stderr:
                    raise MemoryError(f"CUDA out of memory: {stderr}")
                raise RuntimeError(f"bsim_cuda failed: {stderr}")
            output = json.loads(result.stdout)
            if isinstance(output, list):
                results.extend(output)
            else:
                results.append(output)
        return results



# ── Shared input JSON builder (used by all backends + profiling) ──


def build_input_json(inputs, params, input_arrays, bitlength,
                     iterations=1, feedback=None):
    """Build JSON input dict for bsim/bsim_cuda stdin."""
    d = {
        "inputs": _dict_to_json(inputs),
        "params": params or {},
        "n_vectors": bitlength,
    }
    if input_arrays:
        d["input_arrays"] = _arrays_to_json(input_arrays)
    if iterations > 1:
        d["iterations"] = iterations
    if feedback:
        d["feedback"] = feedback
    return d


# ── JSON serialization helpers for C++ backend ───────────────


def _value_to_json(v):
    """Convert a Python int to a JSON-safe representation (hex string if large).

    Unlike to_json_val(), also converts negative ints to hex for C++ backend.
    """
    if isinstance(v, int) and v < 0:
        return hex(v)
    return to_json_val(v)


def _dict_to_json(d: dict) -> dict:
    return {k: _value_to_json(v) for k, v in d.items()}


def _arrays_to_json(d: dict) -> dict:
    """Convert input_arrays {name: {idx: val}} to JSON-safe format."""
    result = {}
    for name, arr in d.items():
        if isinstance(arr, dict):
            max_idx = max(arr.keys()) if arr else -1
            lst = [_value_to_json(arr.get(i, 0)) for i in range(max_idx + 1)]
            result[name] = lst
        elif isinstance(arr, list):
            result[name] = [_value_to_json(v) for v in arr]
        else:
            result[name] = arr
    return result


def _parse_csim_outputs(outputs: dict) -> dict:
    """Parse C++ output values back to Python ints."""
    result = {}
    for k, v in outputs.items():
        if isinstance(v, str) and v.startswith("0x"):
            result[k] = int(v, 16)
        elif isinstance(v, dict):
            arr = {}
            for idx_str, val in v.items():
                idx = int(idx_str)
                if isinstance(val, str) and val.startswith("0x"):
                    arr[idx] = int(val, 16)
                else:
                    arr[idx] = val
            result[k] = arr
        else:
            result[k] = v
    return result


@dataclass
class BenchmarkResult:
    domain: str        # e.g. "circuit_sim"
    program: str       # e.g. "ripple_adder_8bit"
    test_name: str     # e.g. "Random 1000 vectors"
    bitlength: int     # bitstream length
    passed: bool
    n_stmts: int       # statement count
    op_count: int      # bitwise ops executed
    elapsed_s: float   # wall-clock seconds
    exec_ms: float = 0.0  # self-reported execution time (ms)


@dataclass
class ProgramInfo:
    name: str
    source_path: str
    program: Program
    n_stmts: int


# ── Precomputed test data helpers ──────────────────────────────


def _is_regenerated_case(case: dict) -> bool:
    """True if a test case's data is regenerated on demand, not committed.

    The synthetic small/medium/large tier datasets are git-ignored (too
    large to commit) and recreated by ``build.py --generate-only --tier``.
    Every such entry in ``tests.json`` carries a ``provenance`` block
    (generator command, RNG seed, SHA-256); committed unit fixtures and
    inline tests never do. A regenerated case whose ``.bsdata`` file is
    absent from a clean checkout is therefore a legitimate "not yet
    generated" state, not a broken checkout.
    """
    return "provenance" in case


def _parse_value(v):
    """Parse a single JSON value: hex strings -> int, pass through others."""
    if isinstance(v, str) and v.startswith("0x"):
        return int(v, 16)
    return v


def _parse_values(d: dict) -> dict:
    """Parse a dict of stream/int values (hex strings -> ints)."""
    return {k: _parse_value(v) for k, v in d.items()}


def _parse_arrays(d: dict) -> dict[str, dict[int, int]]:
    """Parse input_arrays: JSON lists -> {index: value} dicts."""
    result = {}
    for k, v in d.items():
        if isinstance(v, list):
            result[k] = {i: _parse_value(x) for i, x in enumerate(v)}
        elif isinstance(v, dict):
            result[k] = {int(i): _parse_value(x) for i, x in v.items()}
        else:
            result[k] = v
    return result


def _parse_expected(d: dict) -> dict:
    """Parse expected outputs: lists -> dicts, hex strings -> ints."""
    result = {}
    for k, v in d.items():
        if isinstance(v, list):
            result[k] = {i: _parse_value(x) for i, x in enumerate(v)}
        else:
            result[k] = _parse_value(v)
    return result


def load_test_case(case: dict, tests_dir: str | None = None):
    """Load a test case and return parsed (inputs, params, input_arrays, bitlength, iterations, feedback).

    Handles inline data, data_file (.json/.bsdata). If case has a data_file,
    tests_dir must be provided. Shared by test runner and profiling.
    """
    data = dict(case)

    # Load external data file if present
    data_file = data.get("data_file")
    if data_file and tests_dir:
        data_path = os.path.join(tests_dir, data_file)
        if data_path.endswith(".bsdata"):
            from benchmark.bsdata import read_bsdata
            file_data = read_bsdata(data_path)
        else:
            with open(data_path) as f:
                file_data = json.load(f)
        # Merge file data into case (file overrides case for shared keys)
        if isinstance(file_data, dict):
            data = {**data, **file_data}
            data.pop("data_file", None)

    inputs = _parse_values(data.get("inputs", {}))
    params = data.get("params", {})
    input_arrays = _parse_arrays(data.get("input_arrays", {}))
    # Support both "bitlength" (new) and "n_vectors" (legacy)
    bitlength = data.get("bitlength", data.get("n_vectors", 0))
    iterations = data.get("iterations", 1)
    feedback = data.get("feedback") or {}
    return inputs, params, input_arrays, bitlength, iterations, feedback


def _check_expected(result: dict, expected: dict,
                    mask: int | None = None,
                    int_outputs: frozenset = frozenset()) -> bool:
    """Check that interpreter result matches expected values.

    Stream outputs are masked to bitlength bits (handles NOT producing
    negative/unbounded Python ints). Int outputs (from popcount) are
    compared exactly.
    """
    for k, exp_v in expected.items():
        if k not in result:
            return False
        act_v = result[k]
        should_mask = mask is not None and k not in int_outputs
        if isinstance(exp_v, dict):
            # Array comparison (each element is a stream)
            if not isinstance(act_v, dict):
                return False
            for idx, val in exp_v.items():
                a = act_v.get(idx, 0)
                e = val
                if should_mask:
                    a &= mask
                    e &= mask
                if a != e:
                    return False
        else:
            # Scalar comparison
            a = act_v
            e = exp_v
            if should_mask:
                a &= mask
                e &= mask
            if a != e:
                return False
    return True


class BenchmarkDomain:
    """Abstract base for a benchmark domain."""

    @property
    def domain_name(self) -> str:
        raise NotImplementedError

    def discover_programs(self) -> list[ProgramInfo]:
        raise NotImplementedError

    def run_tests(self, prog: ProgramInfo) -> list[BenchmarkResult]:
        raise NotImplementedError

    def run_all(self) -> list[BenchmarkResult]:
        results = []
        for prog in self.discover_programs():
            results.extend(self.run_tests(prog))
        return results

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def load_bs(filepath: str) -> ProgramInfo:
        """Parse a .bs file."""
        with open(filepath) as f:
            source = f.read()
        program = parse(source)
        name = os.path.splitext(os.path.basename(filepath))[0]
        return ProgramInfo(
            name=name,
            source_path=filepath,
            program=program,
            n_stmts=count_stmts(program),
        )

    @staticmethod
    def timed_run(program: Program,
                  inputs: dict[str, int],
                  params: dict[str, int] | None = None,
                  input_arrays: dict[str, dict[int, int]] | None = None,
                  ) -> tuple[dict, Interpreter, float]:
        """Run a program and return (outputs, interpreter, elapsed_seconds)."""
        interp = Interpreter()
        t0 = time.perf_counter()
        result = interp.run(program, inputs, params, input_arrays)
        elapsed = time.perf_counter() - t0
        return result, interp, elapsed

    @staticmethod
    def timed_backend_run(backend: Backend, prog: 'ProgramInfo',
                          inputs: dict, params: dict | None = None,
                          input_arrays: dict | None = None,
                          bitlength: int = 0,
                          ) -> tuple[dict, int, float, float]:
        """Run via backend and return (outputs, op_count, elapsed_seconds, exec_ms)."""
        t0 = time.perf_counter()
        result, ops, exec_ms = backend.run(prog, inputs, params, input_arrays, bitlength)
        elapsed = time.perf_counter() - t0
        return result, ops, elapsed, exec_ms

    # ── Precomputed test runner ────────────────────────────────

    def _run_precomputed_tests(self, tests_json_path: str,
                               prog: ProgramInfo,
                               run_generated=None,
                               backend: Backend | None = None,
                               size_filter: str | None = None,
                               unit_only: bool = False,
                               ) -> list[BenchmarkResult]:
        """Generic test runner for precomputed bitstream-level test data.

        Dispatch order for each test case:
        1. If data_file exists on disk -> load it (.bsdata or .json)
        2. Else if category=="generated" and run_generated available -> fallback
        3. Else if inline inputs/expected -> run directly
        4. Else skip

        Args:
            tests_json_path: path to tests.json file
            prog: ProgramInfo
            run_generated: callable(case, prog, backend) -> (ok, ops) for generated tests
            backend: Backend instance (defaults to PythonBackend)

        Returns: list of BenchmarkResult
        """
        if backend is None:
            backend = PythonBackend()

        with open(tests_json_path) as f:
            data = json.load(f)

        if isinstance(data, list):
            cases = data
        elif isinstance(data, dict):
            cases = data.get(prog.name, data.get("_default", []))
        else:
            cases = []

        tests_dir = os.path.dirname(tests_json_path)

        # Unit-only filtering: keep only the self-contained unit
        # fixtures. A generated tier dataset carries a "provenance"
        # block (generator command, seed, SHA-256). Those .bsdata are
        # git-ignored and absent from a clean checkout. Inline tests and
        # the small committed fixtures have no provenance block, so this
        # selects exactly the inputs that ship in-tree.
        if unit_only:
            cases = [c for c in cases if not _is_regenerated_case(c)]

        # Size filtering: skip cases outside the requested byte-size range
        if size_filter:
            lo, hi = GenericDomain.SIZE_THRESHOLDS[size_filter]

            def _get_size(c):
                if "size_bytes" in c:
                    return c["size_bytes"]
                df = c.get("data_file", "")
                if df:
                    p = os.path.join(tests_dir, df)
                    return os.path.getsize(p) if os.path.exists(p) else 0
                return 0

            cases = [c for c in cases
                     if lo <= _get_size(c) <= (hi if hi is not None else float("inf"))]

        # ── Batch .bsdata execution for C++/CUDA backends ──
        # Group all .bsdata tests into one subprocess call per program
        _bsdata_results = {}  # case_index → (ok, ops, exec_ms, elapsed)
        if hasattr(backend, 'run_batch'):
            _bsdata_cases = []
            for i, case in enumerate(cases):
                df = case.get("data_file", "")
                if df and df.endswith(".bsdata"):
                    dp = os.path.join(tests_dir, df)
                    if os.path.exists(dp):
                        _bsdata_cases.append((i, case, dp))

            if _bsdata_cases:
                try:
                    from benchmark.bsdata import read_bsdata
                    t0 = time.perf_counter()
                    paths = [p for _, _, p in _bsdata_cases]
                    batch_out = backend.run_batch(prog, paths)
                    elapsed = time.perf_counter() - t0
                    per_test = elapsed / len(_bsdata_cases)

                    ridx = 0
                    int_outputs = frozenset(prog.program.output_int_names)
                    for ci, bcase, dp in _bsdata_cases:
                        fd = read_bsdata(dp)
                        cases_data = fd.get("cases", [fd])

                        all_ok = True
                        tot_ops = 0
                        tot_ms = 0.0
                        for cd in cases_data:
                            if ridx >= len(batch_out):
                                all_ok = False
                                break
                            br = batch_out[ridx]
                            ridx += 1
                            actual = _parse_csim_outputs(br["outputs"])
                            exp = _parse_expected(cd.get("expected", {}))
                            nv = cd.get("bitlength", cd.get("n_vectors", 0))
                            mask = (1 << nv) - 1 if nv > 0 else None
                            ok = _check_expected(actual, exp, mask, int_outputs)
                            tot_ops += br.get("op_count", 0)
                            tot_ms += br.get("exec_ms", 0.0)
                            if not ok:
                                all_ok = False
                                break

                        _bsdata_results[ci] = (all_ok, tot_ops, tot_ms, per_test)
                except (MemoryError, subprocess.TimeoutExpired, RuntimeError):
                    _bsdata_results = {}  # fall back to individual execution

        results = []
        for idx, case in enumerate(cases):
            # Check if already handled by batch execution
            if idx in _bsdata_results:
                ok, ops, exec_ms, elapsed = _bsdata_results[idx]
                results.append(BenchmarkResult(
                    domain=self.domain_name,
                    program=prog.name,
                    test_name=case["name"],
                    bitlength=case.get("bitlength", case.get("n_vectors", 0)),
                    passed=ok,
                    n_stmts=prog.n_stmts,
                    op_count=ops,
                    elapsed_s=elapsed,
                    exec_ms=exec_ms,
                ))
                continue
            optional = case.get("optional", False)
            try:
                # Check if data_file exists on disk
                data_file = case.get("data_file")
                data_file_exists = (data_file and
                                    os.path.exists(os.path.join(tests_dir, data_file)))

                if data_file_exists:
                    t0 = time.perf_counter()
                    rv = self._run_data_file(case, prog, tests_dir, backend)
                    elapsed = time.perf_counter() - t0
                elif case.get("category") == "generated":
                    if run_generated is None:
                        continue
                    t0 = time.perf_counter()
                    rv = run_generated(case, prog, backend)
                    elapsed = time.perf_counter() - t0
                elif "expected" in case or (data_file and not data_file_exists):
                    # Inline test or missing data_file (will error if no expected)
                    if data_file and not data_file_exists:
                        # A regenerated tier dataset or an optional real-data
                        # input is git-ignored and absent until build.py
                        # regenerates it -- skip it rather than failing the
                        # run. run_all.py prints an up-front notice listing
                        # what was skipped and how to generate it. A missing
                        # data_file with no provenance block is a committed
                        # fixture that should exist: that is a real error.
                        if optional or _is_regenerated_case(case):
                            continue
                        raise FileNotFoundError(
                            f"Committed data file not found: {data_file} "
                            f"in {tests_dir}. The repository checkout is "
                            f"incomplete or corrupted.")
                    t0 = time.perf_counter()
                    rv = self._run_precomputed(case, prog, backend)
                    elapsed = time.perf_counter() - t0
                else:
                    continue
            except (MemoryError, subprocess.TimeoutExpired, RuntimeError) as e:
                # GPU out of memory, timeout, or backend crash — skip generated/optional/data_file
                if optional or case.get("category") == "generated" or case.get("data_file"):
                    print(f"  SKIP ({type(e).__name__}): "
                          f"{prog.name}/{case.get('name', '?')}",
                          file=sys.stderr)
                    continue
                raise
            except (ImportError, FileNotFoundError):
                if optional:
                    continue
                raise
            if rv is None:
                continue
            # Unpack: rv is (ok, ops) or (ok, ops, exec_ms)
            if len(rv) == 3:
                ok, ops, exec_ms = rv
            else:
                ok, ops = rv
                exec_ms = 0.0
            results.append(BenchmarkResult(
                domain=self.domain_name,
                program=prog.name,
                test_name=case["name"],
                bitlength=case.get("bitlength", case.get("n_vectors", 0)),
                passed=ok,
                n_stmts=prog.n_stmts,
                op_count=ops,
                elapsed_s=elapsed,
                exec_ms=exec_ms,
            ))
        return results

    @staticmethod
    def _run_precomputed(case: dict, prog: ProgramInfo,
                         backend: Backend | None = None,
                         bsdata_path: str | None = None):
        """Run a single precomputed test case. Returns (ok, op_count, exec_ms)."""
        if backend is None:
            backend = PythonBackend()

        inputs, params, input_arrays, bitlength, n_iter, feedback = \
            load_test_case(case)
        expected = _parse_expected(case.get("expected", {}))

        result, total_ops, exec_ms = backend.run(
            prog, inputs, params, input_arrays,
            bitlength=bitlength, iterations=n_iter,
            feedback=feedback if feedback else None,
            bsdata_path=bsdata_path)

        if result is None:
            return False, 0, 0.0

        # Mask stream outputs to bitlength bits (handles NOT -> negative ints)
        mask = (1 << bitlength) - 1 if bitlength > 0 else None
        int_outputs = frozenset(prog.program.output_int_names)
        ok = _check_expected(result, expected, mask, int_outputs)
        return ok, total_ops, exec_ms

    @staticmethod
    def _run_data_file(case: dict, prog: ProgramInfo, tests_dir: str,
                       backend: Backend | None = None):
        """Run a test with data from a separate file. Returns (ok, ops, exec_ms).

        Supports .bsdata (binary) and .json (legacy) formats.
        """
        data_path = os.path.join(tests_dir, case["data_file"])

        if data_path.endswith(".bsdata"):
            return BenchmarkDomain._run_bsdata_file(case, prog, data_path, backend)

        with open(data_path) as f:
            file_data = json.load(f)

        if isinstance(file_data, list):
            # List of sub-cases: run all, pass if all pass
            all_ok = True
            total_ops = 0
            total_exec_ms = 0.0
            for sub in file_data:
                merged = {**case, **sub}
                merged.pop("data_file", None)
                # Sub-cases don't inherit bitlength from the outer case;
                # each sub-case is an independent run with its own stream width.
                if "bitlength" not in sub and "n_vectors" not in sub:
                    merged.pop("bitlength", None)
                    merged.pop("n_vectors", None)
                ok, ops, exec_ms = BenchmarkDomain._run_precomputed(merged, prog, backend)
                total_ops += ops
                total_exec_ms += exec_ms
                if not ok:
                    all_ok = False
                    break
            return all_ok, total_ops, total_exec_ms
        else:
            # Single data dict with inputs/expected
            merged = {**case, **file_data}
            merged.pop("data_file", None)
            return BenchmarkDomain._run_precomputed(merged, prog, backend)

    @staticmethod
    def _run_bsdata_file(case: dict, prog: ProgramInfo, data_path: str,
                         backend: Backend | None = None):
        """Run a test from a .bsdata binary file. Returns (ok, ops, exec_ms)."""
        from benchmark.bsdata import read_bsdata
        file_data = read_bsdata(data_path)

        # C++/CUDA backends: pass file directly, binary handles multi-case natively
        if hasattr(backend, 'run_batch'):
            batch_results = backend.run_batch(prog, [data_path])

            if "cases" in file_data:
                cases_data = file_data["cases"]
            else:
                cases_data = [file_data]

            all_ok = True
            total_ops = 0
            total_exec_ms = 0.0
            int_outputs = frozenset(prog.program.output_int_names)

            for case_data, result in zip(cases_data, batch_results):
                actual = _parse_csim_outputs(result["outputs"])
                expected = _parse_expected(case_data.get("expected", {}))
                bitlength = case_data.get("bitlength", case_data.get("n_vectors", 0))
                mask = (1 << bitlength) - 1 if bitlength > 0 else None
                ok = _check_expected(actual, expected, mask, int_outputs)
                total_ops += result.get("op_count", 0)
                total_exec_ms += result.get("exec_ms", 0.0)
                if not ok:
                    all_ok = False
                    break

            return all_ok, total_ops, total_exec_ms

        # Python backend: iterate Python-side
        if "cases" in file_data:
            all_ok = True
            total_ops = 0
            total_exec_ms = 0.0
            for entry in file_data["cases"]:
                merged = {**case, **entry}
                merged.pop("data_file", None)
                merged.pop("cases", None)
                ok, ops, exec_ms = BenchmarkDomain._run_precomputed(merged, prog, backend)
                total_ops += ops
                total_exec_ms += exec_ms
                if not ok:
                    all_ok = False
                    break
            return all_ok, total_ops, total_exec_ms
        else:
            merged = {**case, **file_data}
            merged.pop("data_file", None)
            return BenchmarkDomain._run_precomputed(merged, prog, backend)


class GenericDomain(BenchmarkDomain):
    """Generic domain that auto-discovers .bs file and tests from standard layout.

    Standard layout:
        benchmark/apps/<name>/src/<name>.bs
        benchmark/apps/<name>/datasets/tests/tests.json
        benchmark/apps/<name>/src/run.py  (optional: run_generated callback)
    """

    # Size thresholds for --size filtering (by .bsdata file size in bytes)
    SIZE_THRESHOLDS = {
        "small":  (25_000_000,     250_000_000),   # 25-250MB  (~50MB target)
        "medium": (250_000_001,  2_500_000_000),   # 250MB-2.5GB (~500MB target)
        "large":  (2_500_000_001,       None),     # > 2.5GB   (~5GB target)
    }

    def __init__(self, name: str, backend: Backend | None = None,
                 size_filter: str | None = None,
                 apps_dir: str | None = None,
                 module_prefix: str = "benchmark.apps",
                 unit_only: bool = False):
        self._name = name
        self._backend = backend
        self._size_filter = size_filter
        self._unit_only = unit_only
        self._apps_dir = apps_dir or os.path.join(os.path.dirname(__file__), "apps")
        self._module_prefix = module_prefix

    @property
    def domain_name(self) -> str:
        return self._name

    def discover_programs(self) -> list[ProgramInfo]:
        src_dir = os.path.join(self._apps_dir, self._name, "src")
        # Find all .bs files in src/
        bs_files = sorted(
            f for f in os.listdir(src_dir) if f.endswith(".bs"))
        return [self.load_bs(os.path.join(src_dir, f)) for f in bs_files]

    def run_tests(self, prog: ProgramInfo) -> list[BenchmarkResult]:
        tests_json = os.path.join(
            self._apps_dir, self._name, "datasets", "tests", "tests.json")
        return self._run_precomputed_tests(
            tests_json, prog,
            run_generated=self._get_run_generated(),
            backend=self._backend,
            size_filter=self._size_filter,
            unit_only=self._unit_only)

    def _get_run_generated(self):
        """Try to import run_generated from the domain's run.py."""
        import importlib
        try:
            mod = importlib.import_module(
                f"{self._module_prefix}.{self._name}.src.run")
            return getattr(mod, "run_generated", None)
        except (ImportError, ModuleNotFoundError):
            return None
