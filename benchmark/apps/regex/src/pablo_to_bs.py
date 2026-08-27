#!/usr/bin/env python3
"""Convert icgrep Pablo IR output to .bs (3-address bitstream) programs.

Pipeline: regex -> icgrep --ShowPablo -> Pablo IR -> pablo_to_bs.py -> .bs

Handles both input modes produced by icgrep:
  - Packed:   <i8>[1] UTF8_basis -- byte-level operations, PackL/PackH
  - Unpacked: <i1>[8] UTF8_basis -- direct bit-plane operations

Supported Pablo IR operations:
  Advance(x, 1)           -> x << 1
  MatchStar(m, c)         -> ((m & c) + c) ^ c | m  (4 ops)
  Sel(c, t, f)            -> (t & c) | (f & ~c)     (4 ops)
  ScanThru(f, t)          -> (f + t) & ~t            (3 ops)
  AdvanceThenScanTo(x, y) -> x << 1 (when y = ONES for byte-level)
  IndexedAdvance(m, i, n) -> n steps of shift+ScanTo (ONES idx = shift)
  PackL/PackH             -> resolve to individual basis bits
  InFile(expr)            -> identity
  <0> -> ZERO,  <1> -> ONES
  Bitwise: &, |, ^, ~
"""

from __future__ import annotations

import contextlib
import re
import shutil
import subprocess
import os
import sys


# Locally built icgrep (Parabix). Point ICGREP_BIN at the binary, or put
# icgrep on PATH.
ICGREP_BIN = os.environ.get("ICGREP_BIN") or shutil.which("icgrep") or ""

# Pre-processing: expand Unicode-aware shortcuts to ASCII equivalents
# so icgrep produces self-contained regex kernels (no Unicode property lookups).
_SHORTHAND_MAP = {
    r"\d": r"[0-9]",
    r"\D": r"[^0-9]",
    r"\w": r"[A-Za-z0-9_]",
    r"\W": r"[^A-Za-z0-9_]",
    r"\s": r"[\x09\x0a\x0d\x20]",
    r"\S": r"[^\x09\x0a\x0d\x20]",
}


def preprocess_pattern(pattern: str, flags: str = "") -> str:
    """Expand shorthand classes and case-insensitive flag for icgrep.

    Replaces \\d, \\w, \\s with ASCII equivalents so icgrep doesn't
    pull in Unicode property kernels.  For case-insensitive patterns,
    expand literal letters to [Aa] form.  Strips anchors (^, $) since
    the bitstream model is anchor-free.

    Note on PCRE flags: ``/s`` (DOTALL) and ``/m`` (multiline) are
    **not** honoured.  We deliberately follow icgrep's semantics,
    which treats ``.`` as "not newline" regardless of the PCRE flag.
    Patterns whose PCRE source uses ``/s`` with a bare ``.`` therefore
    have a different match set in the bitstream model than in PCRE;
    the spot-check filters those patterns out of the Python ``re``
    reference rather than forcing the .bs translation to mirror PCRE.
    """
    # Expand shorthand classes
    result = pattern
    for short, expanded in _SHORTHAND_MAP.items():
        result = result.replace(short, expanded)

    # Remove ^ at start (icgrep handles it but cleaner to strip).
    # Leave $ anchors intact — icgrep maps them to AtEOF, which
    # pablo_to_bs converts to ONES for the anchor-free model.
    result = re.sub(r"^\^", "", result)

    # Case-insensitive: expand bare ASCII letters outside char classes
    if "i" in flags:
        result = _expand_case_insensitive(result)

    return result


def _expand_case_insensitive(pattern: str) -> str:
    """Expand bare letters to [Aa] form for case-insensitive matching."""
    out = []
    i = 0
    in_class = False
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            # Escape sequence: pass through as-is
            out.append(pattern[i : i + 2])
            # Handle \\xHH
            if pattern[i + 1] == "x" and i + 3 < len(pattern):
                out.pop()
                out.append(pattern[i : i + 4])
                i += 4
            else:
                i += 2
            continue
        if ch == "[":
            in_class = True
            out.append(ch)
            i += 1
            continue
        if ch == "]":
            in_class = False
            out.append(ch)
            i += 1
            continue
        if not in_class and ch.isalpha():
            lo, hi = ch.lower(), ch.upper()
            if lo != hi:
                out.append(f"[{hi}{lo}]")
            else:
                out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Pablo IR parser and converter
# ---------------------------------------------------------------------------


class PabloToBs:
    """Convert a single icgrep Pablo kernel to .bs 3-address code."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.lines: list[str] = []          # emitted .bs lines
        self.var_map: dict[str, str] = {}   # pablo var -> .bs var
        self.repeat_vals: dict[str, int] = {}  # repeat var -> int constant
        self.pack_bits: dict[str, list[int] | int] = {}  # pack var -> bits
        self.temp_count = 0
        self.neg_cache: dict[int, str] = {}  # bit idx -> negated .bs var
        self.input_mode: str | None = None  # 'packed' or 'unpacked'
        self.byte_eq_cache: dict[int, str] = {}  # byte val -> .bs var
        self.byte_lt_cache: dict[int, str] = {}  # byte val -> .bs var
        self.advance_input: dict[str, str] = {}  # adv result -> adv input
        # Declarations hoisted ahead of the body.  The constant, negated-basis
        # and byte-comparison caches are keyed for the whole kernel but emitted
        # lazily on first use.  Inside an if/while body that first use may not
        # execute, which would leave later readers referencing an undefined
        # stream, so those declarations go here instead.  They depend only on
        # the basis bits, so hoisting them is always sound.
        self.preamble: list[str] = []
        self._hoist_depth = 0
        self.indent = 0
        # Pablo Vars: names assigned inside an if/while body that are also
        # live outside it.  Each maps to one stable, mutable .bs variable
        # rather than to a fresh temporary per assignment.
        self.mutable: set[str] = set()
        self.mutable_bs: dict[str, str] = {}
        self.mutable_declared: set[str] = set()

    def fresh(self, hint: str = "t") -> str:
        self.temp_count += 1
        return f"_{self.prefix}{hint}{self.temp_count}"

    def emit(self, line: str):
        # Block bodies are not indented. The braces carry the structure, and
        # icgrep nests `if` blocks up to 39 deep on the Snort patterns, so
        # indenting regex_large would spend 44 MB of the file on leading
        # spaces and push it past the 100 MB limit GitHub enforces.
        if self._hoist_depth:
            self.preamble.append(line)
        else:
            self.lines.append(line)

    @contextlib.contextmanager
    def _hoisted(self):
        """Emit into the preamble instead of the current block."""
        self._hoist_depth += 1
        try:
            yield
        finally:
            self._hoist_depth -= 1

    def get_var(self, pablo_name: str) -> str:
        """Get .bs variable name for a Pablo variable."""
        if pablo_name in self.var_map:
            return self.var_map[pablo_name]
        # Constants <0> and <1>
        if pablo_name == "<0>":
            v = self.fresh("z")
            with self._hoisted():
                self.emit(f"stream {v} = ZERO")
            self.set_var(pablo_name, v)
            return v
        if pablo_name == "<1>":
            v = self.fresh("o")
            with self._hoisted():
                self.emit(f"stream {v} = ONES")
            self.set_var(pablo_name, v)
            return v
        # Check if it's a basis bit reference (unpacked mode)
        m = re.match(r"UTF8_basis\[(\d+)\]", pablo_name)
        if m:
            return f"b[{m.group(1)}]"
        raise KeyError(f"Unknown Pablo variable: {pablo_name}")

    def set_var(self, pablo_name: str, bs_name: str):
        """Map a Pablo variable to a .bs variable."""
        self.var_map[pablo_name] = bs_name

    def get_neg_bit(self, bit: int) -> str:
        """Get or create ~b[bit] with caching."""
        if bit not in self.neg_cache:
            name = f"_{self.prefix}nb{bit}"
            with self._hoisted():
                self.emit(f"stream {name} = ~b[{bit}]")
            self.neg_cache[bit] = name
        return self.neg_cache[bit]

    # -- Packed mode: byte-level expansions --

    def expand_byte_eq(self, byte_val: int) -> str:
        """Expand byte == N to Boolean formula over b[0..7].

        Returns .bs variable name for the equality stream.
        """
        if byte_val in self.byte_eq_cache:
            return self.byte_eq_cache[byte_val]

        # AND of (b[i] if bit set, else ~b[i]) for i in 0..7
        terms = []
        for i in range(8):
            if (byte_val >> i) & 1:
                terms.append(f"b[{i}]")
            else:
                terms.append(self.get_neg_bit(i))

        # Build AND tree in 3-address form
        result = terms[0]
        with self._hoisted():
            for t in terms[1:]:
                new = self.fresh("eq")
                self.emit(f"stream {new} = {result} & {t}")
                result = new

        self.byte_eq_cache[byte_val] = result
        return result

    def expand_byte_lt(self, byte_val: int) -> str:
        """Expand byte < N to comparator circuit over b[0..7].

        Returns .bs variable name for the less-than stream.
        Uses magnitude comparator: scan MSB to LSB, track 'less so far'.
        """
        if byte_val in self.byte_lt_cache:
            return self.byte_lt_cache[byte_val]

        with self._hoisted():
            if byte_val == 0:
                # Nothing is less than 0
                v = self.fresh("lt")
                self.emit(f"stream {v} = ZERO")
                self.byte_lt_cache[byte_val] = v
                return v

            if byte_val == 256:
                # Everything is less than 256 (unsigned 8-bit)
                v = self.fresh("lt")
                self.emit(f"stream {v} = ONES")
                self.byte_lt_cache[byte_val] = v
                return v

            # Magnitude comparator: MSB to LSB
            # lt = "definitely less than N at some higher bit"
            # eq = "equal to N at all higher bits so far"
            lt_var = self.fresh("lt")
            self.emit(f"stream {lt_var} = ZERO")
            eq_var = self.fresh("eq")
            self.emit(f"stream {eq_var} = ONES")

            for i in range(7, -1, -1):
                n_bit = (byte_val >> i) & 1
                if n_bit == 1:
                    # N's bit is 1: a[i]=0 means a < N at this position
                    neg_b = self.get_neg_bit(i)
                    new_lt_term = self.fresh("lt")
                    self.emit(f"stream {new_lt_term} = {eq_var} & {neg_b}")
                    new_lt = self.fresh("lt")
                    self.emit(f"stream {new_lt} = {lt_var} | {new_lt_term}")
                    lt_var = new_lt
                    # Still equal only if a[i]=1
                    new_eq = self.fresh("eq")
                    self.emit(f"stream {new_eq} = {eq_var} & b[{i}]")
                    eq_var = new_eq
                else:
                    # N's bit is 0: a[i] must be 0 to remain equal
                    neg_b = self.get_neg_bit(i)
                    new_eq = self.fresh("eq")
                    self.emit(f"stream {new_eq} = {eq_var} & {neg_b}")
                    eq_var = new_eq
                    # lt unchanged (a[i]=1 means a > N, not less)

        self.byte_lt_cache[byte_val] = lt_var
        return lt_var

    def resolve_pack(self, pablo_var: str) -> list[int] | int:
        """Resolve a Pack variable to bit index(es).

        Returns either a single int (for scalar) or list of ints (for array).
        """
        if pablo_var in self.pack_bits:
            return self.pack_bits[pablo_var]
        raise KeyError(f"Unknown pack variable: {pablo_var}")

    # -- Kernel extraction --

    @staticmethod
    def extract_regex_kernel(icgrep_output: str) -> str | None:
        """Extract the regex-matching kernel (named ic<hash>) from output.

        Returns the kernel body text, or None if not found.
        """
        lines = icgrep_output.split("\n")
        in_kernel = False
        kernel_lines = []
        brace_depth = 0

        for line in lines:
            stripped = line.strip()

            if not in_kernel:
                # Look for kernel ic<hash>
                if re.match(r"kernel\s+ic[0-9a-f]+\s*::", stripped):
                    in_kernel = True
                    kernel_lines.append(stripped)
                    brace_depth = stripped.count("{") - stripped.count("}")
                    continue
            else:
                kernel_lines.append(stripped)
                brace_depth += stripped.count("{") - stripped.count("}")
                if brace_depth <= 0:
                    break

        if not kernel_lines:
            return None
        return "\n".join(kernel_lines)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize a Pablo variable name to be a valid .bs identifier.

        Pablo IR uses names like UTF8_a,d and Byte_22,27 which contain
        commas.  Replace non-identifier characters with underscores.
        """
        return re.sub(r"[^A-Za-z0-9_\[\]]", "_", name)

    # -- Main conversion --

    def convert_kernel(self, kernel_text: str) -> list[str]:
        """Convert a Pablo kernel to .bs statements.

        Returns list of .bs source lines (without input/output declarations).
        """
        # Sanitize all variable names in the kernel text
        # Replace comma-containing identifiers like UTF8_a,d -> UTF8_a_d
        kernel_text = re.sub(
            r"\b([A-Za-z_]\w*(?:,\w+)+)\b",
            lambda m: self._sanitize_name(m.group(0)),
            kernel_text,
        )

        lines = kernel_text.split("\n")

        # Parse header to determine input mode
        header = lines[0]
        if "<i8>[1] UTF8_basis" in header:
            self.input_mode = "packed"
        elif "<i1>[8] UTF8_basis" in header:
            self.input_mode = "unpacked"
        else:
            # Try to detect from the body
            self.input_mode = "packed"

        # Check if kernel uses mIndexing
        uses_mindexing = "mIndexing" in kernel_text

        # Map mIndexing[0] -> ONES if present
        if uses_mindexing:
            ones_var = self.fresh("ones")
            with self._hoisted():
                self.emit(f"stream {ones_var} = ONES")
            self.var_map["mIndexing[0]"] = ones_var

        # Process body lines (skip header; keep braces for block structure)
        body_lines = []
        for line in lines[1:]:
            stripped = line.strip()
            if stripped == "":
                continue
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            body_lines.append(stripped)

        # Drop the kernel's own closing brace so it does not close a block.
        if body_lines and body_lines[-1] == "}":
            body_lines.pop()

        blocks = self._parse_blocks(body_lines)
        self.mutable = self._collect_mutable_vars(body_lines)
        self._emit_blocks(blocks)

        return self.preamble + self.lines

    # -- Block structure --

    @staticmethod
    def _parse_blocks(lines: list[str]) -> list:
        """Group flat Pablo lines into a nested block structure.

        Each returned item is either a statement string or a tuple
        ``(kind, cond_name, body)`` with ``kind`` in ``{"if", "while"}``.
        """
        def build(i: int, out: list) -> int:
            while i < len(lines):
                line = lines[i]
                m = re.match(r"(if|while)\s+(\S+)\s*\{$", line)
                if m:
                    body: list = []
                    i = build(i + 1, body)
                    out.append((m.group(1), m.group(2), body))
                    continue
                if line == "}":
                    return i + 1
                out.append(line)
                i += 1
            return i

        top: list = []
        build(0, top)
        return top

    @staticmethod
    def _collect_mutable_vars(lines: list[str]) -> set[str]:
        """Pablo names whose assignment inside a block escapes that block.

        Pablo declares such names as ``Var``s in an enclosing scope, so they
        must map to one stable, mutable .bs variable: a fresh temporary per
        assignment would be undefined on the path where the block does not
        run.  A name qualifies when it is assigned at a deeper nesting level
        than some place it appears, which is exactly the ``Var`` pattern
        (``m = <0>`` outside, ``m = and_3`` inside).  Names that live entirely
        within one block stay fresh temporaries, so no copy is emitted for
        them.
        """
        min_appear: dict[str, int] = {}
        max_assign: dict[str, int] = {}
        depth = 0
        for line in lines:
            m_block = re.match(r"(?:if|while)\s+(\S+)\s*\{$", line)
            if m_block:
                name = m_block.group(1)
                min_appear[name] = min(min_appear.get(name, depth), depth)
                depth += 1
                continue
            if line == "}":
                depth -= 1
                continue
            for name in re.findall(r"[A-Za-z_]\w*", line):
                min_appear[name] = min(min_appear.get(name, depth), depth)
            m_assign = re.match(r"([A-Za-z_]\w*)\s*\|?=", line)
            if m_assign:
                name = m_assign.group(1)
                max_assign[name] = max(max_assign.get(name, depth), depth)
        return {n for n, d in max_assign.items()
                if d > min_appear.get(n, d)}

    def _emit_blocks(self, blocks: list):
        """Walk the block tree, emitting .bs statements."""
        for item in blocks:
            if isinstance(item, tuple):
                kind, cond_name, body = item
                cond_var = self.get_var(cond_name)
                self.emit(f"{kind} ({cond_var}) {{")
                self.indent += 1
                self._emit_blocks(body)
                self.indent -= 1
                self.emit("}")
            else:
                self._process_line(item)

    def _mutable_name(self, pablo_name: str) -> str:
        """Stable .bs name for a Pablo Var."""
        if pablo_name not in self.mutable_bs:
            self.mutable_bs[pablo_name] = (
                f"_{self.prefix}v_{self._sanitize_name(pablo_name)}")
        return self.mutable_bs[pablo_name]

    def _process_line(self, line: str):
        """Process a single Pablo IR line."""
        # Skip TerminateAt (not relevant for bitstream model)
        if "TerminateAt(" in line:
            return

        # AtEOF(x) → ONES in our anchor-free model
        # (any position could be at end of file)
        # Must be handled as assignment, not skipped, since the variable is used later

        # Output assignment: matches[0] = var
        m = re.match(r"(\w+)\[(\d+)\]\s*=\s*(\w+)", line)
        if m:
            array_name, _idx, rhs_var = m.group(1), m.group(2), m.group(3)
            if array_name == "matches":
                output_var = self.get_var(rhs_var)
                # Strip trailing Advance: icgrep marks position AFTER
                # the match end, but our convention marks the last byte.
                if output_var in self.advance_input:
                    output_var = self.advance_input[output_var]
                self.var_map["__output__"] = output_var
            return

        # Parse assignment: var = rhs
        # Handle the weird Pack double-equals: "packl =  = PackL(...)"
        m_pack_dbl = re.match(r"(\w+)\s*=\s*=\s*(Pack[LH]\(.+\))", line)
        if m_pack_dbl:
            lhs = m_pack_dbl.group(1)
            rhs = m_pack_dbl.group(2)
            self._handle_pack(lhs, rhs)
            return

        # Regular assignment
        m_assign = re.match(r"(\w+(?:\[\d+\])?)\s*=\s*(.+)", line)
        if not m_assign:
            return

        lhs = m_assign.group(1).strip()
        rhs = m_assign.group(2).strip()

        # Compound assignment: var |= expr -> var = var | expr
        if "=" in lhs:
            return  # shouldn't happen with our regex

        m_compound = re.match(r"(\w+)\s*\|=\s*(.+)", line)
        if m_compound:
            lhs = m_compound.group(1)
            rhs = f"{lhs} | {m_compound.group(2)}"

        self._handle_assignment(lhs, rhs)

    def _handle_assignment(self, lhs: str, rhs: str):
        """Handle a var = expr assignment.

        For a Pablo Var (see :meth:`_collect_mutable_vars`) the value is
        computed into whatever the normal path produces, then copied into the
        Var's stable .bs name so that readers outside the block see the right
        stream on every path.  A copy is a primary RHS, which costs no
        operation and no dependency depth.
        """
        if lhs not in self.mutable:
            self._handle_assignment_inner(lhs, rhs)
            return

        self._handle_assignment_inner(lhs, rhs)
        result = self.var_map.get(lhs)
        if result is None:
            # Repeat() and Pack() bind side tables rather than var_map;
            # icgrep never assigns those inside a block.
            return
        name = self._mutable_name(lhs)
        if result != name:
            if name not in self.mutable_declared:
                if self.indent:
                    # First assignment sits inside a block, so the declaration
                    # has to be unconditional.  Pablo Vars are zero-initialised,
                    # which is what a reader sees when the block does not run.
                    with self._hoisted():
                        self.emit(f"stream {name} = ZERO")
                    self.mutable_declared.add(name)
                    self.emit(f"{name} = {result}")
                else:
                    self.emit(f"stream {name} = {result}")
                    self.mutable_declared.add(name)
            else:
                self.emit(f"{name} = {result}")
        self.set_var(lhs, name)

    def _handle_assignment_inner(self, lhs: str, rhs: str):
        """Handle a var = expr assignment."""
        rhs = rhs.strip()

        # Constants
        if rhs == "<0>":
            v = self.fresh("z")
            self.emit(f"stream {v} = ZERO")
            self.set_var(lhs, v)
            return

        if rhs == "<1>":
            v = self.fresh("o")
            self.emit(f"stream {v} = ONES")
            self.set_var(lhs, v)
            return

        # Repeat constant
        m = re.match(r"Repeat\(8,\s*Int8\((\d+)\)\)", rhs)
        if m:
            self.repeat_vals[lhs] = int(m.group(1))
            return

        # InFile(expr) - evaluate the inner expression
        m = re.match(r"InFile\((.+)\)", rhs)
        if m:
            inner = m.group(1).strip()
            result = self._eval_expr(inner)
            self.set_var(lhs, result)
            return

        # AtEOF(x) → ONES in anchor-free model
        m = re.match(r"AtEOF\((.+)\)", rhs)
        if m:
            v = self.fresh("eof")
            self.emit(f"stream {v} = ONES")
            self.set_var(lhs, v)
            return

        # Advance(x, N) - x can be a variable or constant like <1>
        m = re.match(r"Advance\(([^,]+),\s*(\d+)\)", rhs)
        if m:
            arg_name = m.group(1).strip()
            shift_n = int(m.group(2))
            arg_var = self.get_var(arg_name)
            v = self.fresh("adv")
            self.emit(f"stream {v} = {arg_var} << {shift_n}")
            self.advance_input[v] = arg_var
            self.set_var(lhs, v)
            return

        # MatchStar(m, c)
        m = re.match(r"MatchStar\(([^,]+),\s*([^)]+)\)", rhs)
        if m:
            m_arg = self.get_var(m.group(1).strip())
            c_arg = self.get_var(m.group(2).strip())
            result = self._emit_matchstar(m_arg, c_arg)
            self.set_var(lhs, result)
            return

        # Sel(c, t, f)
        m_sel = re.match(r"Sel\(([^,]+),\s*([^,]+),\s*([^)]+)\)", rhs)
        if m_sel:
            c_arg = self.get_var(m_sel.group(1).strip())
            t_arg = self.get_var(m_sel.group(2).strip())
            f_arg = self.get_var(m_sel.group(3).strip())
            result = self._emit_sel(c_arg, t_arg, f_arg)
            self.set_var(lhs, result)
            return

        # ScanThru(f, t)
        m_st = re.match(r"ScanThru\(([^,]+),\s*([^)]+)\)", rhs)
        if m_st:
            f_arg = self.get_var(m_st.group(1).strip())
            t_arg = self.get_var(m_st.group(2).strip())
            result = self._emit_scanthru(f_arg, t_arg)
            self.set_var(lhs, result)
            return

        # ScanTo(x, target) = ScanThru(x, ~target)
        # When target = mIndexing[0] (= ONES), ScanTo(x, ONES) = x (no-op)
        m_sto = re.match(r"ScanTo\(([^,]+),\s*([^)]+)\)", rhs)
        if m_sto:
            x_arg = self.get_var(m_sto.group(1).strip())
            y_name = m_sto.group(2).strip()
            if y_name == "mIndexing[0]" or (
                y_name in self.var_map
                and "ones" in self.var_map.get(y_name, "").lower()
            ):
                # ScanTo(x, ONES) = x (already at valid position)
                self.set_var(lhs, x_arg)
            else:
                y_arg = self.get_var(y_name)
                not_y = self.fresh("nt")
                self.emit(f"stream {not_y} = ~{y_arg}")
                result = self._emit_scanthru(x_arg, not_y)
                v = self.fresh("sto")
                self.emit(f"stream {v} = {result} & {y_arg}")
                self.set_var(lhs, v)
            return

        # AdvanceThenScanTo(x, y) - for byte mode (y=ONES), just Advance
        m_ast = re.match(
            r"AdvanceThenScanTo\(([^,]+),\s*([^)]+)\)", rhs
        )
        if m_ast:
            x_arg = self.get_var(m_ast.group(1).strip())
            y_name = m_ast.group(2).strip()
            # If y is mIndexing[0] (= ONES), just advance
            if y_name == "mIndexing[0]" or (
                y_name in self.var_map
                and "ones" in self.var_map.get(y_name, "").lower()
            ):
                v = self.fresh("adv")
                self.emit(f"stream {v} = {x_arg} << 1")
                self.advance_input[v] = x_arg
                self.set_var(lhs, v)
            else:
                # General case: Advance then ScanTo
                y_arg = self.get_var(y_name)
                adv = self.fresh("adv")
                self.emit(f"stream {adv} = {x_arg} << 1")
                # ScanTo(adv, y) = ScanThru(adv, ~y)
                not_y = self.fresh("nt")
                self.emit(f"stream {not_y} = ~{y_arg}")
                result = self._emit_scanthru(adv, not_y)
                v2 = self.fresh("ast")
                self.emit(f"stream {v2} = {result} & {y_arg}")
                self.set_var(lhs, v2)
            return

        # IndexedAdvance(marker, index, n)
        m_ia = re.match(r"IndexedAdvance\(([^,]+),\s*([^,]+),\s*(\-?\d+)\)", rhs)
        if m_ia:
            marker_arg = self.get_var(m_ia.group(1).strip())
            idx_name = m_ia.group(2).strip()
            n = int(m_ia.group(3))
            if n < 0:
                # Negative IndexedAdvance: emit ZERO (only 1 pattern uses this)
                v = self.fresh("iaz")
                self.emit(f"stream {v} = ZERO")
                self.set_var(lhs, v)
            elif idx_name == "mIndexing[0]" or (
                idx_name in self.var_map
                and "ones" in self.var_map.get(idx_name, "").lower()
            ):
                # ONES index: every position is valid, just shift
                v = self.fresh("adv")
                self.emit(f"stream {v} = {marker_arg} << {n}")
                self.set_var(lhs, v)
            else:
                # General case: iterate n ScanTo steps
                idx_arg = self.get_var(idx_name)
                result = self._emit_indexed_advance(marker_arg, idx_arg, n)
                self.set_var(lhs, result)
            return

        # PackL/PackH (without double equals - shouldn't normally happen)
        m_pack = re.match(r"Pack([LH])\((\d+),\s*(\w+(?:\[\d+\])?)\)", rhs)
        if m_pack:
            self._handle_pack(lhs, rhs)
            return

        # General expression (binary ops, unary, variable references)
        result = self._eval_expr(rhs)
        self.set_var(lhs, result)

    def _eval_expr(self, expr: str) -> str:
        """Evaluate a Pablo expression and return .bs variable name.

        Handles: variable refs, ~expr, binary ops, comparisons.
        """
        expr = expr.strip()

        # Constants <0>, <1>
        if expr == "<0>" or expr == "<1>":
            return self.get_var(expr)

        # Simple variable reference
        if re.match(r"^\w+(?:\[\d+\])?$", expr):
            return self.get_var(expr)

        # Negation: ~expr
        if expr.startswith("~"):
            inner = expr[1:].strip()

            # ~(comparison): ~array < repeat_var
            m_cmp = re.match(r"(\w+(?:\[\d+\])?)\s*<\s*(\w+)", inner)
            if m_cmp and m_cmp.group(2) in self.repeat_vals:
                lt_var = self._eval_comparison(
                    m_cmp.group(1), "<", m_cmp.group(2)
                )
                v = self.fresh("nt")
                self.emit(f"stream {v} = ~{lt_var}")
                return v

            inner_var = self._eval_expr(inner)
            v = self.fresh("nt")
            self.emit(f"stream {v} = ~{inner_var}")
            return v

        # Binary with comparison: expr1 < expr2 & expr3
        # Comparison binds tighter than bitwise
        m_cmp_bin = re.match(
            r"(\w+(?:\[\d+\])?)\s*([<>=!]+)\s*(\w+)\s*([&|^])\s*(.+)", expr
        )
        if m_cmp_bin:
            cmp_lhs = m_cmp_bin.group(1)
            cmp_op = m_cmp_bin.group(2)
            cmp_rhs = m_cmp_bin.group(3)
            bin_op = m_cmp_bin.group(4)
            rest = m_cmp_bin.group(5).strip()

            cmp_result = self._eval_comparison(cmp_lhs, cmp_op, cmp_rhs)
            rest_var = self._eval_expr(rest)
            v = self.fresh("t")
            self.emit(f"stream {v} = {cmp_result} {bin_op} {rest_var}")
            return v

        # Standalone comparison: expr1 == expr2, expr1 < expr2
        m_cmp = re.match(r"(\w+(?:\[\d+\])?)\s*([<>=!]+)\s*(\w+(?:\[\d+\])?)", expr)
        if m_cmp and m_cmp.group(3) in self.repeat_vals:
            return self._eval_comparison(
                m_cmp.group(1), m_cmp.group(2), m_cmp.group(3)
            )

        # Binary operation: a & b, a | b, a ^ b
        # Find the operator, splitting on the LAST occurrence to handle chaining
        for op in ("&", "|", "^"):
            # Split on the operator, respecting that variable names can
            # contain underscores but operators are surrounded by spaces
            parts = re.split(rf"\s+\{op}\s+", expr, maxsplit=1)
            if len(parts) == 2:
                left_var = self._eval_expr(parts[0])
                right_var = self._eval_expr(parts[1])
                v = self.fresh("t")
                self.emit(f"stream {v} = {left_var} {op} {right_var}")
                return v

        # Fallback: try as variable
        try:
            return self.get_var(expr)
        except KeyError:
            raise ValueError(f"Cannot parse Pablo expression: {expr!r}")

    def _eval_comparison(self, lhs_name: str, op: str, rhs_name: str) -> str:
        """Evaluate a byte-level comparison (packed mode)."""
        if self.input_mode != "packed":
            raise ValueError(
                f"Byte comparison in non-packed mode: {lhs_name} {op} {rhs_name}"
            )
        byte_val = self.repeat_vals[rhs_name]
        if op == "==" or op == "==":
            return self.expand_byte_eq(byte_val)
        elif op == "<":
            return self.expand_byte_lt(byte_val)
        else:
            raise ValueError(f"Unsupported comparison op: {op}")

    def _handle_pack(self, lhs: str, rhs: str):
        """Handle PackL/PackH operations to resolve basis bit indices."""
        m = re.match(r"Pack([LH])\((\d+),\s*(\w+(?:\[\d+\])?)\)", rhs)
        if not m:
            raise ValueError(f"Cannot parse Pack operation: {rhs}")

        kind = m.group(1)  # 'L' or 'H'
        width = int(m.group(2))  # 8, 4, or 2
        src = m.group(3)

        if width == 8:
            # PackL(8, UTF8_basis[0]) = [b0,b1,b2,b3]
            # PackH(8, UTF8_basis[0]) = [b4,b5,b6,b7]
            if kind == "L":
                self.pack_bits[lhs] = [0, 1, 2, 3]
            else:
                self.pack_bits[lhs] = [4, 5, 6, 7]
        elif width == 4:
            # PackL(4, [a,b,c,d]) = [a,b]
            # PackH(4, [a,b,c,d]) = [c,d]
            src_bits = self.resolve_pack(src)
            if kind == "L":
                self.pack_bits[lhs] = src_bits[:2]
            else:
                self.pack_bits[lhs] = src_bits[2:]
        elif width == 2:
            # PackL(2, [a,b]) = a (scalar)
            # PackH(2, [a,b]) = b (scalar)
            src_bits = self.resolve_pack(src)
            if kind == "L":
                bit_idx = src_bits[0]
            else:
                bit_idx = src_bits[1]
            self.pack_bits[lhs] = bit_idx
            # Map the variable to the basis bit
            self.set_var(lhs, f"b[{bit_idx}]")
        else:
            raise ValueError(f"Unexpected Pack width: {width}")

    # -- Compound operation emission --

    def _emit_matchstar(self, m_var: str, c_var: str) -> str:
        """Emit MatchStar(m, c) = ((m & c) + c) ^ c | m  (4 ops)."""
        mc = self.fresh("ms")
        self.emit(f"stream {mc} = {m_var} & {c_var}")
        add = self.fresh("ms")
        self.emit(f"stream {add} = {mc} + {c_var}")
        xor = self.fresh("ms")
        self.emit(f"stream {xor} = {add} ^ {c_var}")
        result = self.fresh("ms")
        self.emit(f"stream {result} = {xor} | {m_var}")
        return result

    def _emit_sel(self, c_var: str, t_var: str, f_var: str) -> str:
        """Emit Sel(c, t, f) = (t & c) | (f & ~c)  (4 ops)."""
        tc = self.fresh("sel")
        self.emit(f"stream {tc} = {t_var} & {c_var}")
        nc = self.fresh("sel")
        self.emit(f"stream {nc} = ~{c_var}")
        fnc = self.fresh("sel")
        self.emit(f"stream {fnc} = {f_var} & {nc}")
        result = self.fresh("sel")
        self.emit(f"stream {result} = {tc} | {fnc}")
        return result

    def _emit_scanthru(self, f_var: str, t_var: str) -> str:
        """Emit ScanThru(f, t) = (f + t) & ~t  (3 ops)."""
        add = self.fresh("st")
        self.emit(f"stream {add} = {f_var} + {t_var}")
        nt = self.fresh("st")
        self.emit(f"stream {nt} = ~{t_var}")
        result = self.fresh("st")
        self.emit(f"stream {result} = {add} & {nt}")
        return result

    def _emit_indexed_advance(self, marker: str, idx: str, n: int) -> str:
        """Emit IndexedAdvance(marker, idx, n): advance by n indexed positions.

        Each step: shift by 1, ScanThru past non-indexed positions, AND with idx.
        Caches ~idx across iterations (1 NOT + n * 4 ops).
        """
        cur = marker
        not_idx = self.fresh("nidx")
        self.emit(f"stream {not_idx} = ~{idx}")
        for _ in range(n):
            adv = self.fresh("iadv")
            self.emit(f"stream {adv} = {cur} << 1")
            thru = self._emit_scanthru(adv, not_idx)
            result = self.fresh("iadv")
            self.emit(f"stream {result} = {thru} & {idx}")
            cur = result
        return cur


# ---------------------------------------------------------------------------
# Multi-pattern compilation
# ---------------------------------------------------------------------------


def run_icgrep(pattern: str, flags: str = "") -> str:
    """Run icgrep --ShowPablo on a pattern and return the output.

    Pre-processes the pattern to expand shorthand classes and
    case-insensitive flag.
    """
    processed = preprocess_pattern(pattern, flags)

    cmd = [ICGREP_BIN, "--ShowPablo"]

    # Don't pass -i to icgrep: our pre-processor already expands
    # case-insensitive letters to [Aa] form, avoiding Unicode properties.

    cmd.append(processed)
    cmd.append("/dev/null")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # icgrep may return non-zero even on success (no matches in /dev/null)
        output = result.stdout + result.stderr
        return output
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"icgrep timed out on pattern: {pattern}")
    except FileNotFoundError:
        raise RuntimeError(f"icgrep not found at: {ICGREP_BIN}")


def _output_is_trivial_ones(bs_lines: list[str], output_var: str) -> bool:
    """Return True if the output_var resolves to a constant ONES.

    icgrep occasionally emits a trivially-true Pablo kernel for patterns it
    cannot model (e.g., RTF-control sequences with `{n}` lookarounds).  Such
    kernels would match every byte position when run, which is wrong.
    Detect by checking the single defining assignment of ``output_var``.
    """
    if output_var is None:
        return False
    pattern = re.compile(
        rf"^\s*stream\s+{re.escape(output_var)}\s*=\s*ONES\s*$"
    )
    return any(pattern.match(line) for line in bs_lines)


def compile_pattern_via_icgrep(
    name: str, pattern: str, flags: str = ""
) -> tuple[list[str], str | None]:
    """Compile a regex pattern to .bs via icgrep.

    Returns (list of .bs lines, output variable name or None on failure).
    """
    try:
        output = run_icgrep(pattern, flags)
    except RuntimeError as e:
        print(f"  icgrep failed for /{pattern}/{flags}: {e}", file=sys.stderr)
        return [], None

    kernel_text = PabloToBs.extract_regex_kernel(output)
    if kernel_text is None:
        print(
            f"  No regex kernel found for /{pattern}/{flags}",
            file=sys.stderr,
        )
        return [], None

    converter = PabloToBs(prefix=name)
    try:
        bs_lines = converter.convert_kernel(kernel_text)
    except Exception as e:
        print(
            f"  Conversion failed for /{pattern}/{flags}: {e}",
            file=sys.stderr,
        )
        return [], None

    output_var = converter.var_map.get("__output__")

    # icgrep sometimes emits a trivially-true (ONES) kernel for patterns it
    # cannot represent faithfully (e.g. some RTF control sequences).  Such a
    # kernel would mark every byte position as a match and would dominate
    # ``any_match``.  Substitute a ZERO stub so the pattern contributes
    # nothing — same convention used by the always-ZERO failure path.
    if _output_is_trivial_ones(bs_lines, output_var):
        print(
            f"  trivial-ONES kernel for /{pattern}/{flags}: substituting ZERO",
            file=sys.stderr,
        )
        stub_z1 = f"_{name}z1"
        stub_z2 = f"_{name}z2"
        bs_lines = [
            "// icgrep emitted a trivial-ONES kernel; substituted ZERO.",
            f"stream {stub_z1} = ZERO",
            f"stream {stub_z2} = ZERO",
        ]
        output_var = stub_z2

    return bs_lines, output_var


def compile_patterns_via_icgrep(
    patterns: list[tuple[str, str, str]],
) -> str:
    """Compile multiple patterns to a single .bs program via icgrep.

    Args:
        patterns: list of (name, regex_pattern, flags) tuples

    Returns:
        .bs source code as a string
    """
    header = []
    header.append("// Multi-pattern regex matcher (Snort IDS rules)")
    header.append(
        "// Compiled via icgrep Pablo IR (Parabix, Cameron et al., 2014)"
    )
    header.append(
        "// Source: AutomataZoo snort.regex (Wadden et al., 2018)"
    )
    header.append("//")
    header.append(
        "// Input: 8 basis bit-planes b[0..7] (byte decomposition)"
    )
    header.append(f"// Patterns: {len(patterns)}")
    header.append("")
    header.append("input stream b[8]")
    header.append("")

    # Declare per-pattern streams as internals.  The committed .bs files use
    # a single `output stream any_match` (OR of all pattern streams); the
    # per-pattern streams are computed but not exported.  Format matches
    # commit 674ce74 ("Regex: single any_match output").
    for name, _, _ in patterns:
        header.append(f"stream {name}")
    header.append("")

    body_lines = []
    compiled_count = 0
    failed = []

    for name, pattern, flags in patterns:
        body_lines.append(f"// -- Pattern: /{pattern}/{flags} --")

        bs_lines, output_var = compile_pattern_via_icgrep(name, pattern, flags)

        if bs_lines and output_var is not None:
            body_lines.extend(bs_lines)
            body_lines.append(f"{name} = {output_var}")
            body_lines.append("")
            compiled_count += 1
        else:
            # Fallback: emit ZERO output
            body_lines.append(f"// FAILED: could not compile via icgrep")
            body_lines.append(f"stream _{name}_fail = ZERO")
            body_lines.append(f"{name} = _{name}_fail")
            body_lines.append("")
            failed.append(name)

    # Aggregate per-pattern streams into a single any_match output.
    tail_lines: list[str] = []
    tail_lines.append("")
    tail_lines.append(
        f"// Final OR of all {len(patterns)} pattern match streams"
    )
    tail_lines.append("output stream any_match")
    if patterns:
        names = [n for n, _, _ in patterns]
        tail_lines.append(f"any_match = {names[0]}")
        for n in names[1:]:
            tail_lines.append(f"any_match = any_match | {n}")
    else:
        tail_lines.append("any_match = ZERO")

    if failed:
        print(
            f"WARNING: {len(failed)}/{len(patterns)} patterns failed: "
            + ", ".join(failed),
            file=sys.stderr,
        )

    print(
        f"Compiled {compiled_count}/{len(patterns)} patterns via icgrep"
    )

    all_lines = header + body_lines + tail_lines
    return "\n".join(all_lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI: compile a regex to .bs via icgrep and print to stdout."""
    if len(sys.argv) < 2:
        print("Usage: pablo_to_bs.py <regex> [flags] [name]")
        print("  Compiles a regex to .bs via icgrep --ShowPablo")
        print(f"  icgrep binary: {ICGREP_BIN}")
        sys.exit(1)

    pattern = sys.argv[1]
    flags = sys.argv[2] if len(sys.argv) > 2 else ""
    name = sys.argv[3] if len(sys.argv) > 3 else "match"

    source = compile_patterns_via_icgrep([(name, pattern, flags)])
    print(source)


if __name__ == "__main__":
    main()
