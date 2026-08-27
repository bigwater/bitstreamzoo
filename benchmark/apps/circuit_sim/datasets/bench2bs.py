#!/usr/bin/env python3
"""Convert ISCAS .bench format netlists to .bs (bitstream) format.

Usage:
    python3 benchmark/apps/circuit_sim/datasets/bench2bs.py input.bench [output.bs]

Also usable as a library:
    from benchmark.apps.circuit_sim.datasets.bench2bs import convert
    bs_text = convert(bench_text)
"""

from __future__ import annotations

import sys
import re


# .bs reserved words — identifiers must not collide with these
RESERVED = frozenset({
    "input", "output", "param", "stream", "int",
    "if", "while", "for", "in", "ZERO", "ONES",
})


def sanitize_name(name: str) -> str:
    """Ensure a wire name is a valid .bs identifier.

    Rules:
      - Must match [a-zA-Z_][a-zA-Z0-9_]*
      - Numeric-leading names get 'n' prefix  (1 -> n1)
      - Reserved words get 'w_' prefix        (input -> w_input)
    """
    name = name.strip()
    if name and name[0].isdigit():
        name = "n" + name
    if name in RESERVED:
        name = "w_" + name
    return name


# ── Parsing ────────────────────────────────────────────────────

def parse_bench(text: str):
    """Parse a .bench file.

    Returns (inputs, outputs, gates) where:
      inputs:  list of wire names
      outputs: list of wire names
      gates:   list of (out_wire, gate_type, [in_wires])
    """
    inputs: list[str] = []
    outputs: list[str] = []
    gates: list[tuple[str, str, list[str]]] = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # INPUT(wire)
        m = re.match(r"INPUT\s*\(\s*(\w+)\s*\)", line)
        if m:
            inputs.append(m.group(1))
            continue

        # OUTPUT(wire)
        m = re.match(r"OUTPUT\s*\(\s*(\w+)\s*\)", line)
        if m:
            outputs.append(m.group(1))
            continue

        # gate:  out = TYPE(in1, in2, ...)
        m = re.match(r"(\w+)\s*=\s*(\w+)\s*\(([^)]*)\)", line)
        if m:
            out_wire = m.group(1)
            gate_type = m.group(2).upper()
            in_wires = [w.strip() for w in m.group(3).split(",") if w.strip()]
            gates.append((out_wire, gate_type, in_wires))
            continue

    return inputs, outputs, gates


# ── Topological sort ───────────────────────────────────────────

def topo_sort(inputs: list[str], gates: list) -> list:
    """Topologically sort gates so each gate's inputs are defined first."""
    defined = set(inputs)
    sorted_gates: list = []
    remaining = list(gates)

    while remaining:
        next_remaining = []
        progress = False
        for gate in remaining:
            out_wire, _gate_type, in_wires = gate
            if all(w in defined for w in in_wires):
                sorted_gates.append(gate)
                defined.add(out_wire)
                progress = True
            else:
                next_remaining.append(gate)
        remaining = next_remaining
        if not progress:
            undefined = set()
            for _out, _gt, ins in remaining:
                undefined.update(w for w in ins if w not in defined)
            raise ValueError(
                f"Cyclic dependency or undefined wires: {undefined}"
            )

    return sorted_gates


# ── Code generation ────────────────────────────────────────────

def _chain_binary(name: str, op: str, ins: list[str],
                  lines: list[str], prefix: str) -> None:
    """Emit chained binary ops:  name = ins[0] op ins[1] op ... """
    if len(ins) == 1:
        lines.append(f"{prefix}{name} = {ins[0]}")
    elif len(ins) == 2:
        lines.append(f"{prefix}{name} = {ins[0]} {op} {ins[1]}")
    else:
        prev = ins[0]
        for i in range(1, len(ins)):
            if i == len(ins) - 1:
                lines.append(f"{prefix}{name} = {prev} {op} {ins[i]}")
            else:
                tmp = f"_c_{name}_{i}"
                lines.append(f"stream {tmp} = {prev} {op} {ins[i]}")
                prev = tmp


def _emit_gate(out: str, gate_type: str, ins: list[str],
               is_output: bool) -> list[str]:
    """Emit .bs statements for one gate."""
    prefix = "" if is_output else "stream "
    lines: list[str] = []

    if gate_type == "BUF":
        lines.append(f"{prefix}{out} = {ins[0]}")
    elif gate_type == "NOT":
        lines.append(f"{prefix}{out} = ~{ins[0]}")
    elif gate_type in ("AND", "OR", "XOR"):
        op = {"AND": "&", "OR": "|", "XOR": "^"}[gate_type]
        _chain_binary(out, op, ins, lines, prefix)
    elif gate_type in ("NAND", "NOR", "XNOR"):
        base_op = {"NAND": "&", "NOR": "|", "XNOR": "^"}[gate_type]
        tmp = f"_t_{out}"
        _chain_binary(tmp, base_op, ins, lines, "stream ")
        lines.append(f"{prefix}{out} = ~{tmp}")
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")

    return lines


# ── Main conversion ───────────────────────────────────────────

def convert(bench_text: str) -> str:
    """Convert .bench text to .bs text."""
    inputs, outputs, gates = parse_bench(bench_text)
    gates = topo_sort(inputs, gates)
    output_set = set(outputs)

    lines: list[str] = []
    lines.append("// Auto-generated from .bench netlist")
    lines.append("")

    # Declare inputs
    for inp in inputs:
        lines.append(f"input stream {sanitize_name(inp)}")
    lines.append("")

    # Declare outputs
    for out in outputs:
        lines.append(f"output stream {sanitize_name(out)}")
    lines.append("")

    # Gate logic
    for out_wire, gate_type, in_wires in gates:
        is_output = out_wire in output_set
        out = sanitize_name(out_wire)
        ins = [sanitize_name(w) for w in in_wires]
        gate_lines = _emit_gate(out, gate_type, ins, is_output)
        lines.extend(gate_lines)

    return "\n".join(lines) + "\n"


# ── Gate-level reference simulator ─────────────────────────────

_GATE_EVAL = {
    "BUF":  lambda ins: ins[0],
    "NOT":  lambda ins: 1 - ins[0],
    "AND":  lambda ins: 1 if all(ins) else 0,
    "OR":   lambda ins: 1 if any(ins) else 0,
    "XOR":  lambda ins: sum(ins) % 2,
    "NAND": lambda ins: 0 if all(ins) else 1,
    "NOR":  lambda ins: 0 if any(ins) else 1,
    "XNOR": lambda ins: 1 - (sum(ins) % 2),
}


def simulate_bench(bench_text: str,
                   input_values: dict[str, int]) -> dict[str, int]:
    """Evaluate a .bench netlist on a single input vector.

    input_values: {wire_name: 0 or 1}
    Returns: {output_wire: 0 or 1}
    """
    inputs, outputs, gates = parse_bench(bench_text)
    gates = topo_sort(inputs, gates)

    wire = dict(input_values)
    for out_wire, gate_type, in_wires in gates:
        fn = _GATE_EVAL.get(gate_type)
        if fn is None:
            raise ValueError(f"Unknown gate type: {gate_type}")
        wire[out_wire] = fn([wire[w] for w in in_wires])

    return {o: wire[o] for o in outputs}


# ── CLI ────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bench2bs.py <input.bench> [output.bs]",
              file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        bench_text = f.read()

    bs_text = convert(bench_text)

    if len(sys.argv) >= 3:
        with open(sys.argv[2], "w") as f:
            f.write(bs_text)
    else:
        print(bs_text, end="")


if __name__ == "__main__":
    main()
