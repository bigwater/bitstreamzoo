#!/usr/bin/env python3
"""Generate polar code encoding .bs programs for any power-of-2 N.

Polar code encoding computes x = u * F^{otimes n} where F = [[1,0],[1,1]]
and n = log2(N). This is an in-place butterfly XOR network with n stages.

Usage:
    python polar_gen.py              # generate all 3 sizes
    python polar_gen.py --sizes 64   # generate just N=64
"""

import math
import os
import sys


def generate_polar_bs(N: int) -> str:
    """Generate a .bs program for polar code encoding of length N.

    The butterfly XOR network has n = log2(N) stages.
    Stage s has stride = 2^s, with N/(2*stride) groups of 2*stride elements.
    Within each group: x[base+i] ^= x[base+i+stride] for i in 0..stride-1.
    """
    assert N > 0 and (N & (N - 1)) == 0, f"N must be a power of 2, got {N}"
    n = int(math.log2(N))

    lines = []
    lines.append(f"// Polar code encoding (Arikan 2009), N={N}")
    lines.append(f"// Butterfly XOR network: F^{{otimes {n}}}, "
                 f"{n} stages, {N * n // 2} XOR ops")
    lines.append(f"")
    lines.append(f"input stream u[{N}]")
    lines.append(f"output stream x[{N}]")
    lines.append(f"")
    lines.append(f"// Copy input to working array")
    lines.append(f"for i in 0..{N} {{ x[i] = u[i] }}")

    for s in range(n):
        stride = 1 << s
        group_size = stride * 2
        n_groups = N // group_size

        lines.append(f"")
        lines.append(f"// Stage {s}: stride={stride}, "
                     f"{n_groups} group{'s' if n_groups > 1 else ''} "
                     f"of {group_size}")

        if n_groups == 1:
            # Single group, no outer loop needed
            for i in range(stride):
                b = i + stride
                lines.append(f"x[{i}] = x[{i}] ^ x[{b}]")
        elif stride == 1:
            # stride=1: each group is just one XOR, no inner loop
            lines.append(f"for g in 0..{n_groups} {{")
            lines.append(f"    int a = g * 2")
            lines.append(f"    int b = a + 1")
            lines.append(f"    x[a] = x[a] ^ x[b]")
            lines.append(f"}}")
        else:
            # General case: nested loops
            lines.append(f"for g in 0..{n_groups} {{")
            lines.append(f"    int base = g * {group_size}")
            lines.append(f"    for i in 0..{stride} {{")
            lines.append(f"        int a = base + i")
            lines.append(f"        int b = a + {stride}")
            lines.append(f"        x[a] = x[a] ^ x[b]")
            lines.append(f"    }}")
            lines.append(f"}}")

    lines.append(f"")
    return "\n".join(lines)


SIZES = {
    "polar_small": 64,
    "polar_medium": 256,
    "polar_large": 1024,
}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate polar code .bs programs")
    parser.add_argument("--sizes", nargs="+", type=int,
                        help="Code lengths to generate (default: 64 256 1024)")
    args = parser.parse_args()

    src_dir = os.path.dirname(os.path.abspath(__file__))

    if args.sizes:
        targets = {f"polar_N{N}": N for N in args.sizes}
    else:
        targets = SIZES

    for name, N in targets.items():
        source = generate_polar_bs(N)
        path = os.path.join(src_dir, f"{name}.bs")
        with open(path, "w") as f:
            f.write(source)
        n = int(math.log2(N))
        print(f"  {name}.bs: N={N}, {n} stages, "
              f"{N * n // 2} XOR ops, {2 * N} streams")


if __name__ == "__main__":
    main()
