#!/usr/bin/env python3
"""Wrapper for BNN training and parameter export.

The training script (train_mnist.py) already resides in this directory.
This convert.py provides a unified interface to run the training pipeline
and verify the exported parameter files.

Requirements: pip install torch torchvision numpy

Usage:
    python3 convert.py --mnist       # Train MNIST BNN, export mnist_params.npz
    python3 convert.py --verify      # Verify existing .npz files
"""

import os
import sys
import subprocess

DIR = os.path.dirname(os.path.abspath(__file__))


def train_mnist() -> None:
    """Run the MNIST training script."""
    script = os.path.join(DIR, "train_mnist.py")
    if not os.path.isfile(script):
        print(f"Error: {script} not found")
        sys.exit(1)
    print("Training MNIST BNN (784 -> 1024 -> 1024 -> 1024 -> 10)...")
    subprocess.run([sys.executable, script], check=True, cwd=DIR)


def verify_params() -> None:
    """Verify that exported parameter files exist and have correct shapes."""
    try:
        import numpy as np
    except ImportError:
        print("Error: numpy is required. Install with: pip install numpy")
        sys.exit(1)

    for name, rel_path, shapes in [
        ("mnist_params.npz", os.path.join(DIR, "..", "small", "mnist_params.npz"),
         {"w1": (1024, 784), "w4": (10, 1024)}),
    ]:
        path = rel_path
        if not os.path.isfile(path):
            print(f"  {name}: NOT FOUND")
            continue
        data = np.load(path)
        print(f"  {name}: {list(data.keys())}")
        for key, expected in shapes.items():
            if key in data:
                actual = data[key].shape
                ok = "OK" if actual == expected else f"MISMATCH (got {actual})"
                print(f"    {key}: {actual} {ok}")


if __name__ == "__main__":
    if "--mnist" in sys.argv:
        train_mnist()
    elif "--verify" in sys.argv:
        verify_params()
    else:
        print("Usage: python3 convert.py [--mnist | --verify]")
        sys.exit(1)
