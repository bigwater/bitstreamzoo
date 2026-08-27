#!/usr/bin/env python3
"""Generate tiered real-life .npz datasets for bnn.

Input format: binary weight matrices (w1..w4), integer thresholds (t1..t3),
and binarized test images + labels.

The "data size" for BNN tiers scales by the number of test images.
Each MNIST image is 784 binary features stored as one uint8 per
feature (values 0/1), not bit-packed. Arrays are written with
np.savez_compressed, so on-disk size is well below the raw
784 bytes/image.

Tiers (approximate compressed .npz size, params + images):
  small:  MNIST params + 10K test images   (~0.8 MB)
  medium: MNIST params + 100K images (tiled 10x from 10K set, ~4 MB)
  large:  MNIST params + 1M images (tiled 100x from 10K set, ~37 MB)

Source: Trained BNN parameters from raw/ (train_mnist.py).
        The existing mnist_params.npz already contains 10K test images
        from the MNIST dataset (binarized with threshold 0.5).

Usage:
    python make_data.py                   # Generate all tiers
    python make_data.py --tier small      # Generate one tier
"""

import argparse
import os
import sys

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

# Source: existing MNIST params file
MNIST_PARAMS = os.path.join(DIR, "small", "mnist_params.npz")

TIERS = {
    "small":  10_000,       # 10K images (original MNIST test set)
    "medium": 100_000,      # 100K images (tiled 10x)
    "large":  1_000_000,    # 1M images (tiled 100x)
}

OUTPUT_NAMES = {
    "small":  "mnist_10k.npz",
    "medium": "mnist_100k.npz",
    "large":  "mnist_1m.npz",
}


def generate_tier(tier: str):
    """Generate one tier of bnn data."""
    n_images = TIERS[tier]
    out_dir = os.path.join(DIR, tier)
    out_path = os.path.join(out_dir, OUTPUT_NAMES[tier])

    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"  [{tier}] Already exists: {out_path} ({size:,} bytes) -- skipping")
        return

    if not os.path.isfile(MNIST_PARAMS):
        print(f"  [{tier}] SKIP: source params not found: {MNIST_PARAMS}")
        print(f"         Run train_mnist.py first to generate mnist_params.npz")
        return

    print(f"  [{tier}] Loading base MNIST params from {os.path.basename(MNIST_PARAMS)}...")
    data = np.load(MNIST_PARAMS)

    # Extract components
    w1 = data["w1"]            # (K1, N1) weights
    w2 = data["w2"]            # (K2, K1) weights
    w3 = data["w3"]            # (K3, K2) weights
    w4 = data["w4"]            # (10, K3) weights
    t1 = data["t1"]            # (K1,) thresholds
    t2 = data["t2"]            # (K2,) thresholds
    t3 = data["t3"]            # (K3,) thresholds
    test_images = data["test_images"]   # (10000, 784) uint8
    test_labels = data["test_labels"]   # (10000,) int64
    accuracy = data["accuracy"]         # scalar float64

    base_n = test_images.shape[0]  # 10000
    print(f"  [{tier}] Base: {base_n:,} images, target: {n_images:,} images")

    if n_images <= base_n:
        # Just use the first n_images
        tiled_images = test_images[:n_images]
        tiled_labels = test_labels[:n_images]
    else:
        # Tile the dataset
        tiles = (n_images + base_n - 1) // base_n
        print(f"  [{tier}] Tiling {tiles}x to reach {n_images:,} images")
        tiled_images = np.tile(test_images, (tiles, 1))[:n_images]
        tiled_labels = np.tile(test_labels, tiles)[:n_images]

    # Save with same format as original
    arrays = {
        "w1": w1, "w2": w2, "w3": w3, "w4": w4,
        "t1": t1, "t2": t2, "t3": t3,
        "test_images": tiled_images,
        "test_labels": tiled_labels,
        "accuracy": accuracy,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    size = os.path.getsize(out_path)
    img_bytes = n_images * 784
    print(f"  [{tier}] Wrote {out_path} ({size:,} bytes, compressed .npz)")
    print(f"  [{tier}] Image data: {n_images:,} images x 784 uint8 features = {img_bytes:,} bytes uncompressed")


def main():
    parser = argparse.ArgumentParser(description="Generate bnn tiered datasets")
    parser.add_argument("--tier", choices=["small", "medium", "large"],
                        help="Generate only this tier (default: all)")
    args = parser.parse_args()

    print("bnn: generating tiered datasets from trained MNIST BNN")
    print(f"  Source: {MNIST_PARAMS}")
    print()

    tiers = [args.tier] if args.tier else ["small", "medium", "large"]
    for tier in tiers:
        generate_tier(tier)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
