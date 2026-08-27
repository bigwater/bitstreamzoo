#!/usr/bin/env python3
"""Standalone test harness for Binary Neural Network (BNN) inference.

Verifies the bitstream BNN layer against a Python scalar reference.
Each bit position is one sample in the inference batch.

BNN fully-connected layer: XNOR matching + ripple-carry popcount + threshold.

Sources:
  Hubara et al., "Binarized Neural Networks", NeurIPS 2016
  Rastegari et al., "XNOR-Net", ECCV 2016
  Umuroglu et al., "FINN: Fast, Scalable BNN Inference", FPGA 2017
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))


# ── Reference implementation ──────────────────────────────────────

def bnn_reference(x_features: list[list[int]], weights: list[list[int]],
                  thresholds: list[int]) -> list[list[int]]:
    """Scalar BNN reference: per-sample, per-neuron XNOR-popcount + threshold.

    Args:
        x_features: M x N matrix (list of M samples, each N binary features)
        weights: K x N matrix (list of K neurons, each N binary weights)
        thresholds: K thresholds (list of ints)

    Returns:
        M x K activations (list of M samples, each K binary outputs)
    """
    M = len(x_features)
    K = len(weights)
    N = len(weights[0])
    activations = []
    for m in range(M):
        acts = []
        for k in range(K):
            count = sum(1 for i in range(N) if x_features[m][i] == weights[k][i])
            acts.append(1 if count >= thresholds[k] else 0)
        activations.append(acts)
    return activations


# ── Encoding helpers ──────────────────────────────────────────────

def encode_features(x_features: list[list[int]], N: int) -> dict[int, int]:
    """Pack M samples' N features into N streams.

    Returns dict {i: stream} where bit m of stream i = x_features[m][i].
    """
    M = len(x_features)
    x = {}
    for i in range(N):
        bits = 0
        for m in range(M):
            if x_features[m][i]:
                bits |= 1 << m
        x[i] = bits
    return x


def encode_weights(weights: list[list[int]], N: int, K: int) -> dict[int, int]:
    """Broadcast K*N weights to streams.

    Returns dict {k*N+i: stream} where each entry is all-1s (-1) or 0.
    """
    w = {}
    for k in range(K):
        for i in range(N):
            w[k * N + i] = -1 if weights[k][i] else 0
    return w


def encode_thresholds(thresholds: list[int], K: int, B: int) -> dict[int, int]:
    """Broadcast threshold bits.

    Returns dict {k*B+b: stream} where each entry is all-1s (-1) or 0.
    """
    t = {}
    for k in range(K):
        for b in range(B):
            t[k * B + b] = -1 if ((thresholds[k] >> b) & 1) else 0
    return t


def decode_activations(result: dict, K: int, M: int) -> list[list[int]]:
    """Extract M x K activation matrix from output array.

    result["act"] is a dict {k: stream}.
    """
    act = result["act"]
    activations = []
    for m in range(M):
        acts = []
        for k in range(K):
            acts.append((act.get(k, 0) >> m) & 1)
        activations.append(acts)
    return activations


# ── Main ──────────────────────────────────────────────────────────

def main():
    from benchmark.base import GenericDomain

    _name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    domain = GenericDomain(_name)
    results = domain.run_all()

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print("Binary Neural Network (BNN) Inference Benchmark")
    print()
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.test_name}  ({r.bitlength} samples, {r.op_count} ops)")

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
