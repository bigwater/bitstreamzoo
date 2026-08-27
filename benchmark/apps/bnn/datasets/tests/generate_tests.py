#!/usr/bin/env python3
"""Generate precomputed bitstream-level test data for 3-layer BNN.

Usage:
    python generate_tests.py                # unit tests + MNIST accuracy tests
    python generate_tests.py --tier small   # + small tier data
    python generate_tests.py --tier medium  # + medium tier data
    python generate_tests.py --tier large   # skipped for BNN (no large .bsdata tier)
    python generate_tests.py --describe     # print provenance info
    python generate_tests.py --verify       # verify SHA-256 of existing files
"""

import json
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from simulator.pythonsim import parse
from simulator.pythonsim.interpreter import Interpreter
from benchmark.apps.bnn.src.run import (
    bnn_reference, encode_features, encode_weights, encode_thresholds,
    decode_activations,
)
from benchmark.bsdata import write_bsdata
from benchmark.base import CppBackend, ProgramInfo
from benchmark.tier_generate import (
    parse_generate_args, file_sha256, make_provenance,
    tier_test_entry, print_describe, verify_files,
    getrandbits_large,
)
from benchmark.tier_config import get_tier_vectors

TESTS_DIR = os.path.dirname(__file__)
BS_MNIST_PATH = os.path.join(TESTS_DIR, "../../src/bnn_mnist.bs")

DOMAIN = "bnn"

# Fixed architecture for tier tests (3-layer)
# Narrow I/O, wide hidden — maximizes ops and GPU utilization within tier budget
TIER_N1 = 128
TIER_K1 = 256
TIER_N2 = 256
TIER_K2 = 256
TIER_N3 = 256
TIER_K3 = 4

TIER_SEEDS = {"small": 1000, "medium": 1001, "large": 1002}
TIER_NAMES = {
    "small":  "Tier small",
    "medium": "Tier medium",
    "large":  "Tier large",
}


def mask_stream(v, bitlength):
    """Mask a stream value to bitlength bits (handles NOT -> negative ints)."""
    if bitlength > 0:
        return v & ((1 << bitlength) - 1)
    return v


def make_3layer_params(N1, K1, N2, K2, N3, K3):
    """Compute B values and return full params dict for the 3-layer program.

    Adds N{1,2,3}_CHUNKS = N{1,2,3} // 16 for the FINN-canonical chunked
    tree-reduction popcount in bnn_mnist.bs. The .bs DSL int_expr grammar
    does not support integer division, so the chunk counts are computed
    here at test-generation time and passed as runtime params.

    The .bs program handles the (N - 16*N_CHUNKS) remainder via a
    per-input ripple-carry fallback loop, so any N >= 0 is supported
    (including N < 16, used in the unit tests).
    """
    B1 = N1.bit_length()
    B2 = N2.bit_length()
    B3 = N3.bit_length()
    CHUNK = 16
    return {
        "N1": N1, "K1": K1, "B1": B1, "N1_CHUNKS": N1 // CHUNK,
        "N2": N2, "K2": K2, "B2": B2, "N2_CHUNKS": N2 // CHUNK,
        "N3": N3, "K3": K3, "B3": B3, "N3_CHUNKS": N3 // CHUNK,
    }


def encode_3layer_inputs(x_arrays, weights_list, thresholds_list, params):
    """Encode 3-layer BNN inputs to input_arrays format for .bsdata.

    Args:
        x_arrays: dict {i: stream_int} for input features
        weights_list: [w1, w2, w3] — each K x N list-of-lists of {0,1}
        thresholds_list: [t1, t2, t3] — each K-element list of ints
        params: dict with N1,K1,B1,N2,K2,B2,N3,K3,B3

    Returns: input_arrays dict for write_bsdata
    """
    N1, K1, B1 = params["N1"], params["K1"], params["B1"]
    N2, K2, B2 = params["N2"], params["K2"], params["B2"]
    N3, K3, B3 = params["N3"], params["K3"], params["B3"]

    w1_arrays = encode_weights(weights_list[0], N1, K1)
    t1_arrays = encode_thresholds(thresholds_list[0], K1, B1)
    w2_arrays = encode_weights(weights_list[1], N2, K2)
    t2_arrays = encode_thresholds(thresholds_list[1], K2, B2)
    w3_arrays = encode_weights(weights_list[2], N3, K3)
    t3_arrays = encode_thresholds(thresholds_list[2], K3, B3)

    return {
        "x": x_arrays,
        "w1": w1_arrays, "t1": t1_arrays,
        "w2": w2_arrays, "t2": t2_arrays,
        "w3": w3_arrays, "t3": t3_arrays,
    }


def bnn_3layer_reference(x_features, weights_list, thresholds_list):
    """Scalar reference for 3-layer BNN: chain bnn_reference 3 times.

    Args:
        x_features: M x N1 list of samples
        weights_list: [w1, w2, w3]
        thresholds_list: [t1, t2, t3]

    Returns: M x K3 activation matrix
    """
    acts = x_features
    for w, t in zip(weights_list, thresholds_list):
        acts = bnn_reference(acts, w, t)
    return acts


def bnn_3layer_bitstream_reference(x_arrays, input_arrays, params, bitlength):
    """Compute 3-layer BNN expected output directly on bitstreams.

    Uses pure Python bitwise ops on large integers (efficient for large bitlength).

    Args:
        x_arrays: dict {i: int} for input features
        input_arrays: full input_arrays dict with w1,t1,w2,t2,w3,t3
        params: dict with N1,K1,B1,...
        bitlength: number of bit positions

    Returns: dict {k: int} activation streams for output layer
    """
    mask = (1 << bitlength) - 1

    def bnn_layer_bitstream(inp, w_arrays, t_arrays, N, K, B):
        """One BNN layer on bitstreams. Returns dict {k: stream}."""
        act = {}
        for k in range(K):
            count = [0] * B
            for i in range(N):
                xor_xw = inp[i] ^ w_arrays[k * N + i]
                match = (~xor_xw) & mask
                carry = match
                for b in range(B):
                    s = count[b] ^ carry
                    carry = count[b] & carry
                    count[b] = s & mask
            gt = 0
            eq = mask
            for j in range(B):
                a = count[B - 1 - j]
                tb = t_arrays[k * B + B - 1 - j] & mask
                not_tb = (~tb) & mask
                a_gt = a & not_tb
                xor_ab = a ^ tb
                a_eq = (~xor_ab) & mask
                eq_and_gt = eq & a_gt
                gt = (gt | eq_and_gt) & mask
                eq = (eq & a_eq) & mask
            act[k] = (gt | eq) & mask
        return act

    N1, K1, B1 = params["N1"], params["K1"], params["B1"]
    N2, K2, B2 = params["N2"], params["K2"], params["B2"]
    N3, K3, B3 = params["N3"], params["K3"], params["B3"]

    h1 = bnn_layer_bitstream(x_arrays, input_arrays["w1"], input_arrays["t1"],
                             N1, K1, B1)
    h2 = bnn_layer_bitstream(h1, input_arrays["w2"], input_arrays["t2"],
                             N2, K2, B2)
    act = bnn_layer_bitstream(h2, input_arrays["w3"], input_arrays["t3"],
                              N3, K3, B3)
    return act


def run_3layer_bs(program, x_features, weights_list, thresholds_list):
    """Run 3-layer BNN bitstream program via interpreter.

    Returns (act dict {k: stream_int}, op_count).
    """
    M = len(x_features)
    N1 = len(weights_list[0][0])
    K1 = len(weights_list[0])
    N2 = len(weights_list[1][0])
    K2 = len(weights_list[1])
    N3 = len(weights_list[2][0])
    K3 = len(weights_list[2])
    params = make_3layer_params(N1, K1, N2, K2, N3, K3)

    x_arrays = encode_features(x_features, N1)
    input_arrays = encode_3layer_inputs(x_arrays, weights_list,
                                        thresholds_list, params)

    interp = Interpreter()
    result = interp.run(
        program,
        inputs={},
        params=params,
        input_arrays=input_arrays,
    )
    return result["act"], interp.op_count


def generate_unit_tests(program):
    """Generate unit tests for the 3-layer program."""
    tests = []

    # Small dims for unit tests
    N1, K1 = 8, 4
    N2, K2 = 4, 4    # N2 = K1
    N3, K3 = 4, 2    # N3 = K2
    params = make_3layer_params(N1, K1, N2, K2, N3, K3)

    rng = random.Random(42)

    def gen_layer_params(rng, N, K):
        w = [[rng.randint(0, 1) for _ in range(N)] for _ in range(K)]
        t = [rng.randint(0, N) for _ in range(K)]
        return w, t

    # Test 1: All match — all weights=1, thresholds=0 across all layers
    w1 = [[1]*N1 for _ in range(K1)]
    t1 = [0]*K1
    w2 = [[1]*N2 for _ in range(K2)]
    t2 = [0]*K2
    w3 = [[1]*N3 for _ in range(K3)]
    t3 = [0]*K3
    x = [[1]*N1]
    weights_list = [w1, w2, w3]
    thresholds_list = [t1, t2, t3]

    ref = bnn_3layer_reference(x, weights_list, thresholds_list)
    act, _ = run_3layer_bs(program, x, weights_list, thresholds_list)
    bs_acts = decode_activations({"act": act}, K3, len(x))
    assert bs_acts == ref, f"All match: bs={bs_acts} ref={ref}"

    x_arrays = encode_features(x, N1)
    ia = encode_3layer_inputs(x_arrays, weights_list, thresholds_list, params)
    masked_act = {k: mask_stream(v, len(x)) for k, v in act.items()}
    write_bsdata(
        os.path.join(TESTS_DIR, "bnn_all_match.bsdata"),
        len(x), params=params, input_arrays=ia, expected={"act": masked_act},
    )
    tests.append({"name": "All match", "bitlength": len(x),
                  "data_file": "bnn_all_match.bsdata"})

    # Test 2: No match — all weights=0, high thresholds
    w1 = [[0]*N1 for _ in range(K1)]
    t1 = [N1]*K1
    w2 = [[0]*N2 for _ in range(K2)]
    t2 = [N2]*K2
    w3 = [[0]*N3 for _ in range(K3)]
    t3 = [N3]*K3
    x = [[1]*N1]
    weights_list = [w1, w2, w3]
    thresholds_list = [t1, t2, t3]

    ref = bnn_3layer_reference(x, weights_list, thresholds_list)
    act, _ = run_3layer_bs(program, x, weights_list, thresholds_list)
    bs_acts = decode_activations({"act": act}, K3, len(x))
    assert bs_acts == ref, f"No match: bs={bs_acts} ref={ref}"

    x_arrays = encode_features(x, N1)
    ia = encode_3layer_inputs(x_arrays, weights_list, thresholds_list, params)
    masked_act = {k: mask_stream(v, len(x)) for k, v in act.items()}
    write_bsdata(
        os.path.join(TESTS_DIR, "bnn_no_match.bsdata"),
        len(x), params=params, input_arrays=ia, expected={"act": masked_act},
    )
    tests.append({"name": "No match", "bitlength": len(x),
                  "data_file": "bnn_no_match.bsdata"})

    # Test 3: Multi-neuron chain with random weights
    rng2 = random.Random(100)
    w1, t1 = gen_layer_params(rng2, N1, K1)
    w2, t2 = gen_layer_params(rng2, N2, K2)
    w3, t3 = gen_layer_params(rng2, N3, K3)
    x = [[rng2.randint(0, 1) for _ in range(N1)] for _ in range(4)]
    weights_list = [w1, w2, w3]
    thresholds_list = [t1, t2, t3]

    ref = bnn_3layer_reference(x, weights_list, thresholds_list)
    act, _ = run_3layer_bs(program, x, weights_list, thresholds_list)
    bs_acts = decode_activations({"act": act}, K3, len(x))
    assert bs_acts == ref, f"Multi-neuron chain: bs={bs_acts} ref={ref}"

    x_arrays = encode_features(x, N1)
    ia = encode_3layer_inputs(x_arrays, weights_list, thresholds_list, params)
    masked_act = {k: mask_stream(v, len(x)) for k, v in act.items()}
    write_bsdata(
        os.path.join(TESTS_DIR, "bnn_multi_neuron.bsdata"),
        len(x), params=params, input_arrays=ia, expected={"act": masked_act},
    )
    tests.append({"name": "Multi-neuron chain", "bitlength": len(x),
                  "data_file": "bnn_multi_neuron.bsdata"})

    # Test 4: Exhaustive N1=8 (all 256 input patterns through 3 layers)
    rng3 = random.Random(42)
    w1, t1 = gen_layer_params(rng3, N1, K1)
    w2, t2 = gen_layer_params(rng3, N2, K2)
    w3, t3 = gen_layer_params(rng3, N3, K3)
    x = [[(m >> i) & 1 for i in range(N1)] for m in range(1 << N1)]
    M = len(x)
    weights_list = [w1, w2, w3]
    thresholds_list = [t1, t2, t3]

    ref = bnn_3layer_reference(x, weights_list, thresholds_list)
    act, _ = run_3layer_bs(program, x, weights_list, thresholds_list)
    bs_acts = decode_activations({"act": act}, K3, M)
    assert bs_acts == ref, "Exhaustive N1=8 mismatch"

    x_arrays = encode_features(x, N1)
    ia = encode_3layer_inputs(x_arrays, weights_list, thresholds_list, params)
    masked_act = {k: mask_stream(v, M) for k, v in act.items()}
    write_bsdata(
        os.path.join(TESTS_DIR, "bnn_exhaustive_n8.bsdata"),
        M, params=params, input_arrays=ia, expected={"act": masked_act},
    )
    tests.append({"name": "Exhaustive N1=8", "bitlength": M,
                  "data_file": "bnn_exhaustive_n8.bsdata"})

    return tests


def generate_random_batch_tests(program):
    """Generate random batch tests for the 3-layer program."""
    tests = []

    # Random 16-bit x100: small dims
    N1, K1 = 8, 4
    N2, K2 = 4, 4
    N3, K3 = 4, 2
    params = make_3layer_params(N1, K1, N2, K2, N3, K3)
    rng = random.Random(42)
    M_samples = 16
    count = 100

    sub_cases = []
    for _ in range(count):
        w1 = [[rng.randint(0, 1) for _ in range(N1)] for _ in range(K1)]
        t1 = [rng.randint(0, N1) for _ in range(K1)]
        w2 = [[rng.randint(0, 1) for _ in range(N2)] for _ in range(K2)]
        t2 = [rng.randint(0, N2) for _ in range(K2)]
        w3 = [[rng.randint(0, 1) for _ in range(N3)] for _ in range(K3)]
        t3 = [rng.randint(0, N3) for _ in range(K3)]
        x = [[rng.randint(0, 1) for _ in range(N1)] for _ in range(M_samples)]

        weights_list = [w1, w2, w3]
        thresholds_list = [t1, t2, t3]
        ref = bnn_3layer_reference(x, weights_list, thresholds_list)
        act, _ = run_3layer_bs(program, x, weights_list, thresholds_list)
        bs_acts = decode_activations({"act": act}, K3, M_samples)
        assert bs_acts == ref, "Random 16-bit batch mismatch"

        x_arrays = encode_features(x, N1)
        ia = encode_3layer_inputs(x_arrays, weights_list, thresholds_list, params)
        masked_act = {k: mask_stream(v, M_samples) for k, v in act.items()}
        sub_cases.append({
            "bitlength": M_samples,
            "params": params,
            "input_arrays": ia,
            "expected": {"act": masked_act},
        })

    write_bsdata(
        os.path.join(TESTS_DIR, "bnn_random_16bit_100.bsdata"),
        1600, cases=sub_cases,
    )
    tests.append({"name": "Random 16-bit x100", "bitlength": 1600,
                  "data_file": "bnn_random_16bit_100.bsdata"})

    # Random 64-bit x50: moderate dims
    N1, K1 = 16, 8
    N2, K2 = 8, 8
    N3, K3 = 8, 4
    params = make_3layer_params(N1, K1, N2, K2, N3, K3)
    rng = random.Random(43)
    M_samples = 64
    count = 50

    sub_cases = []
    for _ in range(count):
        w1 = [[rng.randint(0, 1) for _ in range(N1)] for _ in range(K1)]
        t1 = [rng.randint(0, N1) for _ in range(K1)]
        w2 = [[rng.randint(0, 1) for _ in range(N2)] for _ in range(K2)]
        t2 = [rng.randint(0, N2) for _ in range(K2)]
        w3 = [[rng.randint(0, 1) for _ in range(N3)] for _ in range(K3)]
        t3 = [rng.randint(0, N3) for _ in range(K3)]
        x = [[rng.randint(0, 1) for _ in range(N1)] for _ in range(M_samples)]

        weights_list = [w1, w2, w3]
        thresholds_list = [t1, t2, t3]
        ref = bnn_3layer_reference(x, weights_list, thresholds_list)
        act, _ = run_3layer_bs(program, x, weights_list, thresholds_list)
        bs_acts = decode_activations({"act": act}, K3, M_samples)
        assert bs_acts == ref, "Random 64-bit batch mismatch"

        x_arrays = encode_features(x, N1)
        ia = encode_3layer_inputs(x_arrays, weights_list, thresholds_list, params)
        masked_act = {k: mask_stream(v, M_samples) for k, v in act.items()}
        sub_cases.append({
            "bitlength": M_samples,
            "params": params,
            "input_arrays": ia,
            "expected": {"act": masked_act},
        })

    write_bsdata(
        os.path.join(TESTS_DIR, "bnn_random_64bit_50.bsdata"),
        3200, cases=sub_cases,
    )
    tests.append({"name": "Random 64-bit x50", "bitlength": 3200,
                  "data_file": "bnn_random_64bit_50.bsdata"})

    return tests


def generate_mnist_tests():
    """Generate MNIST accuracy tests using real trained weights.

    Produces .bsdata files with 3-layer BNN inputs and expected outputs.
    """
    import numpy as np
    tests = []

    datasets_dir = os.path.join(os.path.dirname(TESTS_DIR), "..")
    npz_path = os.path.join(datasets_dir, "datasets", "small", "mnist_params.npz")
    if not os.path.exists(npz_path):
        print(f"  [skip] MNIST params not found: {npz_path}")
        return tests

    data = np.load(npz_path)
    images = data["test_images"]       # (10000, 784) uint8
    labels = data["test_labels"]       # (10000,) int64
    w1_np = data["w1"]                 # (K1, 784) uint8
    w2_np = data["w2"]                 # (K2, K1) uint8
    w3_np = data["w3"]                 # (K3, K2) uint8
    t1_np = data["t1"]                 # (K1,) int64
    t2_np = data["t2"]                 # (K2,) int64
    t3_np = data["t3"]                 # (K3,) int64
    w4_np = data["w4"]                 # (10, K3) uint8
    model_acc = float(data["accuracy"])
    print(f"  MNIST model accuracy: {model_acc:.4f}")

    # Derive dimensions from weight shapes
    K1, N1 = w1_np.shape
    K2, N2 = w2_np.shape
    K3, N3 = w3_np.shape
    print(f"  Architecture: {N1} -> {K1} -> {K2} -> {K3}")

    # Convert numpy to Python lists
    w1 = [[int(w1_np[k, i]) for i in range(N1)] for k in range(K1)]
    w2 = [[int(w2_np[k, i]) for i in range(N2)] for k in range(K2)]
    w3 = [[int(w3_np[k, i]) for i in range(N3)] for k in range(K3)]
    t1 = [int(t1_np[k]) for k in range(K1)]
    t2 = [int(t2_np[k]) for k in range(K2)]
    t3 = [int(t3_np[k]) for k in range(K3)]
    weights_list = [w1, w2, w3]
    thresholds_list = [t1, t2, t3]

    params = make_3layer_params(N1, K1, N2, K2, N3, K3)

    for n_images, name_suffix, min_acc in [(100, "100", 0.95), (10000, "10k", 0.97)]:
        print(f"  Generating MNIST {name_suffix} ({n_images} images)...")
        imgs = images[:n_images]

        # Encode images as bitstreams
        x_features = [[int(imgs[m, i]) for i in range(N1)] for m in range(n_images)]
        x_arrays = encode_features(x_features, N1)

        # Compute expected 3-layer output via bitstream reference
        input_arrays = encode_3layer_inputs(x_arrays, weights_list,
                                            thresholds_list, params)
        act = bnn_3layer_bitstream_reference(x_arrays, input_arrays, params,
                                             n_images)

        # Verify accuracy: layer 4 (argmax over XNOR-popcount with w4)
        correct = 0
        for m in range(n_images):
            h3 = [(act.get(k, 0) >> m) & 1 for k in range(K3)]
            scores = []
            for c in range(10):
                count = sum(1 for i in range(K3) if h3[i] == int(w4_np[c, i]))
                scores.append(count)
            pred = int(np.argmax(scores))
            if pred == labels[m]:
                correct += 1
        acc = correct / n_images
        print(f"    Accuracy: {acc:.4f} ({correct}/{n_images})")
        assert acc >= min_acc, (
            f"MNIST {name_suffix} accuracy {acc:.4f} < {min_acc}")

        data_file = f"bnn_mnist_{name_suffix}.bsdata"
        write_bsdata(
            os.path.join(TESTS_DIR, data_file),
            n_images,
            params=params,
            input_arrays=input_arrays,
            expected={"act": {k: v for k, v in act.items()}},
        )
        size = os.path.getsize(os.path.join(TESTS_DIR, data_file))
        print(f"    Wrote {data_file} ({size:,} bytes)")
        tests.append({"name": f"MNIST {name_suffix}",
                      "bitlength": n_images,
                      "data_file": data_file})

    return tests


def generate_tier_data(tier, bitlength):
    """Generate tier .bsdata file for 3-layer BNN with random inputs.

    For small/medium tiers, uses pure Python bitstream reference.
    For large tier, uses C++ bsim backend (Python is infeasible ~14h).
    """
    N1, K1 = TIER_N1, TIER_K1
    N2, K2 = TIER_N2, TIER_K2
    N3, K3 = TIER_N3, TIER_K3
    params = make_3layer_params(N1, K1, N2, K2, N3, K3)
    B1, B2, B3 = params["B1"], params["B2"], params["B3"]
    seed = TIER_SEEDS[tier]
    rng = random.Random(seed)

    print(f"  Generating {tier} tier: {bitlength:,} vectors (seed={seed})...")
    print(f"    Architecture: {N1}->{K1}->{K2}->{K3}, "
          f"B1={B1}, B2={B2}, B3={B3}")

    # Random weights and thresholds for all 3 layers
    w1 = [[rng.randint(0, 1) for _ in range(N1)] for _ in range(K1)]
    t1 = [rng.randint(0, N1) for _ in range(K1)]
    w2 = [[rng.randint(0, 1) for _ in range(N2)] for _ in range(K2)]
    t2 = [rng.randint(0, N2) for _ in range(K2)]
    w3 = [[rng.randint(0, 1) for _ in range(N3)] for _ in range(K3)]
    t3 = [rng.randint(0, N3) for _ in range(K3)]

    # Random input streams
    print(f"    Generating x streams ({N1} x {bitlength:,} bits)...")
    x_arrays = {n: getrandbits_large(rng, bitlength) for n in range(N1)}

    weights_list = [w1, w2, w3]
    thresholds_list = [t1, t2, t3]
    input_arrays = encode_3layer_inputs(x_arrays, weights_list,
                                        thresholds_list, params)

    if tier == "large":
        act = _compute_expected_via_bsim(params, input_arrays, bitlength, K3)
    else:
        # Compute expected output via pure Python bitstream reference
        print(f"    Computing expected output via 3-layer bitstream reference...")
        act = bnn_3layer_bitstream_reference(x_arrays, input_arrays, params, bitlength)

    mask = (1 << bitlength) - 1
    for k in range(K3):
        n_active = bin(act[k]).count('1')
        print(f"      neuron {k}: {n_active:,} / {bitlength:,} active "
              f"({100 * n_active / bitlength:.2f}%)")

    masked_act = {k: v & mask for k, v in act.items()}
    data_file = f"bnn_tier_{tier}.bsdata"
    data_path = os.path.join(TESTS_DIR, data_file)
    print(f"    Writing {data_file}...")
    write_bsdata(
        data_path, bitlength,
        params=params,
        input_arrays=input_arrays,
        expected={"act": masked_act},
    )
    size_bytes = os.path.getsize(data_path)
    print(f"    Wrote {data_file} ({size_bytes:,} bytes, {size_bytes / 1e6:.1f} MB)")
    return data_file, size_bytes


def _compute_expected_via_bsim(params, input_arrays, bitlength, K3):
    """Compute expected BNN outputs using C++ bsim backend.

    Writes a temporary inputs-only .bsdata, runs bsim, parses outputs.
    Used for large tier where pure Python is infeasible (~14h).
    """
    print(f"    Computing expected output via C++ bsim backend...")

    # Write inputs-only .bsdata to a temp file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".bsdata")
    os.close(tmp_fd)
    try:
        write_bsdata(tmp_path, bitlength, params=params, input_arrays=input_arrays)
        tmp_size = os.path.getsize(tmp_path)
        print(f"    Wrote temp inputs .bsdata ({tmp_size:,} bytes, "
              f"{tmp_size / 1e9:.2f} GB)")

        # Create ProgramInfo for bnn_mnist.bs
        from simulator.pythonsim.parser import count_stmts
        with open(BS_MNIST_PATH) as f:
            source = f.read()
        program = parse(source)
        prog_info = ProgramInfo(
            name="bnn_mnist",
            source_path=os.path.abspath(BS_MNIST_PATH),
            program=program,
            n_stmts=count_stmts(program),
        )

        # Run bsim (reuse_mem=True is default, needed for memory efficiency)
        # BNN large: 2.7M ops × 4.7M words ≈ 12.7T word-ops → ~3.5h at 1B ops/s
        backend = CppBackend(variant="simd", reuse_mem=True, timeout=14400)
        t0 = time.time()
        result, op_count, exec_ms = backend.run(
            prog_info, inputs={}, bsdata_path=tmp_path)
        t_bsim = time.time() - t0
        print(f"    bsim: {t_bsim:.1f}s ({op_count:,} ops, {exec_ms:.1f}ms exec)")

        # Extract act array from result
        act = {}
        act_raw = result.get("act", {})
        if isinstance(act_raw, dict):
            for k, v in act_raw.items():
                act[k] = v
        else:
            # Single output stream
            act[0] = act_raw

        return act

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_tests_json():
    """Load existing tests.json if it exists."""
    path = os.path.join(TESTS_DIR, "tests.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"bnn_mnist": []}


def save_tests_json(all_tests):
    """Write tests.json (dict keyed by program name)."""
    path = os.path.join(TESTS_DIR, "tests.json")
    with open(path, "w") as f:
        json.dump(all_tests, f, indent=2)
    total = sum(len(v) for v in all_tests.values())
    print(f"  Wrote {path} ({total} entries)")


def merge_tier_entry(tests, new_entry):
    """Merge a tier entry into the test list, replacing any existing entry
    with the same data_file."""
    data_file = new_entry["data_file"]
    for i, entry in enumerate(tests):
        if entry.get("data_file") == data_file:
            tests[i] = new_entry
            return tests
    tests.append(new_entry)
    return tests


def main():
    args = parse_generate_args(DOMAIN)

    # Handle --describe and --verify on existing tests.json
    if args.describe or args.verify:
        tests_json_path = os.path.join(TESTS_DIR, "tests.json")
        if not os.path.exists(tests_json_path):
            print(f"No tests.json found at {tests_json_path}")
            sys.exit(1)
        with open(tests_json_path) as f:
            all_tests = json.load(f)
        if args.describe:
            for prog_name, prog_tests in all_tests.items():
                print(f"{prog_name}: {len(prog_tests)} test entries")
                print_describe(TESTS_DIR, prog_tests)
        if args.verify:
            print(f"bnn: verifying SHA-256 checksums")
            ok = True
            for prog_name, prog_tests in all_tests.items():
                ok = verify_files(TESTS_DIR, prog_tests) and ok
            sys.exit(0 if ok else 1)
        return

    # Parse programs
    with open(BS_MNIST_PATH) as f:
        mnist_program = parse(f.read())

    # -- Tier generation (early return: load existing, merge, save) -----------
    if args.tier:
        tier = args.tier
        bitlength = get_tier_vectors(DOMAIN, tier)
        if bitlength is None:
            print(f"bnn: tier '{tier}' not applicable (skipped)")
            return
        data_file, size_bytes = generate_tier_data(tier, bitlength)
        sha = file_sha256(os.path.join(TESTS_DIR, data_file))
        prov = make_provenance(
            source="synthetic",
            seed=TIER_SEEDS[tier],
            description=(
                f"3-layer BNN with random weights. "
                f"Architecture: {TIER_N1}->{TIER_K1}->{TIER_K2}->{TIER_K3}. "
                f"Input x: {TIER_N1} random bitstreams ({bitlength:,} bits each). "
                f"{bitlength:,} samples."
            ),
            generated_by="generate_tests.py --tier " + tier,
        )
        prov["sha256"] = sha
        entry = tier_test_entry(
            name=TIER_NAMES[tier],
            bitlength=bitlength,
            data_file=data_file,
            size_bytes=size_bytes,
            provenance=prov,
        )
        all_tests = load_tests_json()
        mnist_tests = all_tests.get("bnn_mnist", [])
        mnist_tests = merge_tier_entry(mnist_tests, entry)
        all_tests["bnn_mnist"] = mnist_tests
        save_tests_json(all_tests)
        print(f"bnn: tier '{tier}' merged into tests.json")
        return

    # -- bnn_mnist tests -------------------------------------------------------
    print("Generating bnn_mnist tests...")
    mnist_tests = []

    # Unit tests
    mnist_tests.extend(generate_unit_tests(mnist_program))
    print(f"  {len(mnist_tests)} unit tests generated")

    # Random batch tests
    batch_tests = generate_random_batch_tests(mnist_program)
    mnist_tests.extend(batch_tests)
    print(f"  {len(batch_tests)} random batch tests generated")

    # MNIST accuracy tests
    print("  Generating MNIST accuracy tests...")
    mnist_acc_tests = generate_mnist_tests()
    mnist_tests.extend(mnist_acc_tests)

    # -- Preserve existing tier entries and write tests.json -------------------
    existing = load_tests_json()
    tier_data_files = {f"bnn_tier_{t}.bsdata" for t in TIER_NAMES}
    existing_list = existing.get("bnn_mnist", [])
    for entry in existing_list:
        if entry.get("data_file") in tier_data_files:
            mnist_tests.append(entry)

    all_tests = {"bnn_mnist": mnist_tests}
    save_tests_json(all_tests)
    print(f"bnn: generated {len(mnist_tests)} test entries")


if __name__ == "__main__":
    main()
