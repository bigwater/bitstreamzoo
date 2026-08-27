#!/usr/bin/env python3
"""Train a BNN on MNIST and export weights/thresholds for bnn.bs.

Architecture: FINN LFC style 784 -> 1024 -> 1024 -> 1024 -> 10
- Binary weights with STE (straight-through estimator)
- Batch normalization after each hidden layer
- Post-training: fold batch norm into integer thresholds

Reference: Umuroglu et al., "FINN", FPGA 2017 — LFC achieves 98.4% on MNIST

Requirements: pip install torch torchvision numpy

Output: mnist_params.npz containing:
  w1(H x 784), w2(H x H), w3(H x H), w4(10 x H) — binary {0,1}, H=1024
  t1(H), t2(H), t3(H) — integer thresholds for hidden layers
  test_images(10000x784), test_labels(10000) — binarized MNIST test set
  accuracy — training-time accuracy on the subset
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ── STE binarization ─────────────────────────────────────────────

class SignSTE(torch.autograd.Function):
    """Sign function with straight-through estimator (hardtanh gradient)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # STE: pass gradient through where |x| <= 1
        return grad_output * (x.abs() <= 1).float()


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    return SignSTE.apply(x)


# ── Binary layer with STE ────────────────────────────────────────

class BinaryLinear(nn.Module):
    """Fully-connected layer with binarized weights (sign function + STE)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_bin = sign_ste(self.linear.weight)
        return F.linear(x, w_bin)


# ── BNN model ────────────────────────────────────────────────────

class BNN(nn.Module):
    """784 -> 1024 -> 1024 -> 1024 -> 10 binary neural network (FINN LFC)."""

    def __init__(self, hidden: int = 1024):
        super().__init__()
        self.fc1 = BinaryLinear(784, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = BinaryLinear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = BinaryLinear(hidden, hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.fc4 = BinaryLinear(hidden, 10)  # no BN on last layer
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        # Binarize activations with STE
        x = sign_ste(self.bn1(self.fc1(x)))
        x = sign_ste(self.bn2(self.fc2(x)))
        x = sign_ste(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # raw logits for cross-entropy
        return x


# ── Batch norm folding ───────────────────────────────────────────

def fold_bn(fc: BinaryLinear, bn: nn.BatchNorm1d, N: int):
    """Fold batch norm into binary weights and integer threshold.

    For binary {-1,+1} weights and inputs, the pre-BN output is:
        z = 2 * popcount(XNOR(w, x)) - N

    Batch norm computes:
        y = gamma * (z - mu) / sqrt(var + eps) + beta

    Activation is sign(y), so we need y >= 0:
        gamma * (z - mu) / sqrt(var + eps) + beta >= 0

    When gamma > 0:
        z >= mu - beta * sqrt(var + eps) / gamma

    When gamma < 0 (flip inequality):
        z <= mu - beta * sqrt(var + eps) / gamma
        We flip the weight bits and negate the threshold direction.

    Since z = 2*popcount - N, the threshold on popcount is:
        popcount >= ceil((z_thresh + N) / 2)

    Returns: (weights_01, thresholds) where weights are {0,1}
    """
    with torch.no_grad():
        w_bin = fc.linear.weight.sign()  # {-1, +1}
        gamma = bn.weight.data
        beta = bn.bias.data
        mu = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        K = w_bin.shape[0]
        weights_01 = torch.zeros_like(w_bin)
        thresholds = torch.zeros(K, dtype=torch.long)

        for k in range(K):
            g = gamma[k].item()
            b = beta[k].item()
            m = mu[k].item()
            v = var[k].item()
            std = math.sqrt(v + eps)

            # z_thresh: minimum z value for positive activation
            z_thresh = m - b * std / g

            if g < 0:
                # Flip weight bits to reverse the comparison direction
                weights_01[k] = ((- w_bin[k]) + 1) / 2  # {-1,+1} -> {0,1} with flip
                # When gamma < 0, sign flips: need z <= z_thresh
                # After flipping weights, XNOR becomes XOR, so popcount_new = N - popcount_old
                # z_new = 2*popcount_new - N = -(2*popcount_old - N) = -z_old
                # Need -z_old <= z_thresh, i.e., z_old >= -z_thresh
                z_thresh = -z_thresh
            else:
                weights_01[k] = (w_bin[k] + 1) / 2  # {-1,+1} -> {0,1}

            # Convert z threshold to popcount threshold: z = 2*pop - N => pop = (z+N)/2
            pop_thresh = (z_thresh + N) / 2.0
            thresholds[k] = max(0, min(N, math.ceil(pop_thresh)))

    return weights_01.to(torch.uint8).numpy(), thresholds.numpy()


# ── Scalar reference for cross-validation ────────────────────────

def scalar_bnn_predict(images, w1, t1, w2, t2, w3, t3, w4):
    """Pure-Python scalar BNN inference for cross-validation.

    All weights are {0,1} numpy arrays, thresholds are int arrays.
    images: (M, 784) binary {0,1} numpy array.
    Returns: (M,) predicted class labels.
    """
    M = images.shape[0]
    H = w1.shape[0]  # hidden size (e.g. 256)
    preds = np.zeros(M, dtype=np.int64)

    for m in range(M):
        # Layer 1: 784 -> H
        x = images[m]  # (784,) {0,1}
        act1 = np.zeros(H, dtype=np.uint8)
        for k in range(H):
            count = np.sum(x == w1[k])
            act1[k] = 1 if count >= t1[k] else 0

        # Layer 2: H -> H
        act2 = np.zeros(H, dtype=np.uint8)
        for k in range(H):
            count = np.sum(act1 == w2[k])
            act2[k] = 1 if count >= t2[k] else 0

        # Layer 3: H -> H
        act3 = np.zeros(H, dtype=np.uint8)
        for k in range(H):
            count = np.sum(act2 == w3[k])
            act3[k] = 1 if count >= t3[k] else 0

        # Layer 4: H -> 10 (no threshold, just XNOR-popcount -> argmax)
        scores = np.zeros(10, dtype=np.int64)
        for k in range(10):
            scores[k] = np.sum(act3 == w4[k])
        preds[m] = np.argmax(scores)

    return preds


# ── Training ─────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data: binarize pixels to {-1, +1}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x.view(-1) > 0.5).float() * 2 - 1),
    ])

    data_dir = os.path.join(os.path.dirname(__file__), "mnist_data")
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

    model = BNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Train
    n_epochs = 100
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Clamp weights to [-1, 1]
            with torch.no_grad():
                for m in [model.fc1, model.fc2, model.fc3, model.fc4]:
                    m.linear.weight.clamp_(-1, 1)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            train_acc = correct / total
            print(f"  Epoch {epoch:3d}  loss={total_loss/total:.4f}  train_acc={train_acc:.4f}")

    # Evaluate on full test set (PyTorch)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    pytorch_acc = correct / total
    print(f"\nPyTorch test accuracy: {pytorch_acc:.4f} ({correct}/{total})")

    # ── Fold batch norm and extract parameters ────────────────────
    model.cpu()

    H = model.hidden
    w1_01, t1 = fold_bn(model.fc1, model.bn1, 784)
    w2_01, t2 = fold_bn(model.fc2, model.bn2, H)
    w3_01, t3 = fold_bn(model.fc3, model.bn3, H)
    w4_01 = ((model.fc4.linear.weight.sign() + 1) / 2).to(torch.uint8).detach().numpy()

    # ── Binarize test images ──────────────────────────────────────
    # All 10K test images for GPU-scale benchmarks
    test_images_raw = test_ds.data.numpy()  # (10000, 28, 28)
    test_labels = test_ds.targets.numpy()
    test_images = (test_images_raw.reshape(10000, 784) > 127).astype(np.uint8)

    # ── Cross-validate: scalar reference vs PyTorch ───────────────
    print("\nCross-validating scalar reference vs PyTorch (100 images)...")
    n_xval = 100
    scalar_preds = scalar_bnn_predict(
        test_images[:n_xval], w1_01, t1, w2_01, t2, w3_01, t3, w4_01
    )

    # PyTorch predictions on same images (binarized to {-1, +1})
    with torch.no_grad():
        xval_input = torch.tensor(test_images[:n_xval], dtype=torch.float32) * 2 - 1
        pytorch_preds = model(xval_input).argmax(1).numpy()

    n_match = np.sum(scalar_preds == pytorch_preds)
    print(f"  Scalar vs PyTorch match: {n_match}/{n_xval}")
    if n_match != n_xval:
        # Show mismatches for debugging
        mismatches = np.where(scalar_preds != pytorch_preds)[0]
        for idx in mismatches[:5]:
            print(f"    Image {idx}: scalar={scalar_preds[idx]}, pytorch={pytorch_preds[idx]}")
        print(f"  WARNING: {n_xval - n_match} mismatches detected!")
    else:
        print("  Perfect match — batch norm folding is correct.")

    # Compute accuracy of the folded model on the first 1000 test images
    # (full 10K cross-validation is done by PyTorch above)
    full_scalar_preds = scalar_bnn_predict(
        test_images[:1000], w1_01, t1, w2_01, t2, w3_01, t3, w4_01
    )
    accuracy = np.mean(full_scalar_preds == test_labels[:1000])
    print(f"\nScalar BNN accuracy on 1000 test images: {accuracy:.4f}")

    # ── Save ──────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "..", "small", "mnist_params.npz")
    np.savez_compressed(
        out_path,
        w1=w1_01, w2=w2_01, w3=w3_01, w4=w4_01,
        t1=t1, t2=t2, t3=t3,
        test_images=test_images,
        test_labels=test_labels,
        accuracy=accuracy,
    )
    file_size = os.path.getsize(out_path)
    print(f"\nSaved {out_path} ({file_size / 1024:.1f} KB)")
    print(f"  Weights: w1{w1_01.shape} w2{w2_01.shape} w3{w3_01.shape} w4{w4_01.shape}")
    print(f"  Thresholds: t1{t1.shape} t2{t2.shape} t3{t3.shape}")
    print(f"  Test data: {test_images.shape} images, {test_labels.shape} labels")
    print(f"  Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train()
