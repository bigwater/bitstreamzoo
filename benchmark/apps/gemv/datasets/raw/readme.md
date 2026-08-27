# GEMV Synthetic Data

GEMV uses synthetic data: random K-bit weight matrices and input vectors.

## 3 Tier Configurations

| Config | L | N | K | DNN analogue |
|--------|---|---|---|-------------|
| `small_k2` | 16 | 16 | 2 | Small embedding layer, 2-bit precision |
| `medium_k4` | 64 | 64 | 4 | Hidden layer, INT4 precision |
| `large_k8` | 64 | 128 | 8 | Large hidden layer, INT8 precision |

Weights are uniformly random in [0, 2^K - 1]. Input vectors are random
bitstreams. Seeds are deterministic per config.
