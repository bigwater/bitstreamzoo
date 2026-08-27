# BNN Inference -- Real-Life Data Provenance

## Source

MNIST handwritten digits.

## Origin

- **MNIST**: http://yann.lecun.com/exdb/mnist/
  - 60K training + 10K test grayscale 28x28 images of digits 0-9.
  - LeCun et al., 1998.

## Pre-trained parameters

Training scripts already reside in this directory:
- `train_mnist.py`: Trains 784->1024->1024->1024->10 BNN (~98% accuracy).
  Outputs `mnist_params.npz` (weights, thresholds, test images/labels).

It uses PyTorch with a straight-through estimator (STE) for binary
weights and folds batch normalization into integer thresholds
post-training.

## Download

MNIST is auto-downloaded by `train_mnist.py` via torchvision.

## Conversion

`convert.py` is a wrapper that invokes the training scripts. Images are
binarized (pixel > 0.5 -> 1) and flattened to 1D binary vectors.

## References

- Hubara et al., "Binarized Neural Networks", NeurIPS 2016
- Umuroglu et al., "FINN", FPGA 2017
