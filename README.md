# Self-Pruning Neural Network for CIFAR-10

## Problem Statement

Implement a feed-forward neural network that learns to prune itself during training. Each weight has a learnable gate (0-1) that determines its importance. Using L1 regularization on these gates, the network learns to push unimportant weights to exactly zero, creating a sparse architecture.

## Why L1 Penalty on Sigmoid Gates Encourages Sparsity

The L1 norm (sum of absolute values) creates sparsity because its gradient is constant (±1) regardless of the gate value. This means:

- **Constant push toward zero**: Unlike L2 which weakens as values get smaller, L1 continues pushing at the same rate
- **Sigmoid amplification**: Gates near 0 or 1 have small gradients, making them "sticky" - once pushed to zero, they stay there
- **Competition**: Only weights with strong classification gradients can resist the L1 push, creating a binary outcome (pruned or active)

The result is a bimodal distribution: most gates at 0, a few near 1.

## Network Architecture
