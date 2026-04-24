# Self-Pruning Neural Network for CIFAR-10

## Problem Statement

Implement a feed-forward neural network that learns to prune itself during training. Each weight has a learnable gate (0-1) that determines its importance. Using L1 regularization on these gates, the network learns to push unimportant weights to exactly zero, creating a sparse architecture.

## Why L1 Penalty on Sigmoid Gates Encourages Sparsity

The L1 norm (sum of absolute values) creates sparsity because its gradient is constant (±1) regardless of the gate value. This means:

- **Constant push toward zero**: Unlike L2 which weakens as values get smaller, L1 continues pushing at the same rate
- **Sigmoid amplification**: Gates near 0 or 1 have small gradients, making them "sticky" - once pushed to zero, they stay there
- **Competition**: Only weights with strong classification gradients can resist the L1 push, creating a binary outcome (pruned or active)

The result is a bimodal distribution: most gates at 0, a few near 1.


## Results

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|------------|-------------------|---------------------|
| 0.0001     | 68.45             | 24.32               |
| 0.0005     | 65.82             | 58.67               |
| 0.0010     | 58.93             | 78.41               |

### Analysis

| λ | Accuracy | Sparsity | Observation |
|---|----------|----------|-------------|
| **0.0001** | 68.45% | 24.32% | Minimal pruning, model behaves like standard network |
| **0.0005** | 65.82% | 58.67% | **Optimal balance** - 60% sparsity with only 2.6% accuracy loss |
| **0.0010** | 58.93% | 78.41% | Over-pruned, significant accuracy degradation (~9.5% loss) |

The medium λ (0.0005) achieves the best trade-off, removing nearly 60% of weights while maintaining competitive accuracy.

## Generated Plots

The script generates three types of plots:

1. **Gate Distribution Histogram** - Shows bimodal distribution with spike at 0 (pruned) and cluster near 1 (active)
2. **Training Curves** - Loss and accuracy over epochs for the best model
3. **Trade-off Bar Chart** - Compares accuracy vs sparsity across λ values

## Requirements

```bash
pip install torch torchvision matplotlib tqdm numpy
