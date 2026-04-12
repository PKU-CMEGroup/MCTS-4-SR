# MCTS-4-SR

![iMCTS](./assets/iMCTS.png)

*Improving Monte Carlo Tree Search for Symbolic Regression*

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![C++](https://img.shields.io/badge/C%2B%2B-20-blue)
![Bindings](https://img.shields.io/badge/bindings-pybind11-brightgreen)

MCTS-4-SR is a C++20 implementation of Monte Carlo Tree Search for symbolic regression, with `pybind11` Python bindings.

This repository provides the C++ search core, the Python package interface (`import imcts`), benchmark tooling, and tests required to build and evaluate the project end to end.

## Overview

- C++20 core implementation for symbolic regression
- Python bindings via `pybind11`
- Benchmark runner for synthetic and black-box tasks
- Example scripts and benchmark tooling
- CMake-based build and test workflow

## Installation

Recommended setup:

```bash
conda create -n imcts python=3.11 -y
conda activate imcts
pip install -U pip
pip install -e .
```

Alternative CMake build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DIMCTS_BUILD_PYTHON=ON -DBUILD_TESTING=ON
cmake --build build -j
```

## Quick Start

Run the bundled example:

```bash
python python/demo.py
```

The Python API can also be used directly:

```python
import numpy as np
import imcts

# One-variable regression dataset with shape [n_vars, n_samples].
x = np.linspace(0, 2, 100, dtype=np.float32).reshape(1, 100)
y = (np.sin(x[0]) + 0.5 * x[0]).astype(np.float32)

cfg = imcts.RegressorConfig()
# Primitive set. Add "R" to enable learnable constants.
cfg.ops = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "R"]

# Maximum expression tree depth.
cfg.max_depth = 6

# MCTS budget per search stage.
cfg.K = 500

# Exploration constant in tree search.
cfg.c = 4.0

# Controls the extreme-bandit style allocation behavior.
cfg.gamma = 0.5

# Probability of applying GP-based state jumping.
cfg.gp_rate = 0.2

# Mutation probability inside GP operations.
cfg.mutation_rate = 0.1

# Additional exploration rate during search.
cfg.exploration_rate = 0.2

# Limit on chained unary operators.
cfg.max_unary = 999

# Maximum number of learnable constants in an expression.
cfg.max_constants = 4

# Iterations for constant optimization.
cfg.lm_iterations = 100

# Stop after this many expression evaluations.
cfg.max_evals = 100000

# Early-stop tolerance for near-perfect solutions.
cfg.succ_error_tol = 1e-6

model = imcts.Regressor(x, y, cfg)

# Set a seed for reproducible runs.
result = model.fit(seed=42)

print(result.best_reward)
print(result.expression)
print(imcts.simplify_expression(result.expression))
```

Alternatively, you can use the default algorithm parameters defined in `include/imcts/regressor.hpp` by simply running:

```python
model = imcts.Regressor(x, y)
```

The `fit()` method returns:

- `best_path`
- `best_coefficients`
- `expression`
- `best_reward`
- `n_evals`

## Benchmarks

List available benchmark cases:

```bash
python -m imcts.benchmarks --list
```

Run Nguyen benchmarks:

```bash
python -m imcts.benchmarks --group Nguyen
```

Run black-box benchmarks (small scale):

```bash
python -m imcts.benchmarks --group BlackBox --cases 1-3 --runs 3
```

Black-box benchmarks expect datasets under `datasets/`. The data format is based on [PMLB](https://github.com/EpistasisLab/pmlb). If a `.tsv.gz` file appears as a Git LFS pointer instead of real data, fetch the dataset contents before running black-box benchmarks.

Benchmark results are written to `benchmark_results/` by default.

## Testing

Run Python smoke tests:

```bash
python python/test_imcts.py
```

Run C++ tests:

```bash
ctest --test-dir build --output-on-failure
```

## Repository Structure

```text
include/imcts/     C++ headers
source/            C++ implementation
imcts/             Python package and benchmark runner
python/            bindings and examples
test/              Catch2 tests
assets/            project image and slides
```

## Notes

- `x_train` is expected to have shape `[n_vars, n_samples]`.
- Add `R` to `ops` when learnable constants are required.
- Invalid numerical expressions are penalized during evaluation.

## Citation

If you use this project, please cite:

> Zhengyao Huang, Daniel Zhengyu Huang, Tiannan Xiao, Dina Ma, Zhenyu Ming, Hao Shi, Yuanhui Wen.  
> *Improving Monte Carlo Tree Search for Symbolic Regression*.  
> https://arxiv.org/abs/2509.15929
