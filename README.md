# Automatic Differentiation Engine

A from-scratch scalar automatic-differentiation engine for understanding computational graphs, the chain rule, and reverse-mode backpropagation.

The project represents scalar values, records how they were computed, and automatically calculates gradients through the resulting computational graph.

## AI-assisted learning

This is an AI-assisted learning project. I direct the work, review and test the implementation, and document the concepts I learn throughout the project.

## Source of inspiration

This project is independently implemented for learning, but it is strongly inspired by Andrej Karpathy's educational automatic-differentiation project, [`micrograd`](https://github.com/karpathy/micrograd).

The main learning resource is Karpathy's lecture:

[The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)

The project follows the same central learning idea: represent scalar values as nodes, record the operations that produced them, attach local backward rules, and use the chain rule to propagate gradients through a computational graph. The implementation, tests, explanations, and validation in this repository are my own Project 7 work within my AI Engineering roadmap, informed by Karpathy’s micrograd as a learning reference.

## Planned Features

- `Value` objects storing numerical values and gradients
- Addition and multiplication
- Powers and derived arithmetic operations
- Basic activation functions
- Computational-graph construction
- Topological sorting
- Reverse-mode backpropagation
- Gradient accumulation
- Hand-calculation validation
- Finite-difference gradient checking
- Comparison with PyTorch autograd

## Engineering Foundation

The repository uses the reusable project structure developed in Project 0, including:

- `src` layout
- `pytest`
- Ruff formatting and linting
- Logging
- YAML configuration
- GitHub Actions CI
- Docker
- Pre-commit hooks

## Quick Start

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest
```

Format and lint the project:

```bash
ruff format .
ruff check .
```

Build the Docker image:

```bash
docker build -t automatic-differentiation-engine .
```

Run the Docker container:

```bash
docker run --rm automatic-differentiation-engine
```

## Example

The intended API will support expressions such as:

```python
from autodiff_engine import Value

x = Value(2.0)
w = Value(3.0)
b = Value(1.0)

y = w * x + b
y.backward()

print(y.data)
print(x.grad)
print(w.grad)
print(b.grad)
```

This represents:

```text
y = w*x + b
```

The engine should calculate:

```text
y = 7
dy/dx = 3
dy/dw = 2
dy/db = 1
```

## Learning Notes

Learning notes will be added after the implementation and validation stages are complete.

## Purpose

This repository is **Project 7** of my AI Engineering roadmap.

Its purpose is to understand how automatic differentiation works internally by implementing a small scalar engine from first principles and validating it against hand calculations, finite differences, and PyTorch autograd.
