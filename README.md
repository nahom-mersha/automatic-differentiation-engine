# Automatic Differentiation Engine

A from-scratch scalar automatic-differentiation engine for understanding computational graphs, the chain rule, and reverse-mode backpropagation.

The project represents scalar values, records how they were computed, and automatically calculates gradients through the resulting computational graph.

## AI-assisted learning

This is an AI-assisted learning project. I directed the work, reviewed and tested the implementation, and documented the concepts I learned throughout the project.

## Source of inspiration

This project is independently implemented for learning, but it is strongly inspired by Andrej Karpathy's educational automatic-differentiation project, [`micrograd`](https://github.com/karpathy/micrograd).

The main learning resource is Karpathy's lecture:

[The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)

The project follows the central learning approach demonstrated in Karpathy's `micrograd`: representing scalar values as nodes, recording the operations that produced them, attaching local backward rules, and using the chain rule to propagate gradients through a computational graph. The implementation, tests, explanations, and validation in this repository were written as my own educational reimplementation for Project 7 of my AI Engineering roadmap. They are strongly informed by Karpathy's work, which I credit as the project's primary inspiration.

## Implemented features

- `Value` objects storing scalar values and gradients
- Addition and multiplication
- Powers and derived arithmetic operations
- ReLU and tanh activation functions
- Computational-graph construction
- Topological sorting
- Reverse-mode backpropagation
- Gradient accumulation for shared nodes
- Gradient reset with `zero_grad()`
- Hand-calculation validation
- Central finite-difference gradient checking
- Comparison with PyTorch autograd

## Architecture

Each `Value` object stores:

- `data`: its numerical scalar value;
- `grad`: the gradient accumulated during backpropagation;
- `_prev`: the earlier nodes that produced it;
- `_op`: the operation that produced it;
- `_backward`: the local backward rule for that operation.

During the forward pass, ordinary-looking arithmetic constructs a computational graph. During the backward pass, the engine builds a topological ordering, seeds the final output gradient with `1.0`, and visits the graph in reverse topological order.

Each local backward rule applies the chain rule in the form:

```text
parent gradient += local derivative * upstream gradient
```

## Engineering foundation

The repository uses the reusable project structure developed in Project 0, including:

- `src` layout;
- `pytest`;
- Ruff formatting and linting;
- logging;
- YAML configuration;
- GitHub Actions CI;
- Docker;
- pre-commit hooks.

## Quick start

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

To run the optional PyTorch comparison tests, install the validation dependency:

```bash
pip install -e ".[dev,validation]"
pytest
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

The engine calculates:

```text
y = 7
dy/dx = 3
dy/dw = 2
dy/db = 1
```

## Validation

The implementation is validated through:

- manually calculated linear, polynomial, branched, activation, and composed expressions;
- finite-difference gradient checks using freshly rebuilt graphs for perturbed inputs;
- comparisons of forward values and gradients with PyTorch autograd at matching precision.

## Limitations

This is an educational scalar engine. It intentionally does not provide:

- tensor or matrix operations;
- GPU acceleration;
- broadcasting or shape-aware operations;
- a complete neural-network framework;
- production-level performance or memory optimization.

These capabilities belong to larger tensor-based systems such as PyTorch and are outside this project's scope.

## Learning notes

The detailed explanations and answers to the project's learning questions are maintained in my [AI Notes repository](https://github.com/nahom-mersha/ai-notes), rather than duplicated in this code repository.

## Purpose

This repository is **Project 7** of my AI Engineering roadmap.

Its purpose is to understand how automatic differentiation works internally by implementing a small scalar engine from first principles and validating it against hand calculations, finite differences, and PyTorch autograd.
