from __future__ import annotations

import math
from collections.abc import Callable


class Value:
    def __init__(
        self,
        data: float,
        _children: tuple[Value, ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0

        self._prev = set(_children)
        self._op = _op
        self.label = label

        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: Value | float) -> Value:
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(
            self.data + other.data,
            (self, other),
            "+",
        )

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float) -> Value:
        return self + other

    def __mul__(self, other: Value | float) -> Value:
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(
            self.data * other.data,
            (self, other),
            "*",
        )

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __pow__(self, exponent: float) -> Value:
        out = Value(
            self.data**exponent,
            (self,),
            f"**{exponent}",
        )

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: float) -> Value:
        return other + (-self)

    def __truediv__(self, other: Value | float) -> Value:
        return self * other**-1

    def __rtruediv__(self, other: float) -> Value:
        return other * self**-1

    def relu(self) -> Value:
        out = Value(
            max(0.0, self.data),
            (self,),
            "ReLU",
        )

        def _backward() -> None:
            local_derivative = 1.0 if self.data > 0 else 0.0
            self.grad += local_derivative * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        out = Value(
            math.tanh(self.data),
            (self,),
            "tanh",
        )

        def _backward() -> None:
            local_derivative = 1.0 - out.data**2
            self.grad += local_derivative * out.grad

        out._backward = _backward
        return out

    def _build_topological_order(self) -> list[Value]:
        topological_order: list[Value] = []
        visited: set[Value] = set()

        def build(node: Value) -> None:
            if node in visited:
                return

            visited.add(node)

            for parent in node._prev:
                build(parent)

            topological_order.append(node)

        build(self)
        return topological_order

    def backward(self) -> None:
        topological_order = self._build_topological_order()

        self.grad = 1.0

        for node in reversed(topological_order):
            node._backward()
