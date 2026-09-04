from __future__ import annotations

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
