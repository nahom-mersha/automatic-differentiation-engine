import pytest

from autodiff_engine.value import Value


def test_linear_expression_matches_hand_calculation() -> None:
    x = Value(2.0)
    w = Value(3.0)
    b = Value(1.0)

    y = w * x + b
    y.backward()

    assert y.data == pytest.approx(7.0)
    assert x.grad == pytest.approx(3.0)
    assert w.grad == pytest.approx(2.0)
    assert b.grad == pytest.approx(1.0)


def test_polynomial_expression_matches_hand_calculation() -> None:
    x = Value(2.0)

    y = x**3 + 2 * x**2
    y.backward()

    # y = x^3 + 2x^2
    # dy/dx = 3x^2 + 4x = 12 + 8 = 20
    assert y.data == pytest.approx(16.0)
    assert x.grad == pytest.approx(20.0)


def test_shared_node_matches_hand_calculation() -> None:
    x = Value(3.0)

    y = x * x + x
    y.backward()

    # dy/dx = 2x + 1 = 7
    assert x.grad == pytest.approx(7.0)


def test_relu_matches_hand_calculation() -> None:
    x = Value(4.0)

    y = x.relu()
    y.backward()

    assert y.data == pytest.approx(4.0)
    assert x.grad == pytest.approx(1.0)


def test_tanh_matches_hand_calculation() -> None:
    x = Value(0.0)

    y = x.tanh()
    y.backward()

    # tanh(0) = 0 and its derivative is 1 - 0^2 = 1
    assert y.data == pytest.approx(0.0)
    assert x.grad == pytest.approx(1.0)


def test_deep_expression_matches_hand_calculation() -> None:
    x = Value(3.0)

    m = 2 * x + 1
    y = m**2
    y.backward()

    # m = 7, y = 49
    # dy/dx = 2m * 2 = 2(7)(2) = 28
    assert y.data == pytest.approx(49.0)
    assert x.grad == pytest.approx(28.0)
