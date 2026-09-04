from autodiff_engine.value import Value


def test_value_stores_data_and_initial_gradient() -> None:
    value = Value(2.0)

    assert value.data == 2.0
    assert value.grad == 0.0


def test_addition_computes_forward_value_and_parents() -> None:
    a = Value(2.0)
    b = Value(3.0)

    result = a + b

    assert result.data == 5.0
    assert result._op == "+"
    assert result._prev == {a, b}


def test_addition_backward_rule_passes_gradient_to_both_inputs() -> None:
    a = Value(2.0)
    b = Value(3.0)

    result = a + b
    result.grad = 1.0

    result._backward()

    assert a.grad == 1.0
    assert b.grad == 1.0


def test_addition_with_python_number() -> None:
    a = Value(2.0)

    result = a + 3

    assert result.data == 5.0


def test_reflected_addition_with_python_number() -> None:
    a = Value(2.0)

    result = 3 + a

    assert result.data == 5.0


def test_multiplication_computes_forward_value_and_parents() -> None:
    a = Value(2.0)
    b = Value(3.0)

    result = a * b

    assert result.data == 6.0
    assert result._op == "*"
    assert result._prev == {a, b}


def test_multiplication_backward_rule_uses_other_operand() -> None:
    a = Value(2.0)
    b = Value(3.0)

    result = a * b
    result.grad = 1.0

    result._backward()

    assert a.grad == 3.0
    assert b.grad == 2.0


def test_multiplication_with_python_number() -> None:
    a = Value(2.0)

    assert (a * 3).data == 6.0


def test_reflected_multiplication_with_python_number() -> None:
    a = Value(2.0)

    assert (3 * a).data == 6.0


def test_power_computes_forward_value_and_parent() -> None:
    a = Value(3.0)

    result = a**2

    assert result.data == 9.0
    assert result._op == "**2"
    assert result._prev == {a}


def test_power_backward_rule_uses_power_rule() -> None:
    a = Value(3.0)

    result = a**2
    result.grad = 1.0

    result._backward()

    assert a.grad == 6.0


def test_power_with_fractional_exponent() -> None:
    a = Value(4.0)

    result = a**0.5
    result.grad = 1.0

    result._backward()

    assert result.data == 2.0
    assert a.grad == 0.25


def test_negation() -> None:
    a = Value(3.0)

    result = -a
    result.grad = 1.0
    result._backward()

    assert result.data == -3.0
    assert a.grad == -1.0


def test_subtraction() -> None:
    a = Value(5.0)
    b = Value(2.0)

    result = a - b

    assert result.data == 3.0


def test_reflected_subtraction() -> None:
    a = Value(2.0)

    result = 5 - a

    assert result.data == 3.0


def test_division() -> None:
    a = Value(6.0)
    b = Value(2.0)

    result = a / b

    assert result.data == 3.0
