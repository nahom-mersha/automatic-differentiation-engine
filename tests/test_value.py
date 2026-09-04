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
