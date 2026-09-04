from autodiff_engine.value import Value


def test_value_stores_data_and_initial_gradient() -> None:
    value = Value(2.0)

    assert value.data == 2.0
    assert value.grad == 0.0
