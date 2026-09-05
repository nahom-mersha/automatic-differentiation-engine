import pytest
import torch

from autodiff_engine.value import Value


def test_matches_pytorch_autograd() -> None:
    # Our engine
    x = Value(1.5)
    w = Value(2.0)
    b = Value(0.5)

    result = (w * x + b).tanh()
    result.backward()

    # PyTorch reference
    torch_x = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    torch_w = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    torch_b = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    torch_result = torch.tanh(torch_w * torch_x + torch_b)
    torch_result.backward()

    assert torch_x.grad is not None
    assert torch_w.grad is not None
    assert torch_b.grad is not None
    assert result.data == pytest.approx(torch_result.item())
    assert x.grad == pytest.approx(torch_x.grad.item())
    assert w.grad == pytest.approx(torch_w.grad.item())
    assert b.grad == pytest.approx(torch_b.grad.item())


def test_shared_node_matches_pytorch() -> None:
    # Our engine
    x = Value(3.0)
    result = x * x + x
    result.backward()

    # PyTorch reference
    torch_x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    torch_result = torch_x * torch_x + torch_x
    torch_result.backward()
    assert torch_x.grad is not None

    assert x.grad == pytest.approx(torch_x.grad.item())
