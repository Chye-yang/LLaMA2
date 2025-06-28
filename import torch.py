import torch
from LLaMA2.RMSNorm import RMSNorm

# LLaMA2/test_RMSNorm.py

def test_rmsnorm_output_shape():
    dim = 768
    norm = RMSNorm(dim, 1e-5)
    x = torch.randn(1, 50, dim)
    out = norm(x)
    assert out.shape == (1, 50, dim)

def test_rmsnorm_dtype():
    dim = 768
    norm = RMSNorm(dim, 1e-5)
    x = torch.randn(1, 50, dim)
    out = norm(x)
    assert out.dtype == x.dtype

def test_rmsnorm_weight_is_parameter():
    dim = 768
    norm = RMSNorm(dim, 1e-5)
    assert hasattr(norm, 'weight')
    assert norm.weight.requires_grad

def test_rmsnorm_zero_input():
    dim = 768
    norm = RMSNorm(dim, 1e-5)
    x = torch.zeros(1, 50, dim)
    out = norm(x)
    assert torch.allclose(out, torch.zeros_like(out))

def test_rmsnorm_backward():
    dim = 768
    norm = RMSNorm(dim, 1e-5)
    x = torch.randn(1, 50, dim, requires_grad=True)
    out = norm(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert norm.weight.grad is not Noneimport torch
import pytest
from LLaMA2.RMSNorm import RMSNorm

# LLaMA2/test_RMSNorm.py

def test_rmsnorm_output_shape_and_type():
    dim = 8
    eps = 1e-5
    norm = RMSNorm(dim, eps)
    x = torch.randn(4, dim)
    out = norm(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype

def test_rmsnorm_weight_is_parameter():
    dim = 4
    eps = 1e-6
    norm = RMSNorm(dim, eps)
    assert hasattr(norm, 'weight')
    assert isinstance(norm.weight, torch.nn.Parameter)
    assert norm.weight.requires_grad

def test_rmsnorm_zero_input():
    dim = 6
    eps = 1e-5
    norm = RMSNorm(dim, eps)
    x = torch.zeros(2, dim)
    out = norm(x)
    # 输出应该全为零
    assert torch.allclose(out, torch.zeros_like(out))

def test_rmsnorm_different_eps():
    dim = 5
    x = torch.randn(3, dim)
    norm1 = RMSNorm(dim, 1e-2)
    norm2 = RMSNorm(dim, 1e-8)
    out1 = norm1(x)
    out2 = norm2(x)
    # eps不同，输出应该不同
    assert not torch.allclose(out1, out2)

def test_rmsnorm_backward():
    dim = 7
    norm = RMSNorm(dim, 1e-5)
    x = torch.randn(2, dim, requires_grad=True)
    out = norm(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert norm.weight.grad is not None