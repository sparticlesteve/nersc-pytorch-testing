import pytest
import torch

def test_gloo():
    assert torch.distributed.is_gloo_available()

def test_mpi():
    assert torch.distributed.is_mpi_available()

def test_mkl():
    assert torch.backends.mkl.is_available()
    assert torch.backends.mkldnn.is_available()

def test_geometric():
    import torch_geometric

def test_vision():
    import torchvision
