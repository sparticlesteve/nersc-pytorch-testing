import pytest
import torch

def test_cuda_avail():
    assert torch.cuda.is_available()

def test_cudnn_avail():
    assert torch.backends.cudnn.is_available()

def test_nccl():
    assert torch.distributed.is_nccl_available()

def test_gloo():
    assert torch.distributed.is_gloo_available()

@pytest.mark.skip(reason='not currently deploying pytorch with mpi')
def test_mpi():
    assert torch.distributed.is_mpi_available()

def test_mkl():
    assert torch.backends.mkl.is_available()
    assert torch.backends.mkldnn.is_available()
