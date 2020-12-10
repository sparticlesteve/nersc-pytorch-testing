"""
A simple script which dumps pytorch software information.
"""

import sys
import torch

print('PyTorch module:   ', torch)
print('PyTorch version:  ', torch.__version__)
print()

print('CUDA available:   ', torch.cuda.is_available())
print('CUDA devices:     ', torch.cuda.device_count())
print('CUDA device:      ', torch.cuda.current_device() if torch.cuda.is_available() else None)
print('cuDNN available:  ', torch.backends.cudnn.is_available())
print('cuDNN version:    ', torch.backends.cudnn.version())
print()

print('MKL available:    ', torch.backends.mkl.is_available())
print('MKLDNN available: ', torch.backends.mkldnn.is_available())
print()

print('MPI available:    ', torch.distributed.is_mpi_available())
print('NCCL available:   ', torch.distributed.is_nccl_available())
print('GLOO available:   ', torch.distributed.is_gloo_available())
print()

try:
    import torchvision
    print('torchvision:      ', torchvision.__version__)
except ImportError:
    pass

try:
    import torchtext
    print('torchtext:        ', torchtext.__version__)
except ImportError:
    pass

try:
    import torch_geometric
    print('PyTorch Geometric:', torch_geometric.__version__)
except ImportError:
    pass
