import numpy as np
import torch
from torch import Tensor
from typing import Optional
from torch import nn

def sample_batch(sigma: float, size: tuple) -> Tensor:
    return torch.randn(size) * sigma

def gaussian_encoding( v: Tensor, b: Tensor) -> Tensor:
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)

class GaussianEncoding(nn.Module):
    def __init__(self, sigma, input_size, encoded_size):
        super().__init__()
        self.batch = nn.parameter.Parameter(sample_batch(sigma, (encoded_size, input_size)), requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        return gaussian_encoding(v, self.batch)