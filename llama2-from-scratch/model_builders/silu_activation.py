"""
Created at 05/06/26
@author: raif.viren@gmail.com
"""
from torch import nn
import torch


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
