import torch 
from torch import nn 
from torch import Tensor
import math


class GELU(nn.Module):
    def __init__(self, approximate:str = 'none')->None: 
        super().__init__()
        self.approximate = approximate 
    
    def forward(self, input:Tensor)->Tensor:
        return 0.5 * input * (1 + torch.tanh(math.sqrt(math.pi / 2) * (input + 0.044715 * input ** 3)))