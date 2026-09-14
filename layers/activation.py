import torch
import torch.nn as nn


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # approximation of the cumulative distribution function of the standard normal distribution
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3)))
        )
    

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("SwiGLU activation function is not implemented yet.")

    def forward(self, x):
        pass
