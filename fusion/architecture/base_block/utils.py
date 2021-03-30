import torch.nn as nn
from torch import Tensor


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor.view(input_tensor.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, input_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, input_tensor: Tensor) -> Tensor:
        for _ in range(self.input_dim):
            input_tensor = input_tensor.unsqueeze(-1)

        return input_tensor
