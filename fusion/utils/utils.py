import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, input_dim=2):
        super(Unflatten, self).__init__()
        self.input_dim = input_dim

    def forward(self, input_tensor):
        for _ in range(self.input_dim):
            input_tensor = input_tensor.unsqueeze(-1)
        return input_tensor