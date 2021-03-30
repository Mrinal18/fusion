import torch.nn as nn


class Flatten(nn.Module):

    def __init__(self):
        """
        Class for receive flatten tensor
        Return
            Flatten tensor
        """
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        """
        Method for receive flatten tensor
        Args:
            :param input_tensor: Input tensor for flatten
        Return:
            Flatten tensor
        """
        return input_tensor.view(input_tensor.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, input_dim=2):
        """
        Class for receive unflatten tensor for set dim
        Args:
            :param input_dim: Input dimension

        Return
            unflattering tensor
        """
        super(Unflatten, self).__init__()
        self.input_dim = input_dim

    def forward(self, input_tensor):
        """
        Method for receive unflattering tensor
        Args:
            :param input_tensor: Input tensor for unflattering
        Return
            unflattering tensor
        """
        for _ in range(self.input_dim):
            input_tensor = input_tensor.unsqueeze(-1)
        return input_tensor
