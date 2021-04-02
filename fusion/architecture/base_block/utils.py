import torch.nn as nn


class Flatten(nn.Module):

    def __init__(self):
        """
        Custom Pytorch module that flattens an input tensor
        
        Return:
            Flattened tensor
        """
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        """
        The forward function that receives an input tensor and flattens it
        
        Args:
            :param input_tensor: Input tensor to flatten
        Return:
            Flattened tensor
        """
        # Flatten tensor using .view() to avoid memory copies
        return input_tensor.view(input_tensor.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, input_dim=2):
        """
        Custom Pytorch module that unflattens the tensor for a set number of dimensions
        
        Args:
            :param input_dim: Input dimension, the input tensor is unsqueezed input_dim times

        Return:
            Unflattened tensor
        """
        super(Unflatten, self).__init__()
        self.input_dim = input_dim

    def forward(self, input_tensor):
        """
        The forward function that receives a tensor and unflattens it self.input_dim times
        
        Args:
            :param input_tensor: Input tensor to unflatten
        Return:
            Unflattened tensor
        """
        for _ in range(self.input_dim):
            input_tensor = input_tensor.unsqueeze(-1)
        return input_tensor
