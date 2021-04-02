from torch import Tensor
from torchvision import transforms


class SVHNTransform:
    """

    """
    def __call__(self, x) -> Tensor:
        """
        Make SVHN transform
        Args:
            :param x: Input tensor
        Return:
            Transform tensor
        """
        x = transforms.ToTensor()(x)
        return x


class MNISTTransform:
    """

    """
    def __call__(self, x) -> Tensor:
        """
        Make MNIST transform
        Args:
            :param x: Input tensor
        Return:
            Transform tensor
        """
        x = transforms.Resize((32, 32))(x)
        x = transforms.ToTensor()(x)
        return x
