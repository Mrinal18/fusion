import torch
from torch import Tensor
from torchvision import transforms

from fusion.dataset.abasetransform import ABaseTransform


class UnitIntervalScale(ABaseTransform):
    def __call__(self, x):
        """
        Make  Unit Interval Scale transform
        Args:
            :param x: Input tensor
        Return:
            Transform tensor
        """

        x = (x - x.min()) / (x.max() - x.min())
        return x


class RandomRotation(ABaseTransform):
    def __init__(self, degrees : int = 45):
        """
        Initialization  Class Random Rotation transform
        Args:
            :param degrees: Max angle
        Return:
            Class Random Rotation transform
        """
        self.random_rotation = transforms.RandomRotation(degrees, fill=(0,))

    def __call__(self, x):
        """
        Make  Random Rotation transform
        Args:
            :param x: Input tensor
        Return:
            Transform tensor
        """
        x = self.random_rotation(x)
        x = transforms.ToTensor()(x)
        return x


class UniformNoise(ABaseTransform):
    def __call__(self, x) -> Tensor:
        """
        Make  Uniform Noise transform
        Args:
            :param x: Input tensor
        Return:
            Transform tensor
        """
        x = transforms.ToTensor()(x)
        x = x + torch.rand(x.size())
        x = torch.clamp(x, min=0., max=1.)
        return x


class TwoViewMnistTransform(ABaseTransform):
    def __call__(self, x):
        """
        Make  Two View Mnist transform
        Args:
            :param x: Input tensor
        Return:
            Transform tensor
        """
        x = transforms.ToTensor()(x)
        x = UnitIntervalScale()(x)
        x = transforms.ToPILImage()(x)
        x = transforms.Resize((32, 32))(x)
        v1 = RandomRotation()(x)
        v2 = UniformNoise()(x)
        return (v1, v2)


class RandomRotationTransform(ABaseTransform):
    def __call__(self, x):
        """
        Make  Random Rotation transform
        Args:
            :param x: Input tensor
        Return:
             Transform tensor
        """
        x = transforms.ToTensor()(x)
        x = UnitIntervalScale()(x)
        x = transforms.ToPILImage()(x)
        x = transforms.Resize((32, 32))(x)
        x = RandomRotation()(x)
        return (x,)


class UniformNoiseTransform(ABaseTransform):
    def __call__(self, x):
        """
        Make  Uniform Noise transform
        Args:
            :param x: Input tensor
        Return:
             Transform tensor
        """
        x = transforms.ToTensor()(x)
        x = UnitIntervalScale()(x)
        x = transforms.ToPILImage()(x)
        x = transforms.Resize((32, 32))(x)
        x = UniformNoise()(x)
        return (x,)
