import torch
from torch import Tensor
from torchvision import transforms

from fusion.dataset.abasetransform import ABaseTransform


class UnitIntervalScale(ABaseTransform):
    def __call__(self, x):
        x = (x - x.min()) / (x.max() - x.min())
        return x


class RandomRotation(ABaseTransform):
    def __init__(self, degrees : int = 45):
        self.random_rotation = transforms.RandomRotation(degrees, fill=(0,))

    def __call__(self, x):
        x = self.random_rotation(x)
        x = transforms.ToTensor()(x)
        return x


class UniformNoise(ABaseTransform):
    def __call__(self, x) -> Tensor:
        x = transforms.ToTensor()(x)
        x = x + torch.rand(x.size())
        x = torch.clamp(x, min=0., max=1.)
        return x


class TwoViewMnistTransform(ABaseTransform):
    def __call__(self, x):
        x = transforms.ToTensor()(x)
        x = UnitIntervalScale()(x)
        x = transforms.ToPILImage()(x)
        x = transforms.Resize((32, 32))(x)
        v1 = RandomRotation()(x)
        v2 = UniformNoise()(x)
        return (v1, v2)


class RandomRotationTransform(ABaseTransform):
    def __call__(self, x):
        x = transforms.ToTensor()(x)
        x = UnitIntervalScale()(x)
        x = transforms.ToPILImage()(x)
        x = transforms.Resize((32, 32))(x)
        x = RandomRotation()(x)
        return (x,)


class UniformNoiseTransform(ABaseTransform):
    def __call__(self, x):
        x = transforms.ToTensor()(x)
        x = UnitIntervalScale()(x)
        x = transforms.ToPILImage()(x)
        x = transforms.Resize((32, 32))(x)
        x = UniformNoise()(x)
        return (x,)
