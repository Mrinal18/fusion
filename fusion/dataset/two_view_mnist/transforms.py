import torch
from torchvision import transforms


class UnitIntervalScale:
    def __call__(self, x):
        x = (x - x.min()) / (x.max() - x.min())
        return x


class RandomRotation:
    def __init__(self, degrees=45):
        self.random_rotation = transforms.RandomRotation(degrees, fill=(0,))

    def __call__(self, x):
        x = self.random_rotation(x)
        x = transforms.ToTensor()(x)
        return x


class UniformNoise:
    def __call__(self, x):
        x = transforms.ToTensor()(x)
        x = x + torch.rand(x.size())
        x = torch.clamp(x, min=0., max=1.)
        return x


class TwoViewMnistTransform:
    def __call__(self, x):
        x = transforms.ToTensor()(x)
        x = UnitIntervalScale()(x)
        x = transforms.ToPILImage()(x)
        x = transforms.Resize((32, 32))(x)
        v1 = RandomRotation()(x)
        v2 = UniformNoise()(x)
        return (v1, v2)


class RandomRotationTransform:
    def __call__(self, x):
        x = transforms.ToTensor()(x)
        x = UnitIntervalScale()(x)
        x = transforms.ToPILImage()(x)
        x = transforms.Resize((32, 32))(x)
        x = RandomRotation()(x)
        return (x,)


class UniformNoiseTransform:
    def __call__(self, x):
        x = transforms.ToTensor()(x)
        x = UnitIntervalScale()(x)
        x = transforms.ToPILImage()(x)
        x = transforms.Resize((32, 32))(x)
        x = UniformNoise()(x)
        return (x,)
