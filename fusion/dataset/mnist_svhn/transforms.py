import torch
from torchvision import transforms


class SVHNTransform(object):
    def __call__(self, x):
        x = transforms.ToTensor()(x)
        return x


class MNISTTransform(object):
    def __call__(self, x):
        x = transforms.Resize((32, 32))(x)
        x = transforms.ToTensor()(x)
        return x