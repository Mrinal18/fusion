import torch
from torchvision import transforms


class MNISTSVHNTransform(object):
    def __call__(self, x):
        x = transforms.ToTensor()(x)
        x = transforms.ToPILImage()(x)
        x = transforms.Resize((32, 32))(x)
        return x


