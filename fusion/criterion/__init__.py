from .loss import CustomCrossEntropyLoss
from .loss import BCEWithLogitsLoss
from .loss import AE
from .loss import SpatialMultiDim, VolumetricMultiDim
import torch.nn as nn
from fusion.utils import ObjectProvider


criterion_provider = ObjectProvider()
criterion_provider.register_object('CCE', CustomCrossEntropyLoss)
criterion_provider.register_object('CE', nn.CrossEntropyLoss)
criterion_provider.register_object('BCE', BCEWithLogitsLoss)
criterion_provider.register_object('AE', AE)
criterion_provider.register_object('SpatialMultiDim', SpatialMultiDim)
criterion_provider.register_object('VolumetricMultiDim', VolumetricMultiDim)


__all__ = [
    'CrossEntropyLoss',
    'BCEWithLogitsLoss',
    'AE',
    'criterion_provider',
    'SpatialMultiDim',
    'VolumetricMultiDim'
]
