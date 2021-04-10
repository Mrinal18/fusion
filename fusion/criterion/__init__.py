from .loss import CustomCrossEntropyLoss
from .loss import BCEWithLogitsLoss
from .loss import AE
from .loss import SpatialMultiDim, VolumetricMultiDim
from .loss import RR_AE
from .misc import CanonicalCorrelation
import torch.nn as nn
from fusion.utils import ObjectProvider


criterion_provider = ObjectProvider()
criterion_provider.register_object('CCE', CustomCrossEntropyLoss)
criterion_provider.register_object('CE', nn.CrossEntropyLoss)
criterion_provider.register_object('BCE', BCEWithLogitsLoss)
criterion_provider.register_object('AE', AE)
criterion_provider.register_object('SpatialMultiDim', SpatialMultiDim)
criterion_provider.register_object('VolumetricMultiDim', VolumetricMultiDim)
criterion_provider.register_object('CanonicalCorrelation', CanonicalCorrelation)
criterion_provider.register_object('RR_AE', RR_AE)


__all__ = [
    'CustomCrossEntropyLoss',
    'BCEWithLogitsLoss',
    'AE',
    'criterion_provider',
    'SpatialMultiDim',
    'VolumetricMultiDim',
    'CanonicalCorrelation',
    'RR_AE'
]
