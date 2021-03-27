from .loss import CrossEntropyLoss
from .loss import BCEWithLogitsLoss
from .loss import MSELoss
from .loss import SpatialMultiDim, VolumetricMultiDim

from fusion.utils import ObjectProvider


criterion_provider = ObjectProvider()
criterion_provider.register_object('CE', CrossEntropyLoss)
criterion_provider.register_object('BCE', BCEWithLogitsLoss)
criterion_provider.register_object('MSELoss', MSELoss)
criterion_provider.register_object('SpatialMultiDim', SpatialMultiDim)
criterion_provider.register_object('VolumetricMultiDim', VolumetricMultiDim)


__all__ = [
    'CrossEntropyLoss',
    'BCEWithLogitsLoss',
    'MSELoss',
    'criterion_provider',
]
