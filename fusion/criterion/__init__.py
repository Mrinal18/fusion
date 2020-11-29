from .loss import CrossEntropyLoss
from .loss import BCEWithLogitsLoss
from .loss import MSELoss

from fusion.utils import ObjectProvider


__all__ = [
    'CrossEntropyLoss',
    'BCEWithLogitsLoss'
    'MSELoss'
]

criterion_provider = ObjectProvider()
criterion_provider.register_object('CE', CrossEntropyLoss)
criterion_provider.register_object('BCE', BCEWithLogitsLoss)
criterion_provider.register_object('MSELoss', MSELoss)
