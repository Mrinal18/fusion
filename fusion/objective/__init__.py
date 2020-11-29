from .loss import CrossEntropyLoss
from .loss import BCEWithLogitsLoss
from .loss import MSELoss

from fusion.utils import ObjectProvider


__all__ = [
    'CrossEntropyLoss',
    'BCEWithLogitsLoss'
    'MSELoss'
]

objective_provider = ObjectProvider()
objective_provider.register_object('CE', CrossEntropyLoss)
objective_provider.register_object('BCE', BCEWithLogitsLoss)
objective_provider.register_object('MSELoss', MSELoss)
