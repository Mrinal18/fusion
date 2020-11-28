from .abasemodel import AMultiSourceModel, AUniSourceModel
from .supervised import Supervised

from fusion.utils import ObjectProvider


__all__ = [
    'AMultiSourceModel',
    'AUniSourceModel',
    'Supervised',
]

model_provider = ObjectProvider()
model_provider.register_object('Supervised', Supervised)
