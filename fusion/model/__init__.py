from .abasemodel import AMultiSourceModel, AUniSourceModel
from .supervised import Supervised

from fusion.utils import ObjectProvider


model_provider = ObjectProvider()
model_provider.register_object('Supervised', Supervised)
model_provider.register_object('AMultiSourceModel', AMultiSourceModel)

__all__ = [
    'AMultiSourceModel',
    'AUniSourceModel',
    'Supervised',
    'model_provider',
]
