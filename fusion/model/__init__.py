from .abasemodel import AMultiSourceModel, AUniSourceModel
from .supervised import Supervised
from .misc import ModelOutput
from .ae import AE
from .dim import Dim

from fusion.utils import ObjectProvider


model_provider = ObjectProvider()
model_provider.register_object('Supervised', Supervised)

__all__ = [
    'AMultiSourceModel',
    'AUniSourceModel',
    'Supervised',
    'AE',
    'Dim',
    'ModelOutput',
    'model_provider',
]
