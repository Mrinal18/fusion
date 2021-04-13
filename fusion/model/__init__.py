from .abasemodel import ABaseModel
from .supervised import Supervised
from .ae import AE
from .dim import Dim
from .linear_evaluator import LinearEvaluator

from fusion.utils import ObjectProvider


model_provider = ObjectProvider()
model_provider.register_object('Supervised', Supervised)
model_provider.register_object('AE', AE)
model_provider.register_object('Dim', Dim)
model_provider.register_object('LinearEvaluator', LinearEvaluator)

__all__ = [
    'ABaseModel',
    'Supervised',
    'AE',
    'Dim',
    'model_provider',
]
