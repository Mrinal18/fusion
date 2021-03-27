from .abasemodel import ABaseModel
from .supervised import Supervised
from .misc import ModelOutput
from .ae import AE
from .dim import Dim
from .linear_evaluator import LinearEvaluator

from fusion.utils import ObjectProvider


model_provider = ObjectProvider()
model_provider.register_object('Supervised', Supervised)
model_provider.register_object('Dim', Dim)
model_provider.register_object('LinearEvaluator', LinearEvaluator)

__all__ = [
    'ABaseModel',
    'Supervised',
    'AE',
    'Dim',
    'ModelOutput',
    'model_provider',
]
