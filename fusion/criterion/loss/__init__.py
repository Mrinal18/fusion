from .abaseloss import ABaseLoss
from .pytorch_wrappers import CrossEntropyLoss
from .pytorch_wrappers import BCEWithLogitsLoss
from .pytorch_wrappers import MSELoss


MAXIMIZE = 1
MINIMIZE = -1

__all__ = [
    'ABaseLoss',
    'CrossEntropyLoss',
    'BCEWithLogitsLoss',
    'MSELoss',
    'MAXIMIZE'
]
