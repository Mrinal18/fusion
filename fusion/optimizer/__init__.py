from catalyst.contrib.nn.optimizers.radam import RAdam
from fusion.utils import ObjectProvider


optimizer_provider = ObjectProvider()
optimizer_provider.register_object('RAdam', RAdam)

__all__ = [
    'optimizer_provider'
]
