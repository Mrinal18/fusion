from .abasearchitecture import ABaseArchitecture
from .dcgan import DcganEncoder

from fusion.utils import ObjectProvider


architecture_provider = ObjectProvider()
architecture_provider.register_object('DcganEncoder', DcganEncoder)

__all__ = [
    'ABaseArchitecture',
    'DcganEncoder',
    'architecture_provider',
]
