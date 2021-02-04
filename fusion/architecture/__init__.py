from .abasearchitecture import ABaseArchitecture
from .dcgan import DcganEncoder, DcganDecoder

from fusion.utils import ObjectProvider


architecture_provider = ObjectProvider()
architecture_provider.register_object('DcganEncoder', DcganEncoder)
architecture_provider.register_object('DcganDecoder', DcganDecoder)

__all__ = [
    'ABaseArchitecture',
    'DcganEncoder',
    'architecture_provider',
]
