from .abasearchitecture import ABaseArchitecture
from .dcgan import DcganEncoder, DcganDecoder, DcganAutoEncoder

from fusion.utils import ObjectProvider


architecture_provider = ObjectProvider()
architecture_provider.register_object('DcganEncoder', DcganEncoder)
architecture_provider.register_object('DcganDecoder', DcganDecoder)
architecture_provider.register_object('DcganAutoEncoder', DcganAutoEncoder)


__all__ = [
    'ABaseArchitecture',
    'architecture_provider',
]
