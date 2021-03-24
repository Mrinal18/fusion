from .base_dim import BaseDim
from .cr_dim import CrDim
from .xx_dim import XxDim
from .cc_dim import CcDim
from .rr_dim import RrDim
from fusion.utils import ObjectProvider


dim_mode_provider = ObjectProvider()
dim_mode_provider.register_object('RR', RrDim)
dim_mode_provider.register_object('CR', CrDim)
dim_mode_provider.register_object('XX', XxDim)
dim_mode_provider.register_object('CC', CcDim)


__all__ = [
    'BaseDim',
    'CrDim',
    'XxDim',
    'CcDim',
    'RrDim',
    'dim_mode_provider'
]