from .two_view_mnist import TwoViewMnist
from fusion.utils import ObjectProvider


dataset_provider = ObjectProvider()
dataset_provider.register_object('TwoViewMnist', TwoViewMnist)


__all__ = [
    'TwoViewMnist',
    'dataset_provider',
]
