from fusion.utils.prototype import ObjectProvider
from .two_view_mnist import TwoViewMnist
from fusion.utils import ObjectProvider


__all__ = ['TwoViewMnist']

dataset_provider = ObjectProvider()
dataset_provider.register_object('TwoViewMnist', TwoViewMnist)
