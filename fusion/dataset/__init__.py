from .two_view_mnist import TwoViewMnist
from .mnist_svhn import MnistSvhn
from .utils import seed_worker
from fusion.utils import ObjectProvider



dataset_provider = ObjectProvider()
dataset_provider.register_object('TwoViewMnist', TwoViewMnist)
dataset_provider.register_object('MnistSvhn', MnistSvhn)


__all__ = [
    'TwoViewMnist',
    'MnistSvhn',
    'dataset_provider',
    'seed_worker'
]
