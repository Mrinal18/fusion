from .two_view_mnist import TwoViewMnist
from .mnist_svhn import MnistSvhn
from .oasis import Oasis

from fusion.utils import ObjectProvider


dataset_provider = ObjectProvider()
dataset_provider.register_object("TwoViewMnist", TwoViewMnist)
dataset_provider.register_object("MnistSvhn", MnistSvhn)
dataset_provider.register_object("Oasis", Oasis)
