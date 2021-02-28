from .abaserunner import ABaseRunner
from .catalyst_runner import CatalystRunner
from .mnist_svhn_runner import MnistSvhnRunner
from fusion.utils import ObjectProvider


runner_provider = ObjectProvider()
runner_provider.register_object('CatalystRunner', CatalystRunner)
runner_provider.register_object('MnistSvhnRunner', MnistSvhnRunner)

__all__ = [
    'ABaseRunner',
    'CatalystRunner',
    'MnistSvhnRunner',
    'runner_provider'
]



