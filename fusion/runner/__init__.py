from .abaserunner import ABaseRunner
from .catalyst_runner import CatalystRunner
from fusion.utils import ObjectProvider


runner_provider = ObjectProvider()
runner_provider.register_object('CatalystRunner', CatalystRunner)

__all__ = [
    'ABaseRunner',
    'CatalystRunner',
    'runner_provider'
]



