from .abaserunner import ABaseRunner
from .catalyst_runner import CatalystRunner
from fusion.utils import ObjectProvider


__all__ = [
    'ABaseRuner',
    'CatalystRunner',
]


runner_provider = ObjectProvider()
runner_provider.register_object('CatalystRunner', CatalystRunner)
