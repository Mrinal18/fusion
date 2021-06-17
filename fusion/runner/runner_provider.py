from .catalyst import CatalystRunner, MnistSvhnRunner
from fusion.utils import ObjectProvider


runner_provider = ObjectProvider()
runner_provider.register_object("CatalystRunner", CatalystRunner)
runner_provider.register_object("MnistSvhnRunner", MnistSvhnRunner)
