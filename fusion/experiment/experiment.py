class BaseExperiment:
    _arguments = {}

    def __init__(self):
        self.__dict__ = self._arguments


class Experiment(BaseExperiment):
    # Singleton
    # To have global within experiments arguments

    def __init__(self, config_filename):
        BaseExperiment.__init__(self)
        config_args = self.read_config(config_filename)
        self._arguments.update(config_args)

    def read_config(config_filename):
        pass
