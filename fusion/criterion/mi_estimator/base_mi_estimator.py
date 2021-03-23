import abc


class ABaseMIEstimator(abc.ABC):
    @abc.abstractmethod
    def __init__(self, critic, penalty=None):
        self._critic = critic
        self._penalty = penalty

    @abc.abstractmethod
    def __call__(self, x, y):
        pass
