import abc


class ABaseMIEstimator(abc.ABC):
    def __init__(self, critic, clip=None, penalty=None):
        self._critic = critic
        self._penalty = penalty
        self._clip = clip

    @abc.abstractmethod
    def __call__(self, x, y):
        pass
