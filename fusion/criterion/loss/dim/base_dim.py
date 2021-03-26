import abc


class BaseDim(abc.ABC):
    _name = None

    def __init__(
        self,
        estimator,
        weight=1,
    ):
        self._estimator = estimator
        self._weight = weight

    @abc.abstractmethod
    def __call__(self, reps, convs):
        pass
