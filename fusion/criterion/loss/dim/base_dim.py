import abc
from typing import Optional

class BaseDim(abc.ABC):
    _name: Optional[str] = None

    def __init__(
        self,
        estimator,
        weight: float = 1.0,
    ):
        self._estimator = estimator
        self._weight = weight

    @abc.abstractmethod
    def __call__(self, reps, convs):
        pass
