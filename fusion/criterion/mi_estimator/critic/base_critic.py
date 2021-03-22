import abc


class ABaseCritic(abc.ABC):
    def __init__(self, **critic_attrs):
        self._critic_attrs = critic_attrs

    def score(self, x, y):
        pass
