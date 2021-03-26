import abc


class ABasePenalty(abc.ABC):
    def __call__(self, scores):
        pass
