import abc


class ABaseClip(abc.ABC):
    def __init__(self, clip_value=10.):
        self._clip_value = clip_value

    def __call__(self, scores):
        pass

    @property
    def clip_value(self):
        return self._clip_value

    @clip_value.setter
    def clip_value(self, value):
        self._clip_value = value
