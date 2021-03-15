class Factory:
    def __init__(self):
        self._objects = {}

    def register_object(self, key, model):
        self._objects[key] = model

    def create(self, key, **kwargs):
        obj = self._objects.get(key)
        if not obj:
            raise ValueError(key)
        return obj(**kwargs)


class ObjectProvider(Factory):
    def get(self, idx, **kwargs):
        return self.create(idx, **kwargs)
