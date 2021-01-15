import abc

class ABaseRunner(abc.ABC):
    def _unpack_batch(self, batch):
        x, y = batch
        return x, y
