import abc

class ABaseRunner(abc.ABC):
    def _unpack_batch(self, batch):
        """

        :param batch:
        :return:
        """
        x, y = batch
        return x, y
