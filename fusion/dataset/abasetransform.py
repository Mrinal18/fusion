import abs
from typing import Tuple, Union

from torch import Tensor


class ABaseTransform(abs.ABC):
    @abs.abstractmethod
    def __call__(self, x) -> Union[Tensor, Tuple[Tensor, ...]]:
        pass
