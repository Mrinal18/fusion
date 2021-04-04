import abc
from enum import Enum
from typing import Dict, List, Optional

from tensor.utils.data import DataLoader


class SetId(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'
    INFER = 'infer'


class ABaseDataset(abc.ABC):
    _num_classes: Optional[int] = None
    _data_loaders: Dict[SetId, DataLoader] = {}

    @abc.abstractmethod
    def __init__(
        self,
        dataset_dir: str,
        fold: int = 0,
        num_folds: int = 5,
        sources: List[int] = [0],
        batch_size: int = 2,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
        seed: int = 343,
    ):
        self._dataset_dir = dataset_dir
        self._fold = fold
        self._num_folds = num_folds
        self._sources = sources
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._num_workers = num_workers
        self._seed = seed

    @abc.abstractmethod
    def load(self):
        """Loads the dataset
        """
        pass

    @abc.abstractmethod
    def get_all_loaders(self) -> Dict[SetId, DataLoader]:
        """Returns dictionary with data loaders
        """
        return self._data_loaders

    @abc.abstractmethod
    def get_cv_loaders(self) -> Dict[SetId, DataLoader]:
        """Returns dictionary with cross-validation loaders
        """
        return {set_id: self._data_loaders[set_id] for set_id in [SetId.TRAIN, SetId.VALID]}

    @abc.abstractmethod
    def get_loader(self, set_id: SetId) -> DataLoader:
        """Returns loader with specific set

        Args:
            set_id (SetID): "\'TRAIN\', \'VALID\', \'TEST\'"
        """
        return self._data_loaders[set_id]

    @property
    @abc.abstractmethod
    def num_classes(self) -> Optional[int]:
        """Number of classes
        """
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value: int):
        """Number of classes
        """
        self._num_classes = value
