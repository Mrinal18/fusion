import abc
import torch


class ABaseDataset(abc.ABC):
    _num_classes = -1
    _data_loaders = {}

    @abc.abstractmethod
    def __init__(
        self,
        dataset_dir,
        fold=0,
        num_folds=5,
        sources=[0],
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        seed=343,
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
        torch.manual_seed(self._seed)

    @abc.abstractmethod
    def load(self):
        """Loads the dataset
        """
        pass

    @abc.abstractmethod
    def get_all_loaders(self):
        """Returns dictionary with data loaders
        """
        return self._data_loaders

    @abc.abstractmethod
    def get_cv_loaders(self):
        """Returns dictionary with cross-validation loaders
        """
        return {set_id: self._data_loaders[set_id] for set_id in ['train', 'valid']}

    @abc.abstractmethod
    def get_loader(self, set_id):
        """Returns loader with specific set

        Args:
            set_id (str): "\'train\', \'valid\', \'test\'"
        """
        return self._data_loaders[set_id]

    @property
    @abc.abstractmethod
    def num_classes(self):
        """Number of classes
        """
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        """Number of classes
        """
        self._num_classes = value
