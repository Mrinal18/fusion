from fusion.dataset.ibasedataset import IBaseDataset
import zope.interface


@zope.interface.implementer(IBaseDataset)
class ABaseDataset:
    @zope.interface.interfacemethod
    def __init__(
        self, 
        dataset_dir,
        fold=0,
        num_folds=5
        views=[0],
        batch_size=2, 
        shuffle=False, 
        drop_last=False,
        num_workers=0
    ):
        self._dataset_dir = dataset_dir
        self._fold = fold
        self._num_folds = num_folds
        self._views = views
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._num_workers = num_workers

    @zope.interface.interfacemethod
    def load(self):
        """Loads the dataset
        """
        pass

    @zope.interface.interfacemethod
    def get_all_loaders(self):
        """Returns dictionary with data loaders
        """
        return self._data_loaders

    @zope.interface.interfacemethod
    def get_cv_loaders(self):
        """Returns dictionary with cross-validation loaders
        """
        return {set_id: self._data_loaders[set_id] for set_id in ['train', 'val']}

    @zope.interface.interfacemethod
    def get_loader(set_id):
        """Returns loader with specific set

        Args:
            set_id (str): "\'train\', \'val\', \'test\'"
        """
        return self._data_loaders[set_id]

    @zope.interface.interfacemethod
    def num_classes(self):
        """Number of classes
        """
        return self._num_classes

    @zope.interface.interfacemethod
    def _prepare_transforms(set_id):
        """Creates set of data transforms for specific set of data.

        Args:
            set_id (str): "\'train\', \'val', \'test\'"
        """
        pass
