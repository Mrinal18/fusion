from fusion.dataset.abasedataset import ABaseDataset
from fusion.dataset.two_view_mnist.transforms import TwoViewMnistTransform
from fusion.dataset.two_view_mnist.transforms import RandomRotationTransform
from fusion.dataset.two_view_mnist.transforms import UniformNoiseTransform
from torch.utils.data import DataLoader, data
import torchvision


class TwoViewMnist(ABaseDataset):
    def __init__(
        self,
        dataset_dir,
        fold=0,
        num_folds=5,
        views=[0],
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=0
    ):
        super(TwoViewMnist, self).__init__(
            dataset_dir,
            fold=fold,
            num_folds=num_folds,
            views=views,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

    def load(self):
        for set_id in ['train', 'test']:
            train = True if set_id == 'train' else 'test'
            transforms = self._prepare_transforms(set_id)
            dataset = torchvision.datasets.MNIST(
                self._dataset_dir,
                train=train,
                download=True,
                transform=transforms
            )
            data_loader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                shuffle=self._shuffle,
                drop_last=self._drop_last,
                num_workers=self._num_workers
            )
            self._data_loaders[set_id] = data_loader

    def _prepare_transforms(self, set_id):
        del set_id
        if len(self._views) == 2:
            transforms = TwoViewMnistTransform()
        elif len(self._views) == 1:
            if self._views[0] == 0:
                transforms = RandomRotationTransform()
            elif self._views[1] == 1:
                transforms = UniformNoiseTransform()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return transforms
