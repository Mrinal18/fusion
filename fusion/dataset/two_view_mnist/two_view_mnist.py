import copy
from fusion.dataset.abasedataset import ABaseDataset
from fusion.dataset.two_view_mnist.transforms import TwoViewMnistTransform
from fusion.dataset.two_view_mnist.transforms import RandomRotationTransform
from fusion.dataset.two_view_mnist.transforms import UniformNoiseTransform
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader
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
            num_workers=0,
            seed=343,
    ):
        """

        :param dataset_dir:
        :param fold:
        :param num_folds:
        :param views:
        :param batch_size:
        :param shuffle:
        :param drop_last:
        :param num_workers:
        :param seed:
        """
        super(TwoViewMnist, self).__init__(
            dataset_dir,
            fold=fold,
            num_folds=num_folds,
            views=views,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            seed=seed,
        )
        self._num_classes = None

    def load(self):
        """

        :return:
        """
        for set_id in ['train', 'test']:
            train = True if set_id == 'train' else False
            transforms = self._prepare_transforms(set_id)
            dataset = torchvision.datasets.MNIST(
                self._dataset_dir,
                train=train,
                download=True,
                transform=transforms
            )
            if set_id == 'train':
                self._set_num_classes(dataset.targets)
                cv_datasets = self._prepare_fold(dataset)
                for set_id, dataset in cv_datasets.items():
                    self._set_dataloader(dataset, set_id)
            else:
                self._set_dataloader(dataset, set_id)

    def _set_dataloader(self, dataset, set_id):
        data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            drop_last=self._drop_last,
            num_workers=self._num_workers
        )
        set_id = 'infer' if set_id == 'test' else set_id
        self._data_loaders[set_id] = data_loader

    def _set_num_classes(self, targets):
        self.num_classes = len(torch.unique(targets))

    def _prepare_fold(self, dataset):
        kf = StratifiedKFold(
            n_splits=self._num_folds,
            shuffle=self._shuffle,
            random_state=self._seed
        )
        X, y = dataset.data, dataset.targets
        kf_g = kf.split(X, y)
        for _ in range(1, self._fold): next(kf_g)
        train_index, valid_index = next(kf.split(X, y))
        valid_dataset = copy.deepcopy(dataset)
        valid_dataset.data = dataset.data[valid_index]
        valid_dataset.targets = dataset.targets[valid_index]
        assert valid_dataset.data.size(0) == len(valid_index)
        assert valid_dataset.targets.size(0) == len(valid_index)
        train_dataset = copy.deepcopy(dataset)
        train_dataset.data = dataset.data[train_index]
        train_dataset.targets = dataset.targets[train_index]
        assert train_dataset.data.size(0) == len(train_index)
        assert train_dataset.targets.size(0) == len(train_index)
        return {
            'train': train_dataset,
            'valid': valid_dataset
        }

    def _prepare_transforms(self, set_id):
        if len(self._views) == 2:
            transforms = TwoViewMnistTransform()
        elif len(self._views) == 1:
            view = self._views[0]
            if view == 0:
                transforms = RandomRotationTransform()
            elif view == 1:
                transforms = UniformNoiseTransform()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return transforms

    def get_all_loaders(self):
        return super().get_all_loaders()

    def get_cv_loaders(self):
        return super().get_cv_loaders()

    def get_loader(self, set_id):
        return super().get_loader(set_id)

    def num_classes(self):
        return super().num_classes
