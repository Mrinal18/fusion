import os
import copy

from catalyst.data.loader import BatchPrefetchLoaderWrapper

from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision

from fusion.dataset.abasedataset import ABaseDataset, SetId
from fusion.dataset.abasetransform import ABaseTransform
from fusion.dataset.two_view_mnist.transforms import TwoViewMnistTransform
from fusion.dataset.two_view_mnist.transforms import RandomRotationTransform
from fusion.dataset.two_view_mnist.transforms import UniformNoiseTransform
from fusion.dataset.utils import seed_worker


class TwoViewMnist(ABaseDataset):
    def load(self):
        """

        :return:
        """
        for set_id in [SetId.TRAIN, SetId.TEST]:
            train = True if set_id == SetId.TRAIN else False
            transforms = self._prepare_transforms(set_id)
            download = True
            if os.path.exists(self._dataset_dir):
                download = False
            dataset = torchvision.datasets.MNIST(
                self._dataset_dir, train=train, download=download, transform=transforms
            )
            if set_id == SetId.TRAIN:
                self._set_num_classes(dataset.targets)
                cv_datasets = self._prepare_fold(dataset)
                for set_id, dataset in cv_datasets.items():
                    self._set_dataloader(dataset, set_id)
            else:
                self._set_dataloader(dataset, set_id)

    def _set_dataloader(self, dataset: Dataset, set_id: SetId):
        drop_last = True if set_id == SetId.TRAIN else self._drop_last
        shuffle = True if set_id == SetId.TRAIN else self._shuffle
        data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self._num_workers,
            worker_init_fn=seed_worker,
            prefetch_factor=self._prefetch_factor,
            persistent_workers=self._persistent_workers,
            pin_memory=self._pin_memory,
        )
        set_id = SetId.INFER if set_id == SetId.TEST else set_id
        if torch.cuda.is_available() and self._num_prefetches is not None:
            data_loader = BatchPrefetchLoaderWrapper(
                data_loader, num_prefetches=self._num_prefetches
            )
        self._data_loaders[set_id] = data_loader

    def _set_num_classes(self, targets):
        self._num_classes = len(torch.unique(targets))

    def _prepare_fold(self, dataset: torchvision.datasets.MNIST):
        kf = StratifiedKFold(
            n_splits=self._num_folds, shuffle=True, random_state=self._seed
        )
        X, y = dataset.data, dataset.targets
        kf_g = kf.split(X, y)
        for _ in range(1, self._fold):
            next(kf_g)
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
        return {SetId.TRAIN: train_dataset, SetId.VALID: valid_dataset}

    def _prepare_transforms(self, set_id: SetId) -> ABaseTransform:
        transforms: ABaseTransform
        if len(self._sources) == 2:
            transforms = TwoViewMnistTransform()
        elif len(self._sources) == 1:
            source = self._sources[0]
            if source == 0:
                transforms = RandomRotationTransform()
            elif source == 1:
                transforms = UniformNoiseTransform()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return transforms
