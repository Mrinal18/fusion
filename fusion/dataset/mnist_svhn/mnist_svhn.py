import copy
import os
from typing import Any, Dict, List, Union

from sklearn.model_selection import StratifiedKFold
import torch
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchnet.dataset import TensorDataset, ResampleDataset

from fusion.dataset.utils import seed_worker
from fusion.dataset.abasedataset import ABaseDataset, SetId
from fusion.dataset.mnist_svhn.transforms import SVHNTransform, MNISTTransform


class MnistSvhn(ABaseDataset):
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
        """
        Initialization of Class MnistSvhn dataset
        Args:
            :param dataset_dir: path to dataset
            :param fold: number of fold for validation
            :param num_folds: counts of folds
            :param views: number of views
            :param batch_size: how many samples per batch to load
            :param shuffle: set to True to have the data reshuffled at every epoch
            :param drop_last: set to True to drop the last incomplete batch
            :param num_workers: how many subprocesses to use for data loading
            :param seed: number of seed
        Return:
            Dataset MnistSvhn

        """
        super().__init__(
            dataset_dir,
            fold=fold,
            num_folds=num_folds,
            sources=sources,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            seed=seed,
        )
        self._sources = sources
        self._indexes: Dict[str, Dict[str, Any]] = {}

    def load(self):
        """
        Method to load dataset
        """
        if os.path.exists(self._dataset_dir):
            self._download_dataset(self._dataset_dir)
        self._num_classes = 10
        # Don't touch it, otherwise lazy evaluation and lambda functions will make you cry
        samplers = {
            'mnist': {
                SetId.TRAIN: lambda d, i: self._indexes[SetId.TRAIN]['mnist'][i],
                SetId.VALID: lambda d, i: self._indexes[SetId.VALID]['mnist'][i],
                SetId.TEST: lambda d, i: self._indexes[SetId.TEST]['mnist'][i],
            },
            'svhn': {
                SetId.TRAIN: lambda d, i: self._indexes[SetId.TRAIN]['svhn'][i],
                SetId.VALID: lambda d, i: self._indexes[SetId.VALID]['svhn'][i],
                SetId.TEST: lambda d, i: self._indexes[SetId.TEST]['svhn'][i],
            }
        }

        for set_id in [SetId.TRAIN, SetId.VALID, SetId.TEST]:
            dataset = None
            sampler_mnist = samplers['mnist'][set_id]
            sampler_svhn = samplers['svhn'][set_id]
            if len(self._sources) == 2:
                dataset_mnist, indexes_mnist = self._load(set_id, 'mnist')
                dataset_svhn, indexes_svhn = self._load(set_id, 'svhn')
                self._indexes[set_id] = {}
                self._indexes[set_id]['mnist'] = indexes_mnist
                self._indexes[set_id]['svhn'] = indexes_svhn
                dataset = TensorDataset([
                    ResampleDataset(
                        dataset_mnist.dataset,
                        sampler_mnist,
                        size=len(self._indexes[set_id]['mnist'])
                    ),
                    ResampleDataset(
                        dataset_svhn.dataset,
                        sampler_svhn,
                        size=len(self._indexes[set_id]['svhn'])
                    )
                ])
                # collate_fn or tensor dataset with transforms
            else:
                if self._sources[0] == 0:
                    dataset_mnist, indexes_mnist = self._load(set_id, 'mnist')
                    self._indexes[set_id] = {}
                    self._indexes[set_id]['mnist'] = indexes_mnist
                    dataset = TensorDataset([
                        ResampleDataset(
                            dataset_mnist.dataset,
                            sampler_mnist,
                            size=len(indexes_mnist)
                        ),
                    ])
                elif self._sources[0] == 1:
                    self._indexes[set_id] = {}
                    dataset_svhn, indexes_svhn = self._load(set_id, 'svhn')
                    self._indexes[set_id]['svhn'] = indexes_svhn
                    dataset = TensorDataset([
                        ResampleDataset(
                            dataset_svhn.dataset,
                            sampler_svhn,
                            size=len(indexes_svhn)
                        )
                    ])
            self._set_dataloader(dataset, set_id)


    def _load(self, set_id: SetId, dataset_name: str):
        # define filename for pair indexes
        if set_id != SetId.TEST:
            filename = f"{set_id.lower()}-ms-{dataset_name}-idx-{self._fold}.pt"
        else:
            filename = f"{set_id.lower()}-ms-{dataset_name}-idx.pt"
        # load paired indexes
        indexes = torch.load(os.path.join(self._dataset_dir, filename))
        # load dataset
        if dataset_name == 'mnist':
            # validation uses training set
            train = True if set_id != SetId.TEST else False
            tx = MNISTTransform()
            dataset = torchvision.datasets.MNIST(
                self._dataset_dir, train=train, download=False, transform=tx)
        elif dataset_name == 'svhn':
            # validation uses training set
            split = SetId.TRAIN if set_id != SetId.TEST else SetId.TEST
            tx = SVHNTransform()
            dataset = torchvision.datasets.SVHN(
                self._dataset_dir, split=split, download=False, transform=tx)
        else:
            raise NotImplementedError
        # select fold
        if set_id != SetId.TEST:
            cv_indexes = torch.load(
                os.path.join(
                    self._dataset_dir,
                    f"{set_id.lower()}-ms-{dataset_name}-cv-idx-{self._fold}.pt"
                )
            )
            dataset.data = dataset.data[cv_indexes]
            if dataset_name == 'mnist':
                dataset.targets = dataset.targets[cv_indexes]
            elif dataset_name == 'svhn':
                dataset.labels = dataset.labels[cv_indexes]
            else:
                raise NotImplementedError
        dataset = DataLoader(
            dataset, batch_size=1, shuffle=False,
            pin_memory=True, num_workers=1,

        )
        return dataset, indexes

    def _set_dataloader(self, dataset: Dataset, set_id: SetId):
        data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            drop_last=self._drop_last,
            num_workers=self._num_workers,
            worker_init_fn=seed_worker,
            prefetch_factor=self._batch_size,
            persistent_workers=True,
            pin_memory=True
        )
        set_id = SetId.INFER if set_id == SetId.TEST else set_id
        self._data_loaders[set_id] = data_loader

    def _set_num_classes(self, targets: Tensor):
        self._num_classes = len(torch.unique(targets))

    def _prepare_fold(
            self, dataset: Union[torchvision.datasets.MNIST, torchvision.datasets.SVHN],
            dataset_name: str
    ):
        kf = StratifiedKFold(
            n_splits=self._num_folds,
            shuffle=self._shuffle,
            random_state=self._seed
        )
        if dataset_name == 'MNIST':
            X, y = dataset.data, dataset.targets
        else:
            X, y = dataset.data, dataset.labels
        kf_g = kf.split(X, y)
        for _ in range(1, self._fold): next(kf_g)
        train_index, valid_index = next(kf.split(X, y))
        return train_index, valid_index

    def get_all_loaders(self):
        """
        Return all loaders
        """
        return super().get_all_loaders()

    def get_cv_loaders(self):
        """
        Return all cross-validation loaders
        """
        return super().get_cv_loaders()

    def get_loader(self, set_id: SetId):
        """
        Get loader by set_id
        Args:
            :param set_id:
        Return:
            Loader by set_id
        """
        return super().get_loader(set_id)

    def num_classes(self):
        return super().num_classes

    @staticmethod
    def _rand_match_on_idx(l1, idx1, l2, idx2, max_d: int = 10000, dm: int = 10):
        """
        l*: sorted labels
        idx*: indices of sorted labels in original list
        """
        _idx1, _idx2 = [], []
        for l in l1.unique():  # assuming both have same idxs
            l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
            n = min(l_idx1.size(0), l_idx2.size(0), max_d)
            l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
            for _ in range(dm):
                _idx1.append(l_idx1[torch.randperm(n)])
                _idx2.append(l_idx2[torch.randperm(n)])
        return torch.cat(_idx1), torch.cat(_idx2)

    def _download_dataset(self, dataset_dir: str):
        max_d = 10000  # maximum number of datapoints per class
        dm = 30  # data multiplier: random permutations to match

        # get the individual datasets
        tx = torchvision.transforms.ToTensor()
        if os.path.exists(self._dataset_dir):
            download = False
        else:
            download = True
            os.mkdir(self._dataset_dir)
        # load mnist
        train_mnist = torchvision.datasets.MNIST(
            dataset_dir, train=True, download=download, transform=tx)
        test_mnist = torchvision.datasets.MNIST(
            dataset_dir, train=False, download=download, transform=tx)

        # load svhn
        train_svhn = torchvision.datasets.SVHN(
            self._dataset_dir,
            split="train", download=download, transform=tx)
        test_svhn = torchvision.datasets.SVHN(
            self._dataset_dir,
            split=SetId.TEST, download=download, transform=tx)

        # svhn labels need extra work
        train_svhn.labels = torch.LongTensor(
            train_svhn.labels.squeeze().astype(int)) % 10
        test_svhn.labels = torch.LongTensor(
            test_svhn.labels.squeeze().astype(int)) % 10

        # split on cross-validation folds
        mnist_train_idxs, mnist_valid_idxs = self._prepare_fold(train_mnist, 'MNIST')
        svhn_train_idxs, svhn_valid_idxs = self._prepare_fold(train_svhn, 'SVHN')

        # save and pair training set
        mnist_l, mnist_li = train_mnist.targets[mnist_train_idxs].sort()
        svhn_l, svhn_li = train_svhn.labels[svhn_train_idxs].sort()
        idx1, idx2 = self._rand_match_on_idx(
            mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm)
        torch.save(idx1, os.path.join(
            dataset_dir, f"train-ms-mnist-idx-{self._fold}.pt"))
        torch.save(idx2, os.path.join(
            dataset_dir, f"train-ms-svhn-idx-{self._fold}.pt"))
        torch.save(mnist_train_idxs,
                   os.path.join(
                       dataset_dir, f'train-ms-mnist-cv-idx-{self._fold}.pt'))
        torch.save(svhn_train_idxs,
                   os.path.join(
                       dataset_dir, f'train-ms-svhn-cv-idx-{self._fold}.pt'))

        # save and pair validation set
        mnist_l, mnist_li = train_mnist.targets[mnist_valid_idxs].sort()
        svhn_l, svhn_li = train_svhn.labels[svhn_valid_idxs].sort()
        idx1, idx2 = self._rand_match_on_idx(
            mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm)
        torch.save(idx1, os.path.join(
            dataset_dir, f'valid-ms-mnist-idx-{self._fold}.pt'))
        torch.save(idx2, os.path.join(
            dataset_dir, f'valid-ms-svhn-idx-{self._fold}.pt'))
        torch.save(mnist_valid_idxs,
                   os.path.join(
                       dataset_dir, f'valid-ms-mnist-cv-idx-{self._fold}.pt'))
        torch.save(svhn_valid_idxs,
                   os.path.join(
                       dataset_dir,f'valid-ms-svhn-cv-idx-{self._fold}.pt'))

        # save and pair test set
        mnist_l, mnist_li = test_mnist.targets.sort()
        svhn_l, svhn_li = test_svhn.labels.sort()
        idx1, idx2 = self._rand_match_on_idx(
            mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm)
        torch.save(idx1, os.path.join(dataset_dir, 'test-ms-mnist-idx.pt'))
        torch.save(idx2, os.path.join(dataset_dir, 'test-ms-svhn-idx.pt'))
