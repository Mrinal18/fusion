from fusion.dataset.abasedataset import ABaseDataset
from fusion.dataset.mnist_svhn.transforms import SVHNTransform, MNISTTransform
from sklearn.model_selection import StratifiedKFold
from torchnet.dataset import TensorDataset, ResampleDataset
import torch
from torch.utils.data import DataLoader
import torchvision
import os


class MnistSvhn(ABaseDataset):
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
        super(MnistSvhn, self).__init__(
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
        self._views = views

    def load(self):
        self._download_dataset(self._dataset_dir)

        for set_id in ['train', 'valid', 'test']:
            dataset = None
            if len(self._views) == 2:
                dataset_mnist, indexes_mnist = self._load_mnist(set_id)
                dataset_svhn, indexes_svhn = self._load_svhn(set_id)
                if set_id == 'train':
                    self._set_num_classes(dataset_mnist.dataset.targets)
                else:
                    assert (len(torch.unique(dataset_mnist.dataset.targets)) == self.num_classes)
                dataset = TensorDataset([
                    ResampleDataset(
                        dataset_mnist.dataset,
                        lambda d, i: indexes_mnist[i],
                        size=len(indexes_mnist)
                    ),
                    ResampleDataset(
                        dataset_svhn.dataset,
                        lambda d, i: indexes_svhn[i],
                        size=len(indexes_svhn)
                    )
                ])

            else:
                if self._views[0] == 0:
                    dataset_mnist, indexes_mnist = self._load_mnist(set_id)
                    dataset = TensorDataset([
                        ResampleDataset(
                            dataset_mnist.dataset,
                            lambda d, i: indexes_mnist[i],
                            size=len(indexes_mnist)
                        ),
                    ])
                elif self._views[0] == 1:
                    dataset_svhn, indexes_svhn = self._load_svhn(set_id)
                    dataset = TensorDataset([
                        ResampleDataset(
                            dataset_svhn.dataset,
                            lambda d, i: indexes_svhn[i],
                            size=len(indexes_svhn)
                        )
                    ])
            self._set_dataloader(dataset, set_id)

    def _load_mnist(self, set_id):
        dataset_dir = os.path.join(self._dataset_dir, "MNIST_SVHN")
        if set_id != 'test':
            filename = f"{set_id}-ms-mnist-idx-{self._fold}.pt"
        else:
            filename = f"{set_id}-ms-mnist-idx.pt"
        print('mnist', set_id, filename)
        indexes = torch.load(os.path.join(dataset_dir, filename))
        train = True if set_id != 'test' else False
        tx_mnist = MNISTTransform()
        dataset = torchvision.datasets.MNIST(
            dataset_dir, train=train, download=True, transform=tx_mnist)
        if set_id != 'test':
            cv_indexes = torch.load(
                os.path.join(
                    dataset_dir,
                    f"{set_id}-ms-mnist-cv-idx-{self._fold}.pt"
                )
            )
            dataset.data = dataset.data[cv_indexes]
            dataset.targets = dataset.targets[cv_indexes]
        print('mnist', set_id, len(dataset.data), len(dataset.targets), len(indexes))
        dataset = DataLoader(
            dataset, batch_size=self._batch_size, shuffle=self._shuffle,
            pin_memory=False, num_workers=1
        )
        return dataset, indexes

    def _load_svhn(self, set_id):
        dataset_dir = os.path.join(self._dataset_dir, "MNIST_SVHN")
        if set_id != 'test':
            filename = f"{set_id}-ms-svhn-idx-{self._fold}.pt"
        else:
            filename = f"{set_id}-ms-svhn-idx.pt"
        print ('svhn', set_id, filename)
        indexes = torch.load(os.path.join(dataset_dir, filename))
        split = 'train' if set_id != 'test' else 'test'
        tx_svhn = SVHNTransform()
        dataset = torchvision.datasets.SVHN(
            dataset_dir, split=split, download=True, transform=tx_svhn)
        if set_id != 'test':
            cv_indexes = torch.load(
                os.path.join(
                    dataset_dir,
                    f"{set_id}-ms-svhn-cv-idx-{self._fold}.pt"
                )
            )
            dataset.data = dataset.data[cv_indexes]
            dataset.labels = dataset.labels[cv_indexes]
        print('svhn', set_id, len(dataset.data), len(dataset.labels), len(indexes))
        dataset = DataLoader(
            dataset, batch_size=self._batch_size, shuffle=self._shuffle,
            pin_memory=False, num_workers=1
        )
        return dataset, indexes

    def _set_dataloader(self, dataset, set_id):
        data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            drop_last=self._drop_last,
            num_workers=self._num_workers,
            pin_memory=False
        )
        set_id = 'infer' if set_id == 'test' else set_id
        self._data_loaders[set_id] = data_loader

    def _set_num_classes(self, targets):
        self.num_classes = len(torch.unique(targets))

    def _prepare_fold(self, dataset, dataset_name):
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
        return super().get_all_loaders()

    def get_cv_loaders(self):
        return super().get_cv_loaders()

    def get_loader(self, set_id):
        return super().get_loader(set_id)

    def num_classes(self):
        return super().num_classes

    @staticmethod
    def _rand_match_on_idx(l1, idx1, l2, idx2, max_d=10000, dm=10):
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

    def _download_dataset(self, dataset_dir):
        max_d = 10000  # maximum number of datapoints per class
        dm = 30  # data multiplier: random permutations to match

        # get the individual datasets
        tx = torchvision.transforms.ToTensor()
        if os.path.exists(os.path.join(dataset_dir, "MNIST_SVHN")):
            download = False
        else:
            download = True
            os.mkdir(os.path.join(dataset_dir, "MNIST_SVHN"))
        # load mnist
        train_mnist = torchvision.datasets.MNIST(
            dataset_dir, train=True, download=download, transform=tx)
        test_mnist = torchvision.datasets.MNIST(
            dataset_dir, train=False, download=download, transform=tx)
        # load svhn
        train_svhn = torchvision.datasets.SVHN(
            os.path.join(self._dataset_dir, "MNIST_SVHN"),
            split="train", download=download, transform=tx)
        test_svhn = torchvision.datasets.SVHN(
            os.path.join(self._dataset_dir, "MNIST_SVHN"),
            split='test', download=download, transform=tx)
        # svhn labels need extra work
        train_svhn.labels = torch.LongTensor(train_svhn.labels.squeeze().astype(int)) % 10
        test_svhn.labels = torch.LongTensor(test_svhn.labels.squeeze().astype(int)) % 10
        # split on cross-validation folds
        mnist_train_idxs, mnist_valid_idxs = self._prepare_fold(train_mnist, 'MNIST')
        svhn_train_idxs, svhn_valid_idxs = self._prepare_fold(train_svhn, 'SVHN')
        # save and pair training set
        mnist_l, mnist_li = train_mnist.targets[mnist_train_idxs].sort()
        svhn_l, svhn_li = train_svhn.labels[svhn_train_idxs].sort()
        idx1, idx2 = self._rand_match_on_idx(
            mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm)
        print('len train idx:', len(idx1), len(idx2))
        torch.save(idx1, os.path.join(
            dataset_dir, "MNIST_SVHN", f"train-ms-mnist-idx-{self._fold}.pt"))
        torch.save(idx2, os.path.join(
            dataset_dir, "MNIST_SVHN", f"train-ms-svhn-idx-{self._fold}.pt"))
        torch.save(mnist_train_idxs,
                   os.path.join(
                       dataset_dir, "MNIST_SVHN", f'train-ms-mnist-cv-idx-{self._fold}.pt'))
        torch.save(svhn_train_idxs,
                   os.path.join(
                       dataset_dir, "MNIST_SVHN", f'train-ms-svhn-cv-idx-{self._fold}.pt'))
        # save and pair validation set
        mnist_l, mnist_li = train_mnist.targets[mnist_valid_idxs].sort()
        svhn_l, svhn_li = train_svhn.labels[svhn_valid_idxs].sort()
        idx1, idx2 = self._rand_match_on_idx(
            mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm)
        print('len train idx:', len(idx1), len(idx2))
        torch.save(idx1, os.path.join(
            dataset_dir, "MNIST_SVHN", f'valid-ms-mnist-idx-{self._fold}.pt'))
        torch.save(idx2, os.path.join(
            dataset_dir, "MNIST_SVHN", f'valid-ms-svhn-idx-{self._fold}.pt'))
        torch.save(mnist_valid_idxs,
                   os.path.join(
                       dataset_dir, "MNIST_SVHN", f'valid-ms-mnist-cv-idx-{self._fold}.pt'))
        torch.save(svhn_valid_idxs,
                   os.path.join(
                       dataset_dir, "MNIST_SVHN", f'valid-ms-svhn-cv-idx-{self._fold}.pt'))
        # save and pair test set
        mnist_l, mnist_li = test_mnist.targets.sort()
        svhn_l, svhn_li = test_svhn.labels.sort()
        idx1, idx2 = self._rand_match_on_idx(
            mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm)
        print('len test idx:', len(idx1), len(idx2))
        torch.save(idx1, os.path.join(dataset_dir, "MNIST_SVHN", 'test-ms-mnist-idx.pt'))
        torch.save(idx2, os.path.join(dataset_dir, "MNIST_SVHN", 'test-ms-svhn-idx.pt'))
