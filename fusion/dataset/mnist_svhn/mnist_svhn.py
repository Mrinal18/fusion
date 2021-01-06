import copy
from fusion.dataset.abasedataset import ABaseDataset
from fusion.dataset.mnist_svhn.transforms import SVHNTransform, MNISTTransform
from sklearn.model_selection import StratifiedKFold
from torchnet.dataset import TensorDataset, ResampleDataset
import torch
from torch.utils.data import DataLoader
from collections import namedtuple
import torchvision
import tqdm
import os


def rand_match_on_idx(l1, idx1, l2, idx2, max_d=10000, dm=10):
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


def download_dataset(dataset_dir):
    max_d = 10000  # maximum number of datapoints per class
    dm = 30  # data multiplier: random permutations to match

    # get the individual datasets
    tx = torchvision.transforms.ToTensor()
    if os.path.exists(os.path.join(dataset_dir, "MNIST")):
        download = False
    else:
        download = True
    train_mnist = torchvision.datasets.MNIST(dataset_dir, train=True, download=download, transform=tx)
    test_mnist = torchvision.datasets.MNIST(dataset_dir, train=False, download=download, transform=tx)
    if os.path.exists(os.path.join(dataset_dir, "MNIST_SVHN")):
        download = False
    else:
        download = True
        os.mkdir(os.path.join(dataset_dir, "MNIST_SVHN"))
    train_svhn = torchvision.datasets.SVHN(os.path.join(dataset_dir, "MNIST_SVHN"), split="train", download=download,
                                           transform=tx)
    test_svhn = torchvision.datasets.SVHN(os.path.join(dataset_dir, "MNIST_SVHN"), split='test', download=download,
                                          transform=tx)
    # svhn labels need extra work
    train_svhn.labels = torch.LongTensor(train_svhn.labels.squeeze().astype(int)) % 10
    test_svhn.labels = torch.LongTensor(test_svhn.labels.squeeze().astype(int)) % 10

    mnist_l, mnist_li = train_mnist.targets.sort()
    svhn_l, svhn_li = train_svhn.labels.sort()
    idx1, idx2 = rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm)
    print('len train idx:', len(idx1), len(idx2))
    torch.save(idx1, os.path.join(dataset_dir, "MNIST_SVHN", 'train-ms-mnist-idx.pt'))
    torch.save(idx2, os.path.join(dataset_dir, "MNIST_SVHN", 'train-ms-svhn-idx.pt'))

    mnist_l, mnist_li = test_mnist.targets.sort()
    svhn_l, svhn_li = test_svhn.labels.sort()
    idx1, idx2 = rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm)
    print('len test idx:', len(idx1), len(idx2))
    torch.save(idx1, os.path.join(dataset_dir, "MNIST_SVHN", 'test-ms-mnist-idx.pt'))
    torch.save(idx2, os.path.join(dataset_dir, "MNIST_SVHN", 'test-ms-svhn-idx.pt'))


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
        download_dataset(self._dataset_dir)
        preloaded_mnist = {}
        preloaded_svhn = {}

        preloaded_mnist["train"] = torch.load(os.path.join(self._dataset_dir, "MNIST_SVHN",
                                                           'train-ms-mnist-idx.pt'))
        preloaded_svhn["train"] = torch.load(os.path.join(self._dataset_dir, "MNIST_SVHN",
                                                          'train-ms-svhn-idx.pt'))
        preloaded_mnist["test"] = torch.load(os.path.join(self._dataset_dir, "MNIST_SVHN",
                                                          'test-ms-mnist-idx.pt'))
        preloaded_svhn["test"] = torch.load(os.path.join(self._dataset_dir, "MNIST_SVHN",
                                                         'test-ms-svhn-idx.pt'))
        for set_id in ['train', 'test']:
            train = True if set_id == 'train' else False

            if os.path.exists(os.path.join(self._dataset_dir, "MNIST")):
                download = False
            else:
                download = True

            transforms_SVHN, transforms_MNIST = self._prepare_transforms(set_id)
            dataset_mnist = torchvision.datasets.MNIST(
                self._dataset_dir,
                train=train,
                download=download,
                transform=transforms_MNIST
            )
            dataset_svhn = torchvision.datasets.SVHN(
                os.path.join(self._dataset_dir, "MNIST_SVHN"),
                split=set_id,
                download=download,
                transform=transforms_SVHN
            )
            if len(self._views) == 2:
                dataset = TensorDataset([
                    ResampleDataset(
                        dataset_mnist, lambda d, i: preloaded_mnist[set_id][i],
                        size=len(preloaded_mnist[set_id])
                    ),
                    ResampleDataset(
                        dataset_svhn, lambda d, i: preloaded_svhn[set_id][i],
                        size=len(preloaded_svhn[set_id])
                    )
                ])
                data, targets = torch.ones([len(dataset), 4, 32, 32]), torch.ones([len(dataset), 2])
            else:
                if self._views[0] == 0:
                    dataset = TensorDataset([
                        ResampleDataset(
                            dataset_mnist, lambda d, i: preloaded_mnist[set_id][i],
                            size=len(preloaded_mnist[set_id])
                        ),
                    ])
                elif self._views[0] == 1:
                    dataset = TensorDataset([
                        ResampleDataset(
                            dataset_svhn, lambda d, i: preloaded_svhn[set_id][i],
                            size=len(preloaded_svhn[set_id])
                        ),
                    ])
                data, targets = torch.ones([len(dataset), 1, 32, 32]), torch.ones([len(dataset), 1])

            Dataset_Special = namedtuple("Dataset_Special", ["data", "targets"])
            k = 0
            for x in tqdm.tqdm(dataset):
                if len(self._views) == 2:
                    data[k][0],  data[k][1:] = x[0][0], x[1][0]

                    targets[k] = torch.tensor([x[0][1], x[1][1]])
                    k += 1
                else:
                    data[k] = x[0][0]
                    targets[k] = torch.tensor(x[0][1])
                    k += 1

            dataset = Dataset_Special(data=data, targets=targets)

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
        if len(self._views) == 2:
            y_split = y[:, 0]
        else:
            y_split = y

        kf_g = kf.split(X, y_split)
        Dataset_Special = namedtuple("Dataset_Special", ["data", "targets"])
        for _ in range(1, self._fold): next(kf_g)
        train_index, valid_index = next(kf.split(X, y_split))
        valid_dataset = Dataset_Special(data=dataset.data[valid_index],
                                        targets=dataset.targets[valid_index])
        assert valid_dataset.data.size(0) == len(valid_index)
        assert valid_dataset.targets.size(0) == len(valid_index)
        train_dataset = Dataset_Special(data=dataset.data[train_index],
                                        targets=dataset.targets[train_index])
        assert train_dataset.data.size(0) == len(train_index)
        assert train_dataset.targets.size(0) == len(train_index)
        return {
            'train': train_dataset,
            'valid': valid_dataset
        }

    def _prepare_transforms(self, set_id):
        transforms_SVHN = SVHNTransform()
        transforms_MNIST = MNISTTransform()

        return transforms_SVHN, transforms_MNIST

    def get_all_loaders(self):
        return super().get_all_loaders()

    def get_cv_loaders(self):
        return super().get_cv_loaders()

    def get_loader(self, set_id):
        return super().get_loader(set_id)

    def num_classes(self):
        return super().num_classes
