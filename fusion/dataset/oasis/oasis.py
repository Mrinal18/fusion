import logging
import os
import pandas as pd
from typing import List, Optional

from catalyst.data.sampler import BalanceClassSampler
from catalyst.data.loader import BatchPrefetchLoaderWrapper
from torch.utils.data.dataloader import DataLoader

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import torchio as tio
from torchio import Subject, ScalarImage, SubjectsDataset

from fusion.dataset.abasedataset import ABaseDataset, SetId
from fusion.dataset.utils import seed_worker
from fusion.task import TaskId

from .transforms import MNIMaskTransform
from .transforms import VolumetricRandomCrop


class Oasis(ABaseDataset):
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
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        num_prefetches: Optional[int] = None,
        is_only_one_pair_per_subject: bool = False,
        use_balanced_sampler: bool = False,
        use_separate_augmentation: bool = False,
        only_labeled: bool = False,
        task_id: TaskId = TaskId.PRETRAINING
    ):
        """
        Initialization of Class Oasis dataset
        Args:
            dataset_dir: path to dataset
            fold: number of fold for validation
            num_folds: counts of folds
            source_ids: number of source_ids
            batch_size: how many samples per batch to load
            shuffle: set to True to have the data reshuffled at every epoch
            drop_last: set to True to drop the last incomplete batch
            num_workers: how many subprocesses to use for data loading
            seed: number of seed
            is_only_one_pair_per_subject: set to True to use only one pair (given pandas algorithm uses first entry in the dataframe) of images per subjects
            use_balanced_sampler: set to True to use balanced data sampler
            use_separate_augmentation: set to True to have different augmentation for modalities
            only_labeled: set to True to remove unlabeled ("-1") samples
            task_id: the task id, needed for transforms
        Return:
            Dataset Oasis

        """
        super().__init__(
            dataset_dir + f'/{fold}/',
            fold=fold,
            num_folds=num_folds,
            sources=sources,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            seed=seed,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            num_prefetches=num_prefetches,
        )
        assert shuffle != use_balanced_sampler, "Sampler and Shuffle do not go together for dataloader in PyTorch"
        self._is_only_one_pair_per_subject = is_only_one_pair_per_subject
        self._use_balanced_sampler = use_balanced_sampler
        self._use_separate_augmentation = use_separate_augmentation
        self._only_labeled = only_labeled
        self._task_id = task_id

    def load(self):
        for set_id in [SetId.TRAIN, SetId.VALID, SetId.INFER]:
            list_of_subjects, labels = self._prepare_subject_list(set_id)
            transforms = self._prepare_transforms(
                set_id, self._use_separate_augmentation)
            dataset = SubjectsDataset(
                list_of_subjects,
                transform=transforms
            )
            self._set_dataloader(dataset, set_id, labels)

    def _set_dataloader(
        self,
        dataset: SubjectsDataset,
        set_id: SetId,
        labels: List[int]
    ):
        drop_last = self._drop_last
        shuffle = self._shuffle
        if set_id == SetId.TRAIN:
            sampler = BalanceClassSampler(
                labels, mode='upsampling',
            ) if self._use_balanced_sampler else None
            drop_last = True
            shuffle = True
            self._set_num_classes(labels)
        # as sampler and shuffle do not go together
        if sampler is not None:
            data_loader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                drop_last=drop_last,
                num_workers=self._num_workers,
                worker_init_fn=seed_worker,
                prefetch_factor=self._prefetch_factor,
                persistent_workers=self._persistent_workers,
                pin_memory=self._pin_memory
            )
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                drop_last=drop_last,
                shuffle=shuffle,
                num_workers=self._num_workers,
                worker_init_fn=seed_worker,
                prefetch_factor=self._prefetch_factor,
                persistent_workers=self._persistent_workers,
                pin_memory=self._pin_memory
            )
        if torch.cuda.is_available() and self._num_prefetches is not None:
            data_loader = BatchPrefetchLoaderWrapper(
                data_loader, num_prefetches=self._num_prefetches
            )
        self._data_loaders[set_id] = data_loader

    def _prepare_subject_list(self, set_id: SetId):
        df = self._load_csv(self._dataset_dir, set_id)
        if self._is_only_one_pair_per_subject:
            df = self._drop_duplicate_pairs(df)
        if self._only_labeled:
            df = self._keep_only_labeled(df)
        list_of_subjects = self._prepare_list_of_torchio_subjects(df, self._sources)
        labels = df['target'].values
        if set_id == SetId.TRAIN and self._only_labeled:
            self._set_num_classes(labels)
        return (list_of_subjects, labels)

    def _prepare_transforms(
        self,
        set_id: SetId,
        use_separate_augmentation: bool = False
    ):
        assert use_separate_augmentation == False, 'Separate augmentations have not been implemented'
        self.landmarks = {}
        self.train_histogram_standartization()
        canonical = tio.transforms.ToCanonical()
        mask = MNIMaskTransform()
        hist_standard = tio.transforms.HistogramStandardization(
            self.landmarks)
        znorm = tio.transforms.ZNormalization(
            masking_method=tio.transforms.ZNormalization.mean)
        pad_size = self.input_size // 8
        pad = tio.transforms.Pad(
            padding=(
                pad_size, pad_size,
                pad_size, pad_size,
                pad_size, pad_size
            )
        )
        crop = VolumetricRandomCrop(self.input_size)
        flip = tio.transforms.RandomFlip(axes=(0, 1, 2), p=0.5)
        transforms = [mask, canonical]
        transforms.append(hist_standard)
        transforms.append(znorm)
        if (
            self._task_id in [
                TaskId.PRETRAINING,
                TaskId.LINEAR_EVALUATION
            ]
        ) and set_id != SetId.INFER:
            transforms.append(pad)
            transforms.append(crop)
            transforms.append(flip)
        transforms = tio.transforms.Compose(transforms)
        return transforms

    @staticmethod
    def _drop_duplicate_pairs(df: pd.DataFrame):
        logging.debug(f'Shape with multiple pairs per subject {df.shape}')
        df = df.drop_duplicates(subset='subject').reset_index(drop=True)
        logging.debug(f'Shape with only one pair per subject {df.shape}')
        return df

    @staticmethod
    def _keep_only_labeled(df: pd.DataFrame):
        logging.debug('Cleaning labels with -1')
        df = df[df["target"] != -1].reset_index(drop=True)
        logging.debug(f'Shape without label -1 {df.shape}')
        logging.debug(df["target"].value_counts())
        return df

    @staticmethod
    def _load_csv(dataset_dir: str, set_id: SetId):
        csv_file = os.path.join(dataset_dir, f'{set_id}.csv')
        df = pd.read_csv(csv_file)
        return df

    @staticmethod
    def _prepare_list_of_torchio_subjects(df: pd.DataFrame, sources: List[int]):
        list_of_subjects = []
        for i in df.index:
            subject_dict = {}
            for source_id in sources:
                filename = df[f"filename_{source_id + 1}"].iloc[i]
                # print (filename)
                subject_dict[f"source_{source_id}"] = ScalarImage(filename)
            subject_dict['label'] = df.at[i, "target"]
            subject = Subject(subject_dict)
            list_of_subjects.append(subject)
        return list_of_subjects

    def _set_num_classes(self, targets: Tensor):
        self._num_classes = len(torch.unique(targets))
