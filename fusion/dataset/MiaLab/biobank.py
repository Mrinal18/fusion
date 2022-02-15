from catalyst.data.loader import BatchPrefetchLoaderWrapper
import copy
import os
from typing import Any, Dict, List, Union, Optional

from sklearn.model_selection import StratifiedKFold
import torch
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchnet.dataset import TensorDataset, ResampleDataset

from fusion.dataset.utils import seed_worker
from fusion.dataset.abasedataset import ABaseDataset, SetId
from fusion.dataset.mnist_svhn.transforms import SVHNTransform, MNISTTransform


class BioBankDataset(ABaseDataset):
    """
    This Class is for the BioBank Dataset which has the information on the 
    """
    
    def __init__(self, 
                root: str,
                set_id: SetId,
                transform: Optional[torchvision.transforms.Compose] = None,
                download: bool = False,
                ):
        """
        Args:
            root: path to the root of the dataset
            set_id: set id of the dataset
            transform: transform to apply to the dataset
            download: whether to download the dataset
        """
        super().__init__(root, set_id, transform, download)
        self.root = root
        self.set_id = set_id
        self.transform = transform
        self.download = download
        self.data_dir = os.path.join(self.root, self.set_id)

        self.train_data = None

    def data_loader(self, batch_size: int, shuffle: bool = True, **kwargs) -> DataLoader:
        "" "Return a data loader for the dataset."""
        if self.train_data is None:
            self.train_data = self.train_data_loader(batch_size, shuffle, **kwargs)
        return self.train_data

    def train_data_loader(self, batch_size: int, shuffle: bool = True, **kwargs) -> DataLoader:
        """
        Args: 
            batch_size: batch size of the data loader
            shuffle: whether to shuffle the data
            **kwargs: keyword arguments to pass to the data loader with respect to the SMIR dataset
        """
        train_data = BioBankDataset(
            root=self.root,
            set_id=self.set_id,
            transform=self.transform,
            download=self.download,
        )
        
    

        train_data_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )

        return train_data_loader



    def _dataset_into_nii(self, data_dir, set_id):
        """
        Args:
        data_dir: path to the data directory
        set_id: set id of the dataset

        Returns:
        nii_image: nii image of the dataset
        """

        nii_image = nib.load(os.path.join(data_dir, set_id))
        return nii_image
        
    def reshape_nii_image(self, nii_image):
        """
        Args:
        nii_image: nii image of the dataset
            
        Returns:    
        nii_image: nii image of the dataset
        """
        
        nii_image = nii_image.get_fdata()
        #used if we need to reshape the image to a specific size
        nii_image = nii_image.reshape(nii_image.shape[0], nii_image.shape[1], nii_image.shape[2], 1)

        return nii_image

    
    def _get_data(self, data_dir, set_id):
        """
        Args:
        data_dir: path to the data directory
        set_id: set id of the dataset
            
            Returns:
            data: data of the dataset
        """

        nii_image = self._dataset_into_nii(data_dir, set_id)
        nii_image = self.reshape_nii_image(nii_image)
        data = nii_image

        return data


    def _get_labels(self, data_dir, set_id):
        """
        Args:
        data_dir: path to the data directory
        set_id: set id of the dataset

        Returns:
        labels: labels of the dataset
        """

        labels = self._dataset_into_nii(data_dir, set_id)
        labels = self.reshape_nii_image(labels)
        labels = labels.astype(int)

        return labels
        

