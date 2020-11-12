from fusion.dataset.abasedataset import ABaseDataset
import torchvision


class TwoViewMnist(ABaseDataset):

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
	        self._data_loaders

