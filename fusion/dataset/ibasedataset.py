import zope.interface


class IBaseDataset(zope.interface.Interface):

    _dataset_dir = zope.interface.Attribute(
        """Dataset directory path"""
    )
    _num_classes = zope.interface.Attribute(
        """Number of classes"""
    )
    _fold = zope.interface.Attribute(
        """Fold number"""
    )
    _num_folds = zope.interface.Attribute(
        """Number of folds"""
    )
    _views = zope.interface.Attribute(
        """List of selected views"""
    )
    _data_loaders = zope.interface.Attribute(
        """Dictionary with data loaders"""
    )
    # Data loader parameters
    _batch_size = zope.interface.Attribute(
        """Batch size"""
    )
    _shuffle = zope.interface.Attribute(
        """Shuffle the data"""
    )
    _drop_last = zope.interface.Attribute(
        """Drop last bath"""
    )
    _num_workers = zope.interface.Attribute(
        """Number of workers of dataloader"""
    )

    def load():
        """Loads the dataset
        """
        pass

    def get_all_loaders():
        """Returns dictionary with data loaders
        """
        pass

    def get_cv_loaders():
        """Returns dictionary with cross-validation loaders
        """
        pass

    def get_loader(set_id):
        """Returns loader with specific set

        Args:
            set_id (str): "\'train\', \'val\', \'hold_out_test\'"
        """
        pass

    def num_classes():
        """Number of classes
        """
        pass

    def _prepare_transforms(set_id):
        """Creates set of data transforms for specific set of data.

        Args:
            set_id (str): "\'train\', \'val', \'hold_out_test\'"
        """
        pass
