import zope.interface


class IBaseDataset(zope.interface.Interface):

    num_classes = zope.interface.Attribute(
        """Number of classes"""
    )
    data_loaders = zope.interface.Attribute(
        """Dictionary with data loaders"""
    )

    def load():
        """Loads the dataset
        """
        pass

    @zope.interface.interfacemethod
    def get_all_loaders():
        """Returns dictionary with data loaders
        """
        return data_loaders

    @zope.interface.interfacemethod
    def get_cv_loaders():
        """Returns dictionary with cross-validation loaders
        """
        return {key: data_loaders[key] for key in ['train', 'val']}

    def get_loader(set_id):
        """Returns loader with specific set

        Args:
            set_id (str): "\'train\', \'val'\, \'hold_out_test\'"
        """
        pass

    @zope.interface.interfacemethod
    def num_classes():
        """Number of classes
        """
        return num_classes

    def _prepare_transforms(set_id):
        """Creates set of data transforms for specific set of data.

        Args:
            set_id (str): "\'train\', \'val'\, \'hold_out_test\'"
        """
        pass
