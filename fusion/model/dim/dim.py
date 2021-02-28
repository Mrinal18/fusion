from fusion.model import AMultiSourceModel


class Dim(AMultiSourceModel):
    def __init__(
        self,
        architecture,
        architecture_params
    ):
        # create encoders for each view
        super(Dim, AMultiSourceModel).__init__(architecture, architecture_params)
        # create convolutional heads

        # create latent heads