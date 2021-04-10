from typing import Tuple, Optional, Dict, Any

from fusion.criterion.loss import ABaseLoss
from fusion.criterion.loss.dim import CR_MODE
from fusion.criterion.loss.multi_dim import SpatialMultiDim, VolumetricMultiDim
from fusion.criterion.misc.utils import total_loss_summation
from fusion.model.misc import ModelOutput
from fusion.utils import Setting

from torch import Tensor


def choose_multi_dim(input_dim):
    if input_dim == 2:
        multi_dim_type = SpatialMultiDim
    elif input_dim == 3:
        multi_dim_type = VolumetricMultiDim
    else:
        raise NotImplementedError
    return multi_dim_type

class CR_CCA(ABaseLoss):
    def __init__(
        self,
        dim_cls,
        input_dim,
        estimator_setting: Setting,
        cca_setting: Setting
    ):
        super().__init__()
        self._cr_loss = choose_multi_dim(input_dim)(
            dim_cls,
            estimator_setting=estimator_setting,
            modes=[CR_MODE],
            weights=[1.]
        )
        self._cca_loss = cca_setting.class_type(**cca_setting.args)

    def forward(
        self,
        preds: ModelOutput,
        target: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        total_loss = None
        raw_losses = {}
        loss, cr_raw = self._cr_loss(preds)
        raw_losses.update(cr_raw)
        total_loss = total_loss_summation(total_loss, loss)
        loss, cca_raw = self._cca_loss(preds)
        raw_losses.update(cca_raw)
        total_loss = total_loss_summation(total_loss, loss)
        return total_loss, raw_losses
