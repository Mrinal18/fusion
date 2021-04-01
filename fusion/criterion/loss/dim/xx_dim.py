from fusion.criterion.loss.dim import BaseDim


XX_MODE = 'XX'


class XxDim(BaseDim):
    _name = XX_MODE

    def __call__(self, reps, convs):
        ret_loss = None
        raw_losses = {}
        dim_conv_latent = 1
        for rep_source_id, rep in reps.items():
            for conv_source_id, conv in convs.items():
                if rep_source_id != conv_source_id:
                    for dim_conv, conv_latent in conv.items():
                        assert dim_conv_latent in rep.keys()
                        loss, penalty = self._estimator(
                           rep[dim_conv_latent],  conv_latent)
                        loss = self._weight * loss
                        name = self._name_it(
                            rep_source_id, conv_source_id, dim_conv)
                        raw_losses[f'{name}_loss'] = loss.item()
                        ret_loss = ret_loss + loss if ret_loss is not None else loss
                        if penalty is not None:
                            raw_losses[f'{name}_penalty'] = penalty.item()
                            ret_loss = ret_loss + penalty if ret_loss is not None else penalty
        return ret_loss, raw_losses

    def _name_it(self, rep_source_id, conv_source_id, dim_conv):
        return f"{self._name}{dim_conv}_" \
               f"{rep_source_id}_{conv_source_id}"
