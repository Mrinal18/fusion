from fusion.criterion.loss.dim import BaseDim


CR_MODE = 'CR'


class CrDim(BaseDim):
    _name = CR_MODE

    def __call__(self, reps, convs):
        ret_loss = 0
        raw_losses = {}
        dim_conv_latent = 1
        for source_id, rep in reps.items():
            for dim_conv, conv in convs[source_id].items():
                assert dim_conv_latent in rep.keys()
                loss, penalty = self._estimator(
                    conv, rep[dim_conv_latent])
                loss = self._weight * loss
                name = self._name_it(source_id, dim_conv)
                raw_losses[f'{name}_loss'] = loss
                ret_loss += loss
                if penalty is not None:
                    raw_losses[f'{name}_penalty'] = penalty
                    ret_loss += penalty
        return ret_loss, raw_losses

    def _name_it(self, source_id, dim_conv):
        return f"{self._name}{dim_conv}_{source_id}"
