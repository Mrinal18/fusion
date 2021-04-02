from fusion.criterion.loss.dim import BaseDim


CR_MODE = 'CR'


class CrDim(BaseDim):
    _name = CR_MODE

    def __call__(self, reps, convs):
        ret_loss = None
        raw_losses = {}
        dim_conv_latent = 1
        for source_id, rep in reps.items():
            for dim_conv, conv in convs[source_id].items():
                assert dim_conv_latent in rep.keys()
                name = self._name_it(source_id, dim_conv)
                loss, penalty = self._estimator(
                    conv, rep[dim_conv_latent]
                )
                ret_loss, raw_losses = self._update_loss(
                    name, ret_loss, raw_losses, loss, penalty
                )
        return ret_loss, raw_losses

    def _name_it(self, source_id, dim_conv):
        return f"{self._name}{dim_conv}_{source_id}"
