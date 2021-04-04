from fusion.criterion.loss.dim import BaseDim


RR_MODE = 'RR'


class RrDim(BaseDim):
    _name = RR_MODE

    def __call__(self, reps, convs):
        ret_loss = None
        raw_losses = {}
        dim_conv_latent = 1
        for rep_source_id_one, rep_one in reps.items():
            for rep_source_id_two, rep_two in reps.items():
                if rep_source_id_one != rep_source_id_two:
                    assert dim_conv_latent in rep_two.keys()
                    assert dim_conv_latent in rep_one.keys()
                    name = self._name_it(
                        rep_source_id_one, rep_source_id_two, dim_conv_latent
                    )
                    loss, penalty = self._estimator(
                        rep_one[dim_conv_latent], rep_one[dim_conv_latent]
                    )
                    ret_loss, raw_losses = self._update_loss(
                        name, ret_loss, raw_losses, loss, penalty
                    )
        return ret_loss, raw_losses

    def _name_it(self, rep_source_id, conv_source_id, dim_conv):
        return f"{self._name}{dim_conv}_" \
               f"{rep_source_id}_{conv_source_id}"
