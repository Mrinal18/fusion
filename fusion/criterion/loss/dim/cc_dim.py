from fusion.criterion.loss.dim import BaseDim


CC_MODE = 'CC'


class CcDim(BaseDim):
    _name = CC_MODE

    def __call__(self, reps, convs):
        ret_loss = 0
        raw_losses = {}
        for rep_source_id, rep in reps.items():
            for conv_source_id, conv in convs.items():
                if rep_source_id != conv_source_id:
                    for dim_conv in conv.keys():
                        assert dim_conv in rep.keys()
                        loss, penalty = self._estimator(
                            conv[dim_conv], rep[dim_conv])
                        loss = self._weight * loss
                        name = self._name_it(
                            rep_source_id, conv_source_id, dim_conv)
                        raw_losses[f'{name}_loss'] = loss
                        ret_loss += loss
                        if penalty is not None:
                            raw_losses[f'{name}_penalty'] = penalty
                            ret_loss += penalty
        return ret_loss, raw_losses

    def _name_it(self, rep_source_id, conv_source_id, dim_conv):
        return f"{self._name}{dim_conv}_" \
               f"{rep_source_id}_{conv_source_id}"
