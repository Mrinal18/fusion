from .catalyst_runner import CatalystRunner


class OasisRunner(CatalystRunner):
    def _unpack_batch(self, batch):
        x = [
            v['data'] for k, v in batch.items() if not k.startswith('label')
        ]
        y = batch["label"]
        return x, y