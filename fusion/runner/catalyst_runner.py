from catalyst import dl, metrics
from fusion.runner import ABaseRunner
import torch.nn.functional as F
from typing import Mapping, Any


class CatalystRunner(ABaseRunner, dl.Runner):

    def predict_batch(
        self,
        batch: Mapping[str, Any],
        **kwargs
    ) -> Mapping[str, Any]:
        """

        :param batch:
        :param kwargs:
        :return:
        """
        x, y = batch
        return self.model([x_.to(self.device) for x_ in x]), y

    # ToDo: _handle_batch -> handle_batch Catalyst v21
    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        """

        :param batch:
        :return:
        """
        x, y = self._unpack_batch(batch)
        outputs = self.model(x)
        loss = self.criterion(outputs, y)

        if isinstance(loss, tuple):
            loss, raw_losses = loss
            self.batch_metrics.update(raw_losses)

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.batch_metrics['loss'] = loss.item()
        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key], self.batch_size)

        self.batch = {"targets": y}
        # ToDo: Add self.batch for callbacks
        for source_id, source_z in outputs.z.items():
            probs = F.softmax(source_z, dim=-1)
            self.batch = {
                f"logits_{source_id}": source_z,
                f"probs_{source_id}": probs
            }

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in ["loss"]
        }

    def on_loader_end(self, runner):
        for key in ["loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)
