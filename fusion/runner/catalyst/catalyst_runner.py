import logging
from catalyst import dl, metrics
from catalyst.typing import Scheduler
from fusion.runner import ABaseRunner
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, \
    OneCycleLR, CyclicLR
import torch.nn.functional as F
from typing import Mapping, Any


class CatalystRunner(ABaseRunner, dl.Runner):
    epoch = 0
    batch_id = 0

    def predict_batch(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """

        Args:

            batch:
            kwargs:
        Return

        """
        x, y = self._unpack_batch(batch)
        return self.model([x_.to(self.device) for x_ in x]), y

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        """

        batch:
        :return:
        """
        if self.is_train_loader:
            for param in self.model.parameters():
                param.grad = None

        x, y = self._unpack_batch(batch)
        outputs = self.model(x)
        loss = self.criterion(outputs, y)

        if isinstance(loss, tuple):
            loss, raw_losses = loss
            self.batch_metrics.update(raw_losses)

        if self.is_train_loader:
            self.batch_id += 1
            loss.backward()
            self.optimizer.step()
            if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler.step(
                    epoch=int(self.epoch + self.batch_id / len(self._loaders['train'])))
            elif isinstance(self.scheduler, (OneCycleLR, CyclicLR)):
                self.scheduler.step()
            else:
                raise NotImplementedError


        self.batch_metrics["loss"] = loss.item()
        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key], self.batch_size)

        self.batch = {"targets": y}

        for source_id, source_z in outputs.z.items():
            probs = F.softmax(source_z, dim=1)
            self.batch[f"logits_{source_id}"] = source_z
            self.batch[f"probs_{source_id}"] = probs

    def get_loaders(self, stage):
        return self._loaders

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False) for key in ["loss"]
        }

    def on_loader_end(self, runner):
        for key in ["loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        if self.is_train_loader:
            self.epoch += 1
            self.batch_id = 0
        super().on_loader_end(runner)

    def _unpack_batch(self, batch):
        x, y = batch
        return x, y
