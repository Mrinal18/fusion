from catalyst import dl
from fusion.runner import ABaseRunner
from typing import Mapping, Any


class CatalystRunner(ABaseRunner, dl.Runner):
    def predict_batch(
        self,
        batch: Mapping[str, Any],
        **kwargs
    ) -> Mapping[str, Any]:
        x, y = batch
        return self.model([x_.to(self.device) for x_ in x]), y

    def _handle_batch(self, batch: Mapping[str, Any]) -> None:
        x, y = batch
        outputs = self.model(x)
        loss = self.criterion(outputs, y)

        if isinstance(loss, tuple):
            loss, raw_losses = loss
            self.batch_metrics = {"loss": loss}
            self.batch_metrics.update(raw_losses)
        else:
            self.batch_metrics = {"loss": loss}

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
