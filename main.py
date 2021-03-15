from fusion.experiment import Experiment
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="fusion/configs", config_name="default_pretraining")
def my_experiment(cfg: DictConfig) -> None:
    exp = Experiment(cfg)
    exp.setup_new_experiment()
    exp.start()


if __name__ == "__main__":
    my_experiment()
