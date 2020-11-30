from omegaconf import DictConfig, OmegaConf
import hydra
from fusion.task import TaskDirector, task_provider


class Experiment:
    # Singleton
    # To have global within experiments arguments
    def __init__(self, config):
        self._config = config


@hydra.main(config_path="../configs", config_name="default_config")
def my_experiment(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print (cfg.dataset.args.dataset_dir)
    exp = Experiment(cfg)


if __name__ == "__main__":
    my_experiment()

