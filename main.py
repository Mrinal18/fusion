# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from fusion.experiment import Experiment
import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(config_path="fusion/configs", config_name="default_config")
def my_experiment(cfg: DictConfig) -> None:
    exp = Experiment(cfg)
    exp.setup_new_experiment()
    exp.start()


if __name__ == "__main__":
    my_experiment()