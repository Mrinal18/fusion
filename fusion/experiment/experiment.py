from catalyst.utils.misc import set_global_seed

from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import random


from fusion.task import TaskDirector, task_builder_provider


class Experiment:
    # Singleton
    # To have global within experiments arguments
    def __init__(self, config: DictConfig):
        """

        config:
        """
        print(OmegaConf.to_yaml(config))
        self._config = config["experiment"]
        self._task = None
        self._seed = self._config["seed"]

    def setup_new_experiment(self):
        """

        :return:
        """
        np.random.seed(self._seed)
        random.seed(self._seed)
        torch.manual_seed(self._seed)
        set_global_seed(self._seed)
        # torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)
        task_builder = task_builder_provider.get(self._config.task.name)
        task_director = TaskDirector(task_builder, self._config)
        task_director.construct_task()
        self._task = task_director.get_task()

    def start(self):
        self._task.run()
