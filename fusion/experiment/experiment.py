from fusion.task import TaskDirector, task_builder_provider
import hydra
from omegaconf import DictConfig, OmegaConf

class Experiment:
    # Singleton
    # To have global within experiments arguments
    def __init__(self, config: DictConfig):
        """

        :param config:
        """
        print(OmegaConf.to_yaml(config))
        self._config = config
        self._task = None

    def setup_new_experiment(self):
        """

        :return:
        """
        task_builder = task_builder_provider.get(self._config.task.name)
        task_director = TaskDirector(task_builder, self._config)
        task_director.construct_task()
        self._task = task_director.get_task()

    def start(self):
        self._task.run()
