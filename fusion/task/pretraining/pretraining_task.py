from fusion.architecture import architecture_provider
from fusion.dataset import dataset_provider
from fusion.model import model_provider
from fusion.objective import objective_provider
from fusion.runner import runner_provider
from fusion.task import ATask, ATaskBuilder



class PretrainingTaskBuilder(ATaskBuilder):
    def create_new_task(self, args):
        self._task = PretrainingTask()

    def add_dataset(self, dataset_config):
        self._task.dataset = architecture_provider(
            dataset_config.name, dataset_config.args)


class PretrainingTask(ATask):
    def run(self):
        pass