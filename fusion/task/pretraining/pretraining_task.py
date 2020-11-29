from fusion.architecture import architecture_provider
from fusion.dataset import dataset_provider
from fusion.model import model_provider
from fusion.criterion import criterion_provider
from fusion.runner import runner_provider
from fusion.task import ATask, ATaskBuilder


class PretrainingTaskBuilder(ATaskBuilder):
    def create_new_task(self, task_args):
        self._task = PretrainingTask(task_args)

    def add_dataset(self, dataset_config):
        self._task.dataset = dataset_provider(
            dataset_config.name, dataset_config.args
        )

    def add_model(self, model_config):
        self._task.model = model_provider(
            model_config.name, model_config.args
        )

    def add_criterion(self, criterion_config):
        self.criterion = criterion_provider(
            criterion_config.name, criterion_config.args
        )

    def add_runner(self, runner_config):
        self.runner = runner_provider(
            runner_config.name, runner_config.args
        )


class PretrainingTask(ATask):
    def __init__(self, task_args) -> None:
        super(PretrainingTask, self).__init__(task_args)

    def run(self):
        self._runner.train(
            model=self._model,
            criterion=self._criterion,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            loaders=self._dataset.get_cv_loaders(),
            logdir=self._task_args['logdir'],
            num_epochs=self._task_args['num_epochs'],
            verbose=self._task_args['verbose'],
            resume=self._task_args['resume'],
            timeit=True,
            callbacks=self._callbacks,
        )
