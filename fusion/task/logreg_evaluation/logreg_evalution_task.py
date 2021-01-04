from fusion.dataset import dataset_provider
from fusion.model import model_provider
from fusion.criterion import criterion_provider
from fusion.optimizer import optimizer_provider
from fusion.runner import runner_provider
from fusion.scheduler import scheduler_provider
from fusion.task import ATask, ATaskBuilder


class LogRegTaskBuilder(ATaskBuilder):
    def create_new_task(self, task_args):
        self._task = LogRegTask(task_args.args)

    def add_dataset(self, dataset_config):
        self._task.dataset = dataset_provider.get(
            dataset_config.name, **dataset_config.args
        )
        self._task.dataset.load()

    def add_model(self, model_config):
        model_config.args['num_classes'] = self._task.dataset.num_classes
        self._task.model = model_provider.get(
            model_config.name, **model_config.args
        )

    def add_criterion(self, criterion_config):
        self._task.criterion = criterion_provider.get(
            criterion_config.name, **criterion_config.args
        )

    def add_runner(self, runner_config):
        self._task.runner = runner_provider.get(
            runner_config.name, **{}
        )

    def add_optimizer(self, optimizer_config):
        args = dict(**optimizer_config.args)
        args['params'] = self._task.model.parameters()
        self._task.optimizer = optimizer_provider.get(
            optimizer_config.name, **args
        )

    def add_scheduler(self, scheduler_config):
        args = dict(**scheduler_config.args)
        args['optimizer'] = self._task.optimizer
        args['steps_per_epoch'] = len(
            self._task.dataset.get_loader('train'))
        args['epochs'] = self._task.task_args['num_epochs']
        self._task.scheduler = scheduler_provider.get(
            scheduler_config.name, **args
        )


class LogRegTask(ATask):
    def __init__(self, task_args) -> None:
        super(LogRegTask, self).__init__(task_args)

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
            timeit=self._task_args['timeit'],
            callbacks=self._callbacks,
        )
