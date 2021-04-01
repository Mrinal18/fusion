from catalyst import dl
from fusion.dataset import dataset_provider
from fusion.model import model_provider
from fusion.criterion import criterion_provider
from fusion.optimizer import optimizer_provider
from fusion.runner import runner_provider
from fusion.scheduler import scheduler_provider
from fusion.task import ATask, ATaskBuilder
import logging


class PretrainingTaskBuilder(ATaskBuilder):
    def create_new_task(self, task_args):
        """

        :param task_args:
        :return:
        """
        self._task = PretrainingTask(task_args.args)

    def add_dataset(self, dataset_config):
        """

        :param dataset_config:
        :return:
        """
        self._task.dataset = dataset_provider.get(
            dataset_config.name, **dataset_config.args
        )
        self._task.dataset.load()

    def add_model(self, model_config):
        """

        :param model_config:
        :return:
        """
        if 'num_classes' in model_config.args.keys():
            model_config.args['num_classes'] = self._task.dataset._num_classes
        model_args = {**model_config.args}
        model_args.pop('pretrained_checkpoint')
        self._task.model = model_provider.get(
            model_config.name, **model_args
        )

    def add_criterion(self, criterion_config):
        """
        :param criterion_config:
        :return:
        """
        args = {} if criterion_config.args is None else criterion_config.args
        self._task.criterion = criterion_provider.get(
            criterion_config.name, **args
        )

    def add_runner(self, runner_config):
        """

        :param runner_config:
        :return:
        """
        runner_args = {} if runner_config.args is None else runner_config.args
        self._task.runner = runner_provider.get(
            runner_config.name, **runner_args
        )

    def add_optimizer(self, optimizer_config):
        """

        :param optimizer_config:
        :return:
        """
        args = dict(**optimizer_config.args)
        args['params'] = self._task.model.parameters()
        self._task.optimizer = optimizer_provider.get(
            optimizer_config.name, **args
        )

    def add_scheduler(self, scheduler_config):
        """

        :param scheduler_config:
        :return:
        """
        args = dict(scheduler_config.args)
        args['optimizer'] = self._task.optimizer
        args['steps_per_epoch'] = len(
            self._task.dataset.get_loader('train'))
        args['epochs'] = self._task.task_args['num_epochs']
        self._task.scheduler = scheduler_provider.get(
            scheduler_config.name, **args
        )


class PretrainingTask(ATask):
    def __init__(self, task_args) -> None:
        super(PretrainingTask, self).__init__(task_args)

    def run(self):
        logging.info(f"logdir: {self._task_args['logdir']}")
        self._callbacks = [
            dl.CheckpointCallback(
                logdir=self._task_args['logdir'], loader_key="infer", metric_key="loss", minimize=True, save_n_best=3
            ),
        ]
        self._runner.train(
            model=self._model,
            criterion=self._criterion,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            loaders=self._dataset.get_cv_loaders(),
            #logdir=self._task_args['logdir'],
            num_epochs=self._task_args['num_epochs'],
            verbose=self._task_args['verbose'],
            # resume=self._task_args['resume'],
            timeit=self._task_args['timeit'],
            callbacks=self._callbacks,
        )
