
from catalyst import dl

from omegaconf import DictConfig
from fusion.dataset import dataset_provider
from fusion.dataset.abasedataset import SetId
from fusion.model import model_provider
from fusion.criterion import criterion_provider
from fusion.optimizer import optimizer_provider
from fusion.runner import runner_provider
from fusion.scheduler import scheduler_provider
from fusion.task import ATask, ATaskBuilder
import logging


class PretrainingTaskBuilder(ATaskBuilder):
    _task: ATask

    def create_new_task(self, task_args: DictConfig):
        """

        Args:
            :param task_args:
        """
        self._task = PretrainingTask(task_args.args)

    def add_dataset(self, dataset_config: DictConfig):
        """

        Args:
            :param dataset_config:
        """
        self._task.dataset = dataset_provider.get(
            dataset_config.name, **dataset_config.args
        )
        self._task.dataset.load()

    def add_model(self, model_config: DictConfig):
        """

        Args:
            :param model_config:
        """
        if 'num_classes' in model_config.args.keys():
            model_config.args['num_classes'] = self._task.dataset._num_classes
        model_args = {**model_config.args}
        model_args.pop('pretrained_checkpoint')
        self._task.model = model_provider.get(
            model_config.name, **model_args
        )

    def add_criterion(self, criterion_config: DictConfig):
        """

        Args:
            :param criterion_config:
        """
        args = {} if criterion_config.args is None else criterion_config.args
        self._task.criterion = criterion_provider.get(
            criterion_config.name, **args
        )

    def add_runner(self, runner_config: DictConfig):
        """

        Args:
            :param runner_config:
        """
        runner_args = {} if runner_config.args is None else runner_config.args
        self._task.runner = runner_provider.get(
            runner_config.name, **runner_args
        )

    def add_optimizer(self, optimizer_config: DictConfig):
        """

        Args:
            :param optimizer_config:
        """
        args = dict(**optimizer_config.args)
        args['params'] = self._task.model.parameters()
        self._task.optimizer = optimizer_provider.get(
            optimizer_config.name, **args
        )

    def add_scheduler(self, scheduler_config: DictConfig):
        """

        Args:
            :param scheduler_config:
        """
        args = dict(scheduler_config.args)
        args['optimizer'] = self._task.optimizer
        args['steps_per_epoch'] = len(
            self._task.dataset.get_loader(SetId.TRAIN))
        args['epochs'] = self._task.task_args['num_epochs']
        self._task.scheduler = scheduler_provider.get(
            scheduler_config.name, **args
        )


class PretrainingTask(ATask):
    def __init__(self, task_args: DictConfig) -> None:
        """
        Initilization of class Pretraining Task
        	:param task_args: task parameters
        Return:
        	class Logical Pretraining Task
        """
        super().__init__(task_args)

    def run(self):
        """
        Method launch training of Pretraining Task
        """
        logging.info(f"logdir: {self._task_args['logdir']}")
        self._callbacks = [
            dl.CheckpointCallback(
                logdir=self._task_args['logdir'], loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
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
