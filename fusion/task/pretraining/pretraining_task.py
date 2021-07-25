from catalyst import dl

from omegaconf import DictConfig
from fusion.dataset import dataset_provider
from fusion.dataset.misc import SetId
from fusion.model import model_provider
from fusion.criterion import criterion_provider
from fusion.optimizer import optimizer_provider
from fusion.runner import runner_provider
from fusion.scheduler import scheduler_provider
from fusion.task import ATask, ATaskBuilder
import logging


class PretrainingTaskBuilder(ATaskBuilder):
    _task: ATask

    def create_new_task(self, task_args: DictConfig, seed: int = 343):
        """
        Method for create new pretraining task
        Args:
        task_args: dictionary with task's parameters from config
        """
        self._task = PretrainingTask(task_args.args, seed=seed)

    def add_dataset(self, dataset_config: DictConfig):
        """
        Method for add dataset to pretraining task
        Args:
                dataset_config: dictionary with dataset's parameters from config
        """
        self._task.dataset = dataset_provider.get(
            dataset_config.name, **dataset_config.args
        )
        self._task.dataset.load()

    def add_model(self, model_config: DictConfig):
        """
        Method for add model to pretraining task
        Args:
            model_config: dictionary with model's parameters from config
        """
        if "num_classes" in model_config.args.keys():
            model_config.args["num_classes"] = self._task.dataset._num_classes
        model_args = {**model_config.args}
        model_args.pop("pretrained_checkpoint")
        self._task.model = model_provider.get(model_config.name, **model_args)

    def add_criterion(self, criterion_config: DictConfig):
        """
        Method for add criterion to pretraining task
        Args:
            criterion_config: dictionary with criterion's parameters from config
        """
        args = {} if criterion_config.args is None else criterion_config.args
        self._task.criterion = criterion_provider.get(criterion_config.name, **args)

    def add_runner(self, runner_config: DictConfig):
        """
        Method for add runner to pretraining task
        Args:
            runner_config: dictionary with runner's parameters from config
        """
        runner_args = {} if runner_config.args is None else runner_config.args
        self._task.runner = runner_provider.get(runner_config.name, **runner_args)

    def add_optimizer(self, optimizer_config: DictConfig):
        """
        Method for add optimizer to pretraining task
        Args:
            optimizer_config: dictionary with optimizer's parameters from config
        """
        args = dict(**optimizer_config.args)
        args["params"] = self._task.model.parameters()
        self._task.optimizer = optimizer_provider.get(optimizer_config.name, **args)

    def add_scheduler(self, scheduler_config: DictConfig):
        """
        Method for add scheduler to pretraining task
        Args:
            scheduler_config: dictionary with scheduler's parameters from config
        """
        args = dict(scheduler_config.args)
        args["optimizer"] = self._task.optimizer
        args["steps_per_epoch"] = len(self._task.dataset.get_loader(SetId.TRAIN))
        args["epochs"] = self._task.task_args["num_epochs"]
        self._task.scheduler = scheduler_provider.get(scheduler_config.name, **args)


class PretrainingTask(ATask):
    def run(self):
        """
        Method launch training of Pretraining Task
        """
        self._loggers = {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._task_args["logdir"]),
            "tensorboard": dl.TensorboardLogger(logdir=self._task_args["logdir"]),
            "wandb": dl.WandbLogger(
                project=self._task_args["project"],
                name=self._task_args["name"]
            )
        }
        logging.info(f"logdir: {self._task_args['logdir']}")
        self._callbacks = [
            dl.CheckpointCallback(
                logdir=self._task_args["logdir"],
                loader_key="valid",
                metric_key="loss",
                minimize=True,
                save_n_best=3,
            ),
        ]
        self._runner.train(
            model=self._model,
            criterion=self._criterion,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            loaders=self._dataset.get_cv_loaders(),
            logdir=self._task_args["logdir"],
            num_epochs=self._task_args["num_epochs"],
            verbose=self._task_args["verbose"],
            # resume=self._task_args['resume'],
            timeit=self._task_args["timeit"],
            callbacks=self._callbacks,
            loggers=self._loggers
        )
