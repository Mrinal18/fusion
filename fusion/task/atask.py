import abc


class TaskDirector:
    def __init__(self, task_builder, config):
        self._builder = task_builder
        self._config = config

    def construct_task(self):
        self._builder.create_new_task(self._config.task)
        self._builder.add_dataset(self._config.dataset)
        self._builder.add_model(self._config.model)
        self._builder.add_criterion(self._config.criterion)
        self._builder.add_runner(self._config.runner)
        self._builder.add_optimizer(self._config.optimizer)
        self._builder.add_scheduler(self._config.scheduler)

    def get_task(self):
        return self._builder.task


class ATaskBuilder(abc.ABC):
    _task = None

    @abc.abstractmethod
    def create_new_task(self, args):
        pass

    @abc.abstractmethod
    def add_dataset(self, dataset_config):
        pass

    @abc.abstractmethod
    def add_model(self, model_config):
        pass

    @abc.abstractmethod
    def add_criterion(self, criterion_config):
        pass

    @abc.abstractmethod
    def add_runner(self, runner_config):
        pass

    @abc.abstractmethod
    def add_optimizer(self, optimizer_config):
        pass

    @abc.abstractmethod
    def add_scheduler(self, scheduler_config):
        pass


class ATask(abc.ABC):
    _dataset = None
    _model = None
    _criterion = None
    _optimizer = None
    _scheduler = None
    _runner = None
    _callbacks = None

    @abc.abstractmethod
    def __init__(self, task_args):
        self._task_args = task_args

    @abc.abstractmethod
    def run(self):
        pass

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, criterion):
        self._criterion = criterion

    @property
    def runner(self):
        return self._runner

    @runner.setter
    def runner(self, runner):
        self._runner = runner

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler):
        self._scheduler = scheduler

    @property
    def task_args(self):
        return self._task_args
