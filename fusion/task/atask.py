import abc


class TaskDirector():
    def __init__(self, task_builder, config):
        self._builder = task_builder
        self._config = config

    def construct_task(self):
        self._builder.create_new_task(self._config.task)
        self._builder.add_dataset(self._config.dataset)
        self._builder.add_model(self._config.model)
        self._builder.add_objective(self._config.objective)
        self._builder.add_runner(self._config.runner)

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
    def add_objective(self, objective_config):
        pass

    @abc.abstractmethod
    def add_runner(self, runner_config):
        pass


class ATask(abc.ABC):
    def __init__(self):
        self._dataset = None
        self._model = None
        self._objective = None
        self._runner = None

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
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, objective):
        self._objective = objective

    @property
    def runner(self):
        return self._runner

    @runner.setter
    def runner(self, runner):
        self._runner = runner

    @abc.abstractmethod
    def run(self):
        pass

