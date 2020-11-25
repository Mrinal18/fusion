import abc


class TaskDirector():
    def __init__(self, task_builder):
        self._builder = task_builder

    def construct_task(self):
        self._builder.create_new_task()
        self._builder.add_dataset()
        self._builder.add_model()
        self._builder.add_loss()
        self._builder.add_runner()

    def get_task(self):
        return self._builder.task


class TaskBuilder(abc.ABC):
    _task = None

    @abc.abstractmethod
    def create_new_task(self):
        self._task = Task()

    @abc.abstractmethod
    def add_dataset(self):
        pass

    @abc.abstractmethod
    def add_model(self):
        pass

    @abc.abstractmethod
    def add_loss(self):
        pass

    @abc.abstractmethod
    def add_runner(self):
        pass


class Task():
    def __init__(self):
        self._dataset = None
        self._model = None
        self._loss = None
        self._runner = None

    def run(self):
        pass

