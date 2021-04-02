import copy

from catalyst.utils.checkpoint import load_checkpoint, unpack_checkpoint
from omegaconf import DictConfig

from fusion.criterion import criterion_provider
from fusion.model import model_provider
from fusion.optimizer import optimizer_provider
from fusion.runner import runner_provider
from fusion.scheduler import scheduler_provider
from fusion.task import ATask, PretrainingTaskBuilder


class LinearEvaluationTaskBuilder(PretrainingTaskBuilder):
	def create_new_task(self, task_args: DictConfig):
		"""

        :param task_args:
        :return:
        """
		self._task = LinearEvaluationTask(task_args.args)

	def add_model(self, model_config: DictConfig):
		"""

		:param model_config:
		:return:
		"""
		self._task.model = {}
		# get number of classes
		num_classes = self._task.dataset._num_classes
		model_config.args['num_classes'] = num_classes
		pretrained_checkpoint = model_config.args.pretrained_checkpoint
		# create model
		model_args = copy.deepcopy({**model_config.args})
		model_args.pop('pretrained_checkpoint')
		pretrained_model = model_provider.get(
			model_config.name, **model_args
		)
		# load checkpoint
		checkpoint = load_checkpoint(pretrained_checkpoint)
		unpack_checkpoint(checkpoint, pretrained_model)
		# create linear evaluators
		for source_id, encoder in pretrained_model.get_encoder_list().items():
			linear_evaluator_args = {
				'encoder': encoder,
				'num_classes': num_classes,
				'dim_l': model_args['dim_l'],
				'source_id': int(source_id)
			}
			linear_evaluator = model_provider.get(
				'LinearEvaluator', **linear_evaluator_args
			)
			self._task.model[source_id] = linear_evaluator

	def add_criterion(self, criterion_config: DictConfig):
		"""

		:param criterion_config:
		:return:
		"""
		# TODO: add check for CrossEntropy or BinaryCrossEntropyWithLogits
		self._task.criterion = criterion_provider.get(
			criterion_config.name, **criterion_config.args
		)

	def add_runner(self, runner_config: DictConfig):
		"""

		:param runner_config:
		:return:
		"""
		runner_args = {} if runner_config.args is None else runner_config.args
		self._task.runner = runner_provider.get(
			runner_config.name, **runner_args
		)

	def add_optimizer(self, optimizer_config: DictConfig):
		"""

		:param optimizer_config:
		:return:
		"""
		self._task.optimizer = {}
		for source_id, source_model in self._task.model.items():
			args = dict(**optimizer_config.args)
			args['params'] = source_model.parameters()
			optimizer = optimizer_provider.get(
				optimizer_config.name, **args
			)
			self._task.optimizer[source_id] = optimizer

	def add_scheduler(self, scheduler_config: DictConfig):
		"""

		:param scheduler_config:
		:return:
		"""
		self._task.scheduler = {}
		for source_id, _ in self._task.model.items():
			args = dict(**scheduler_config.args)
			args['optimizer'] = self._task.optimizer[source_id]
			args['steps_per_epoch'] = len(
				self._task.dataset.get_loader('train'))
			args['epochs'] = self._task.task_args['num_epochs']
			scheduler = scheduler_provider.get(
				scheduler_config.name, **args
			)
			self._task.scheduler[source_id] = scheduler


class LinearEvaluationTask(ATask):
	def __init__(self, task_args: DictConfig) -> None:
		super().__init__(task_args)

	def run(self):
		for source_id, source_model in self._model.items():
			self._runner.train(
				model=source_model,
				criterion=self._criterion,
				optimizer=self._optimizer[source_id],
				scheduler=self._scheduler[source_id],
				loaders=self._dataset.get_cv_loaders(),
				logdir=self._task_args['logdir'] + f'/linear_{source_id}/',
				num_epochs=self._task_args['num_epochs'],
				verbose=self._task_args['verbose'],
				# TODO: Resume by search in logdir or from hydra config
				resume=self._task_args['resume'],
				timeit=self._task_args['timeit'],
				callbacks=self._callbacks,
			)
