<<<<<<< HEAD
<<<<<<< HEAD
import copy
=======
>>>>>>> 1) Add linear_evaluation
=======
import copy
>>>>>>> Latest state. Fixed some bugs with respect hydra. Added pretrained checkpoint
from fusion.model import model_provider
from fusion.criterion import criterion_provider
from fusion.optimizer import optimizer_provider
from fusion.runner import runner_provider
from fusion.scheduler import scheduler_provider
from fusion.task import ATask, PretrainingTaskBuilder
from catalyst.utils.checkpoint import load_checkpoint, unpack_checkpoint


class LinearEvalualtionTaskBuilder(PretrainingTaskBuilder):

    def add_model(self, model_config):
        self._task.model = {}
        # get number of classes
<<<<<<< HEAD
<<<<<<< HEAD
        num_classes = self._task.dataset._num_classes
        model_config.args['num_classes'] = num_classes
        pretrained_checkpoint = model_config.args.pretrained_checkpoint
<<<<<<< HEAD
        # create model
        model_args = copy.deepcopy({**model_config.args})
        model_args.pop('pretrained_checkpoint')
        pretrained_model = model_provider.get(
            model_config.name, **model_args
        )
        # load checkpoint
        checkpoint = load_checkpoint(pretrained_checkpoint)
=======
        num_classes = self._task.dataset.num_classes
=======
        num_classes = self._task.dataset._num_classes
>>>>>>> Fix the hybrid config and add some fixes to make code run
        model_config.args['num_classes'] = num_classes
=======
>>>>>>> Latest state. Fixed some bugs with respect hydra. Added pretrained checkpoint
        # create model
        model_args = copy.deepcopy({**model_config.args})
        model_args.pop('pretrained_checkpoint')
        pretrained_model = model_provider.get(
            model_config.name, **model_args
        )
        # load checkpoint
<<<<<<< HEAD
        checkpoint = load_checkpoint(model_config.pretrained_checkpoint)
>>>>>>> 1) Add linear_evaluation
=======
        checkpoint = load_checkpoint(pretrained_checkpoint)
>>>>>>> Latest state. Fixed some bugs with respect hydra. Added pretrained checkpoint
        unpack_checkpoint(checkpoint, pretrained_model)
        # create linear evaluators
        for id_view, encoder in pretrained_model.get_encoder_list():
            linear_evaluator_args = {
                # TODO: name of arguments for LinearEvaluator can change
                'encoder': encoder,
                'num_classes': num_classes,
                'view': id_view
            }
            linear_evaluator = model_provider(
                # TODO: name of model can change and string is bad 
                'LinearEvaluatorWithEncoder', **linear_evaluator_args
            )
            self._task.model[id_view] = linear_evaluator

    def add_criterion(self, criterion_config):
        # TODO: add check for CrossEntropy or BinaryCrossEntropyWithLogits
        self._task.criterion = criterion_provider.get(
            criterion_config.name, **criterion_config.args
        )

    def add_runner(self, runner_config):
<<<<<<< HEAD
<<<<<<< HEAD
        runner_args = {} if runner_config.args is None else runner_config.args
=======
>>>>>>> 1) Add linear_evaluation
=======
        runner_args = {} if runner_config.args is None else runner_config.args
>>>>>>> Fix the hybrid config and add some fixes to make code run
        self._task.runner = runner_provider.get(
            runner_config.name, **runner_config.args
        )

    def add_optimizer(self, optimizer_config):
        self._task.optimizer = {}
        for id_view, view_model in self._task.model.items():
            args = dict(**optimizer_config.args)
            args['params'] = view_model.parameters()
            optimizer = optimizer_provider.get(
                optimizer_config.name, **args
            )
            self._task.optimizer[id_view] = optimizer

    def add_scheduler(self, scheduler_config):
        self._task.scheduler = {}
        for id_view, view_model in self._task.model.items():
            args = dict(**scheduler_config.args)
            args['optimizer'] = self._task.optimizer[id_view]
            args['steps_per_epoch'] = len(
                self._task.dataset.get_loader('train'))
            args['epochs'] = self._task.task_args['num_epochs']
            scheduler = scheduler_provider.get(
                scheduler_config.name, **args
            )
            self._task.scheduler[id_view] = scheduler


class LinearEvalualtionTask(ATask):
    def __init__(self, task_args) -> None:
        super(LinearEvalualtionTask, self).__init__(task_args)

    def run(self):
        for id_view, _ in self._model.keys():
            self._runner.train(
                model=self._model[id_view],
                criterion=self._criterion,
                optimizer=self._optimizer[id_view],
                scheduler=self._scheduler[id_view],
                loaders=self._dataset.get_cv_loaders(),
                logdir=self._task_args['logdir'] + f'/linear_{id_view}/',
                num_epochs=self._task_args['num_epochs'],
                verbose=self._task_args['verbose'],
                # TODO: Resume by search in logdir or from hydra config
                resume=self._task_args['resume'] + f'/linear_{id_view}/',
                timeit=self._task_args['timeit'],
                callbacks=self._callbacks,
            )
