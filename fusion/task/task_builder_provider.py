from .pretraining import PretrainingTaskBuilder
from .linear_evaluation import LinearEvaluationTaskBuilder
from .misc import TaskId

from fusion.utils import ObjectProvider


task_builder_provider = ObjectProvider()
task_builder_provider.register_object(TaskId.PRETRAINING, PretrainingTaskBuilder)
task_builder_provider.register_object(
    TaskId.LINEAR_EVALUATION, LinearEvaluationTaskBuilder)
