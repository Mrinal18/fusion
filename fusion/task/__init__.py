from .atask import ATask, ATaskBuilder, TaskDirector
from .pretraining import PretrainingTaskBuilder
from .linear_evaluation import LinearEvaluationTaskBuilder
from dataclasses import dataclass
from fusion.utils import ObjectProvider


@dataclass
class TaskId():
    PRETRAINING = 'PretrainingTask'
    LINEAR_EVALUATION = 'LinearEvaluationTask'


task_builder_provider = ObjectProvider()
task_builder_provider.register_object(
    TaskId.PRETRAINING, PretrainingTaskBuilder)
task_builder_provider.register_object(
    TaskId.LINEAR_EVALUATION, LinearEvaluationTaskBuilder)

__all__ = [
    'ATask',
    'ATaskBuilder',
    'TaskDirector',
    'PretrainingTaskBuilder',
    'task_builder_provider',
]
