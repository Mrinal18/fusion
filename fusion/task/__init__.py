from .atask import ATask, ATaskBuilder, TaskDirector
from .pretraining import PretrainingTaskBuilder
from .linear_evaluation import LinearEvalualtionTaskBuilder
from fusion.utils import ObjectProvider


task_builder_provider = ObjectProvider()
task_builder_provider.register_object(
    'PretrainingTask', PretrainingTaskBuilder)
task_builder_provider.register_object(
    'LinearEvaluationTask', LinearEvalualtionTaskBuilder)

__all__ = [
    'ATask',
    'ATaskBuilder',
    'TaskDirector',
    'PretrainingTaskBuilder',
    'task_builder_provider',
]
