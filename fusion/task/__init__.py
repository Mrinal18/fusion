from .atask import ATask, ATaskBuilder, TaskDirector
from .pretraining.pretraining_task import PretrainingTaskBuilder
from fusion.utils import ObjectProvider


task_provider = ObjectProvider()
task_provider.register_object('PretrainingTask', PretrainingTaskBuilder)

__all__ = [
    'ATask',
    'ATaskBuilder',
    'TaskDirector',
    'PretrainingTaskBuilder',
    'task_provider',
]
