from .pretraining import PretrainingTaskBuilder
from .linear_evaluation import LinearEvaluationTaskBuilder
from .logreg_evaluation import LogRegEvaluationTaskBuilder
from .misc import TaskId
from .saliency import SaliencyTaskBuilder

from fusion.utils import ObjectProvider


task_builder_provider = ObjectProvider()
task_builder_provider.register_object(TaskId.PRETRAINING, PretrainingTaskBuilder)
task_builder_provider.register_object(
    TaskId.LINEAR_EVALUATION, LinearEvaluationTaskBuilder
)
task_builder_provider.register_object(
    TaskId.LOGREG_EVALUATION, LogRegEvaluationTaskBuilder
)
task_builder_provider.register_object(
    TaskId.SALIENCY, SaliencyTaskBuilder
)