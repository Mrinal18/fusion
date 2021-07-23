from enum import auto
from strenum import LowercaseStrEnum


class TaskId(LowercaseStrEnum):
    PRETRAINING: "TaskId" = auto()
    LINEAR_EVALUATION: "TaskId" = auto()
    LOGREG_EVALUATION: "TaskId" = auto()
