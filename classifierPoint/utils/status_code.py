from enum import Enum


class STATUS(Enum):
    SUCCESS = 0
    PAUSE = 1
    INITERROR = 2
    FAILURE = 3
    MEMORYERROR = 4
    DATAERROR = 5
