from types import MethodType
from typing import Any, Dict, Optional
from warnings import warn
from qos.backends.types import QPU
import random
import logging
from time import sleep

# from qos.scheduler_old import Scheduler, scheduler_policy, fifo_policy
# from qstack.qernel import Qernel, QernelArgs
# from qstack.types import QPUWrapper


class TestQPU(QPU):
    def __init__() -> None:
        pass

    def run(self) -> int:
        sleep(5)
        return 42
