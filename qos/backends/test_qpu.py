from types import MethodType
from typing import Any, Dict, Optional
from warnings import warn
from qos.backends.types import QPU
import logging
from time import sleep

# from qos.scheduler_old import Scheduler, scheduler_policy, fifo_policy
# from qstack.qernel import Qernel, QernelArgs
# from qstack.types import QPUWrapper


class TestQPU(QPU):
    logger: logging.Logger

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        return

    def run(self, circuit) -> int:
        self.logger.log(10, "Running qernel")
        sleep(5)
        return 42
