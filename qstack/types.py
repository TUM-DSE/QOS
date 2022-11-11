from abc import ABC, abstractmethod
from typing import Any, Dict, List
import sys
from threading import Thread, Lock, Semaphore

from qstack.qernel import Qernel, QernelArgs


class QOSEngineI(ABC):
    """Generic interface for implementations of QOS Layers"""

    @abstractmethod
    def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
        pass

    @abstractmethod
    def execute_qernel(self, qid: int, args: QernelArgs, shots: int) -> None:
        pass


class Scheduler_base(ABC):
    """Local scheduler abstract class so we can use it in this file without
    importing the scheduler file because it creates a circular dependency
    But it can't be really abstract because this attribute is needed"""

    """For now registering a qernel will open a new thread to register the qernel and exit just after that.
	This is not the best implementation in terms of performance, it would be best to have permanently two runnuing threads
	one for registering and another for executing qernels"""
    # register_thread:threading.Thread
    # executer_thread:threading.Thread

    @abstractmethod
    def register_job(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
        pass

    @abstractmethod
    def _executor(self, qid: int, args: QernelArgs, shots: int) -> None:
        pass


class QPUWrapper(ABC):

    # Added this, check if you agree
    backend_name: str
    scheduler: Scheduler_base

    @abstractmethod
    def cost(self, qernel: Qernel) -> float:
        pass

    @abstractmethod
    def overhead(self, qid: int) -> int:
        pass


class Job:
    """Job class that holds the Qernel to be run and the runinng costs for all QPUs"""

    """If we have alot of QPUs this might create a large overhead"""
    _qernel: Qernel
    id: int
    costs: Dict[str, float]
    assiged_qpu: QPUWrapper

    def __init__(self, qernel: Qernel, id: int) -> None:
        self._qernel = qernel
        self.costs = {}
        self.id = id


class scheduler_policy(ABC):
    @abstractmethod
    def advise(self, new_job: Job, kargs: Dict):
        pass


class distributor_policy(ABC):
    @abstractmethod
    def advise(self, kargs: Dict) -> QPUWrapper:
        pass
