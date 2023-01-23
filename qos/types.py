from abc import ABC, abstractmethod
from typing import Any, Dict, List
import sys
from threading import Thread, Lock, Semaphore


class QOSEngine(ABC):
    """Generic interface for implementations of QOS Engines"""

    @abstractmethod
    def register(self, args: Dict[str, Any]) -> int:
        pass

    @abstractmethod
    def execute(self, id: int, args: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def fetch(self, id: int, args: Dict[str, Any]) -> None:
        pass


# Task class that holds a single circuit to be run and the runinng costs for all QPUs
# If we have alot of QPUs this might create a large overhead
class Task(ABC):
    _qernel: Qernel
    costs: Dict[str, float]


class Scheduler_base(ABC):
    """Local scheduler abstract class so we can use it in this file without
    importing the scheduler file because it creates a circular dependency
    But it can't be really abstract because this attribute is needed"""


# Generic interface for implementations of QOS Engines
class Backend(ABC):
    @abstractmethod
    def run(self, args: Dict[str, Any]) -> int:
        pass


'''
class Scheduler_base(QOSEngineI, ABC):
	#Local scheduler abstract class so we can use it in this file without
	#importing the scheduler file because it creates a circular dependency
	#But it can't be really abstract because this attribute is needed
	queue:List[Job]

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

<<<<<<< HEAD:qstack/types.py
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
    def schedule(self, new_job: Job, kargs: Dict):
        pass


class distributor_policy(ABC):
    @abstractmethod
    def distribute(self, kargs: Dict) -> QPUWrapper:
        pass
=======
	@abstractmethod
	def advise(self, kargs:Dict) -> QPUWrapper:
		pass
<<<<<<< HEAD:qstack/types.py
=======
'''
