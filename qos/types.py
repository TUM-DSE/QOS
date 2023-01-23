from abc import ABC, abstractmethod
from typing import Any, Dict, List
import sys

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
	_qernel:Qernel
	costs:Dict[str, float]

	def __init__(self, qernel) -> None:
		self._qernel = qernel
		self.costs={}


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
	def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
		pass
	
	@abstractmethod
	def execute_qernel(self, qid: int, args: QernelArgs, shots: int) -> None:
		pass

class QPUWrapper(QOSEngineI, ABC):

	#Added this, check if you agree
	backend_name:str
	scheduler:Scheduler_base

	@abstractmethod
	def cost(self, qernel: Qernel) -> float:
		pass

	@abstractmethod
	def overhead(self, qid: int) -> int:
		pass

class scheduler_policy(ABC):

	@abstractmethod
	def advise(self, kargs:Dict):
		pass


class distributor_policy(ABC):

	@abstractmethod
	def advise(self, kargs:Dict) -> QPUWrapper:
		pass
<<<<<<< HEAD:qstack/types.py
=======
'''
>>>>>>> main:qos/types.py
