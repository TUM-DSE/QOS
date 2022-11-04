from abc import ABC, abstractmethod
from typing import Any, Dict, List
import sys

from qstack.qernel import Qernel, QernelArgs

class QOSEngineI(ABC):
    """Generic interface for implementations of QOS Layers"""

    @abstractmethod
    def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
        pass

    @abstractmethod
    def execute_qernel(self, qid: int, args: QernelArgs, shots: int) -> None:
        pass

class Job():
	'''Job class that holds the Qernel to be run and the runinng costs for all QPUs'''
	'''If we have alot of QPUs this might create a large overhead'''
	_qernel:Qernel
	costs:Dict[str, float]

	def __init__(self, qernel) -> None:
		self._qernel = qernel
		self.costs={}

class Scheduler_base(QOSEngineI, ABC):
	'''Local scheduler abstract class so we can use it in this file without
	importing the scheduler file because it creates a circular dependency
	But it can't be really abstract because this attribute is needed'''
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
	def advise(self, **kargs):
		pass


class distributor_policy(ABC):

	@abstractmethod
	def advise(self, **kargs):
		pass