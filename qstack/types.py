from abc import ABC, abstractmethod
from typing import Any, Dict, List

from qstack.qernel import Qernel, QernelArgs


class QOSEngineI(ABC):
    """Generic interface for implementations of QOS Layers"""

    @abstractmethod
    def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
        pass

    @abstractmethod
    def execute_qernel(self, qid: int, args: QernelArgs, shots: int) -> None:
        pass

class QPUWrapper(QOSEngineI, ABC):
    @abstractmethod
    def cost(self, qid: int) -> float:
        pass

    @abstractmethod
    def overhead(self, qid: int) -> int:
        pass


class Job():
	'''Job class that holds the Qernel to be run and the runinng costs for all QPUs'''
	'''If we have alot of QPUs this might create a large overhead'''
	_qernel:Qernel
	costs:Dict[str, float]

	def __init__(self, qernel) -> None:
		self._qernel = qernel
		self.costs={}


class scheduler_policy(ABC):

	@abstractmethod
	def advise(self, **kargs):
		pass


class distributor_policy(ABC):

	@abstractmethod
	def advise(self, **kargs):
		pass