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


class Scheduler(ABC):
	'''Base Abstract Scheduler Class'''

	'''The advise method outputs where should the distributor insert the new job
	based on the input that the scheduler policy needs'''
	@abstractmethod
	def advise(self, **kwargs):
		pass


class fifo_policy(Scheduler):
	'''First Come First Served Policy or First In First Out'''
	'''
	This policy works with a single-queue. The scheduler assigns the oldest
	job on the queue to the 
	'''
	def advise(self, run_costs:Dict[str, Any], ):
		#TODO
		pass