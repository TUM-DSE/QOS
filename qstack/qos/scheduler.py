from typing import Any, Dict, List

from qstack.qernel.qernel import Qernel, QernelArgs
from qstack.types import QOSEngineI, scheduler_policy, Job, Scheduler_base

class Scheduler(Scheduler_base):
	'''Local scheduler. Each instance of it will run on a separate thread'''

	def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
		#TODO
		pass

	def execute_qernel(self, qid: int, args: QernelArgs, shots: int) -> None:
		#TODO
		pass


class fifo_policy(scheduler_policy):
	'''First Come First Served Policy or First In First Out'''
	'''
	This policy works with a single-queue. The scheduler sends/executes the
	oldest job on the queue
	'''

	def advise(self, run_costs:Dict[str, Any]):
		#TODO
		pass