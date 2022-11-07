from types import MethodType
from typing import Any, Dict, Optional
from warnings import warn

import random

from qstack.qos.scheduler import Scheduler, scheduler_policy, fifo_policy
from qstack.qernel import Qernel, QernelArgs
from qstack.types import QPUWrapper


class TestQPU(QPUWrapper):
	
	#Added this, check if you agree
	backend_name: str
	_qernels: Dict[int, Qernel]
	gen_seed: int
	policy: scheduler_policy
	scheduler: Scheduler
	
	def __init__(self, backend_name: str, gen_seed:int, policy:str) -> None:
		self.backend_name = backend_name
		self._qernels = {}
		self.scheduler = Scheduler()
		self._qid_ctr: int = 0
		self.gen_seed = gen_seed

		if(policy == 'fifo'):
			self.policy = fifo_policy()

	def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
		self._qernels[self._qid_ctr] = qernel
		self._qid_ctr += 1
		return self._qid_ctr - 1

	def execute_qernel(self, qid: int, args: QernelArgs, shots: int) -> None:
		qernel = self._qernels[qid]
		print("Executing Qernel", qid)

	def cost(self, qernel: Qernel) -> float:
		#curr_qernel = self._qernels[qid]
		random.seed(self.gen_seed)
		tmp = random.randint(0,50)/100
		return tmp

	def overhead(self, qid: int) -> int:
		pass
