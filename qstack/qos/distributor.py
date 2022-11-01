import secrets
from backends import IBMQQPU
from typing import Any, Dict, List

from qstack.qernel.qernel import Qernel, QernelArgs
from qstack.types import QOSEngineI, QPUWrapper, Job, distributor_policy
from scheduler import Scheduler


#This engine manages the main queue it is the same as the Global Scheduler
class Distributor(QOSEngineI):
	'''QOS Engine for scheduling qernels on a set of distributed QPUs'''
	'''For now lets just use the IMBQ QPUs, but in the future we might
	need to introduce a `type` variable which indicates which type of QPU we
	are using'''
	
	_qpus: List[(IBMQQPU, Scheduler)]
	policy: str

	'''The distributor has the information about every QPU's queue'''
	queue: List[Job]

	#We should be able to say that this __init__ can receive a List[QPUWrapper]
	#instead of specifying that it is an IBMQPU. I dont know how. Or maybe it is
	#just my IDE that is giving me an error when there is none.
	def __init__(self, qpus: List[IBMQQPU], policy:str) -> None:
		self.queue = list()
		self._qpus = qpus

		if (policy == "fifo"):
			self.policy = dist_fifo_policy()
		else:
			raise RuntimeError(
                "[ERROR] - Distributing Policy not implemented."
            )

	def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
		self.queue.append(Job())
		tmp_qid = -1 #This is to make it work with the internal ids of each QPU, for now

		#Compute all cost of running the qernel on every IBMQQPU
		for i in self._qpus:
			tmp_qid=i.register_qernel(qernel, {})
			self.queue[-1].costs[i.backend_name] = i.cost(tmp_qid)

		#This is useless here only makes sense to the IBMQQPU
		return tmp_qid

	def execute_qernel(self, qid: int, input: QernelArgs, exec_args: Dict[str, Any]) -> None:
		#TODO
		pass


class dist_fifo_policy(distributor_policy):

	def advise(self, **kargs):
		pass