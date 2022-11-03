import secrets
from typing import Any, Dict, List

from qstack.qernel import Qernel, QernelArgs
from qstack.types import QOSEngineI, QPUWrapper, Job, distributor_policy
#from qstack.qos import Scheduler

#This engine manages the main queue it is the same as the Global Scheduler
class Distributor(QOSEngineI):
	'''QOS Engine for scheduling qernels on a set of distributed QPUs'''
	'''For now lets just use the IMBQ QPUs, but in the future we might
	need to introduce a `type` variable which indicates which type of QPU we
	are using'''
	
	#_qpus: List[(IBMQQPU, Scheduler)]
	_qpus: List[QPUWrapper]
	policy: str
	job_counter = 0
	'''The distributor has the information about every QPU's queue'''
	queue: List[Job]

	def __init__(self, qpus: List[QPUWrapper], policy:str) -> None:
		self.queue = list()
		self._qpus = qpus

		if (policy == "fifo"):
			self.policy = dist_fifo_policy()
		else:
			raise RuntimeError(
                "[ERROR] - Distributing Policy not implemented."
            )

	def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
		self.queue.append(Job(qernel))
		#pdb.set_trace()
		tmp_qid = -1 #This is to make it work with the internal ids of each QPU, for now

		#Compute all cost of running the qernel on every QPU
		for i in self._qpus:
			tmp_qid=i.register_qernel(qernel, {})
			self.queue[-1].costs[i.backend_name] = i.cost(tmp_qid)
		
		self.job_counter += 1

		return self.job_counter-1

	def execute_qernel(self, qid: int, input: QernelArgs, exec_args: Dict[str, Any]) -> None:
		#TODO
		pass

	def add_QPU(self, new_QPU)->int:
		self._qpus.append(new_QPU)
		return self._qpus.index(new_QPU) #Returns the QPU id

	def remove_QPU(self, qpuid)->int:
		self._qpus.remove(qpuid)
		return qpuid


class dist_fifo_policy(distributor_policy):

	def advise(self, **kargs):
		pass