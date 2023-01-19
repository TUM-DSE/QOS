from typing import Any, Dict, List

class Scheduler(Scheduler_base):
	'''Local scheduler. Each instance of it will run on a separate thread'''
	#queue:List[Job]

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

	def advise(self, run_costs:Dict[str, Any], ):
		#TODO
		pass

'''
import secrets
from typing import Any, Dict, List

from qstack.qernel import Qernel, QernelArgs
from qstack.types import QOSEngineI, QPUWrapper, Job, distributor_policy
#from qstack.qos import Scheduler

#This engine manages the main queue it is the same as the Global Scheduler
class Distributor(QOSEngineI):
	QOS Engine for scheduling qernels on a set of distributed QPUs
	For now lets just use the IMBQ QPUs, but in the future we might
	need to introduce a `type` variable which indicates which type of QPU we
	are using
	
	# _qpus: List[(IBMQQPU, Scheduler)]
	_qpus: List[QPUWrapper]
	policy: distributor_policy
	qernel_id_counter = 0
	The distributor has the information about every QPU's queue
	#_queue: List[Job]

	def __init__(self, qpus: List[QPUWrapper], policy:str) -> None:
		#self._queue = [-1] # To avoid the queue from being removed by the garbage collector if there are no objects on the list
		self._qpus = qpus

		if (policy == "fifo"):
			self.policy = fifo_policy()
		else:
			raise RuntimeError("[ERROR] - Distributing Policy not implemented.")

	def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
		running_costs = {}
		advise_qpu:QPUWrapper

		# Compute all cost of running the qernel on every QPU
		for i in self._qpus:
			running_costs[i.backend_name] = i.cost(qernel)

		# Ask for advice from the distributor policy
		advise_qpu = self.policy.advise({"backends":self._qpus, "costs":running_costs})

		# Send the job to the QPU as advised by the policy
		print("Job sent to QPU", advise_qpu.backend_name)

		#advise_qpu.execute_qernel(self.qernel_id_counter, None, 10)

		self.qernel_id_counter +=1

		return self.qernel_id_counter-1

	def execute_qernel(self, qid: int, args: QernelArgs, shots: int) -> None:
		pass

	def add_QPU(self, new_QPU)->int:
		self._qpus.append(new_QPU)
		return self._qpus.index(new_QPU) #Returns the QPU id

	def remove_QPU(self, qpuid)->int:
		self._qpus.remove(qpuid)
		return qpuid


class fifo_policy(distributor_policy):

	def advise(self, kargs:Dict) -> QPUWrapper:
		This method simply advises and does not change the queue.
		Since this is the FIFO policy it simply returns the zero index and
		the QPU with the smallest queue

		all_queues = []
		backends = kargs["backends"] # This returns all the backends/QPU available to the Distributor

		# Fetches the queues from all the QPU local schedulers to find the queue
		# with the least number of jobs
		qpu:QPUWrapper
		for qpu in backends:
			all_queues.append(len(qpu.scheduler.queue))
		
		# Returns the queue with the least number of jobs
		return backends[min(all_queues)]
'''