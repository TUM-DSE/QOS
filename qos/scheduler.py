from typing import Any, Dict, List

from components.types import QOSEngine, Task, Backend

class Scheduler(QOSEngine):
	#queue:List[Job]

	def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
		#TODO
		pass

	def execute_qernel(self, qid: int, args: QernelArgs, shots: int) -> None:
		#TODO
		pass


class fifo_policy(scheduler_policy):
	#First Come First Served Policy or First In First Out
	#This policy works with a single-queue. The scheduler sends/executes the
	#oldest job on the queue

	def advise(self, run_costs:Dict[str, Any], ):
		#TODO
		pass

'''
import secrets
import logging
from typing import Any, Dict, List
import pdb
import time
from qstack.qernel import Qernel, QernelArgs
from qstack.types import QPUWrapper, Job, distributor_policy

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

class Distributor:
    """
    Engine for assigning qernels to QPUs based on a specific policy set
    by the user.
    """

    _qpus: List[QPUWrapper]
    policy: distributor_policy
    job_id_counter = 0

    def __init__(self, qpus: List[QPUWrapper], policy: str) -> None:
        self._qpus = qpus

        if policy == "fifo":
            self.policy = fifo_policy()
        else:
            raise RuntimeError("[ERROR] - Distributing Policy not implemented.")

    def run_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
        """
        This is supposed to be used from outside the class and before this a job does
        not exist yet. When a Qernel is registered here using this method a new
        job is created to be filled with the respective costs for each QPU, and
        the selected QPU afterwards
        """

        running_costs = {}
        choosen_qpu: QPUWrapper
        self.job_id_counter += 1
        new_job = Job(qernel, self.job_id_counter)

        ''' TODO - Compute all cost of running the qernel on every QPU, this is the job of the
        matching engine, should be moved from here'''
        for i in self._qpus:
            running_costs[i.backend_name] = i.cost(qernel)

        # Just to format the cost string to show on the debugs log
        aux_string = str([(i.backend_name, running_costs[i.backend_name]) for i in self._qpus]).replace("\',", ":").replace("[","").replace("{","").replace("}","").replace("]", "").replace("'", "")
        logging.log(42, "Costs: %s", aux_string)

        choosen_qpu = self.policy.distribute(new_job,
            {"backends": self._qpus,
            "costs": running_costs})

        logging.info("Job %d will be sent to %s", self.job_id_counter, choosen_qpu.backend_name)

        return self.job_id_counter - 1

    def add_QPU(self, new_QPU) -> int:
        '''
        Simple method to add a QPU to the list of QPUs without
        the need recreate the Distributor object
        '''

        self._qpus.append(new_QPU)
        return self._qpus.index(new_QPU)  # Returns the QPU id


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
