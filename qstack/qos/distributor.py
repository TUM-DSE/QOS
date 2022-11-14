import secrets
import logging
from typing import Any, Dict, List
import pdb
import time
from qstack.qernel import Qernel, QernelArgs
from qstack.types import QPUWrapper, Job, distributor_policy

# from qstack.qos import Scheduler

# This engine manages the main queue it is the same as the Global Scheduler
class Distributor:
    """QOS Engine for scheduling qernels on a set of distributed QPUs"""

    """For now lets just use the IMBQ QPUs, but in the future we might
	need to introduce a `type` variable which indicates which type of QPU we
	are using"""

    # _qpus: List[(IBMQQPU, Scheduler)]
    _qpus: List[QPUWrapper]
    policy: distributor_policy
    job_id_counter = 0
    """The distributor has the information about every QPU's queue"""
    # _queue: List[Job]

    def __init__(self, qpus: List[QPUWrapper], policy: str) -> None:
        # self._queue = [-1] # To avoid the queue from being removed by the garbage collector if there are no objects on the list
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

        # Compute all cost of running the qernel on every QPU
        for i in self._qpus:
            running_costs[i.backend_name] = i.cost(qernel)

        # logging.debug("Costs: ", {"backends": self._qpus, "costs": running_costs})
        # print("Costs: ", {"backends": self._qpus, "costs": running_costs})

        choosen_qpu = self.policy.distribute(new_job,
            {"backends": self._qpus, "costs": running_costs}
        )

        logging.info(
            "[INFO] - Job %d will be sent to QPU: %s",
            self.job_id_counter,
            choosen_qpu.backend_name,
        )
        # pdb.set_trace()

        return self.job_id_counter - 1

    def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
        pass

    def execute_qernel(self, qid: int, args: QernelArgs, shots: int) -> None:
        pass

    def add_QPU(self, new_QPU) -> int:
        self._qpus.append(new_QPU)
        return self._qpus.index(new_QPU)  # Returns the QPU id

    def remove_QPU(self, qpuid) -> int:
        self._qpus.remove(qpuid)
        return qpuid


class fifo_policy(distributor_policy):
    def distribute(self, new_job:Job, kargs: Dict) -> QPUWrapper:
        """This method simply advises and does not change the queue."""
        """Since this is the FIFO policy it simply returns the zero index and
		the QPU with the smallest queue"""

        print("Distributing job")

        all_queues = []
        choosen_qpu:QPUWrapper
        backends = kargs[
            "backends"
        ]  # This returns all the backends/QPU available to the Distributor

        # Fetches the queues from all the QPU local schedulers to find the queue
        # with the least number of jobs
        qpu: QPUWrapper
        for qpu in backends:
            qpu.scheduler.queue_lock.acquire()

        for qpu in backends:
            all_queues.append(len(qpu.scheduler.queue))
        
        # Puts job on the queue with the least number of jobs
        choosen_qpu = backends[all_queues.index(min(all_queues))]
        new_job.assiged_qpu = choosen_qpu

        print("Registring now")
        
        choosen_qpu.scheduler.register_job(new_job, 10)

        for qpu in backends:
            #if qpu == choosen_qpu:
            #    continue
            qpu.scheduler.queue_lock.release()
            #print(qpu.scheduler)

        time.sleep(5)

        #choosen_qpu.scheduler.queue_lock.release()

        print("Current queues:", all_queues)
        print("Choosen QPU:", choosen_qpu)

        return backends[all_queues.index(min(all_queues))]
