import secrets
import logging
from typing import Any, Dict, List
import pdb
import time
from qstack.qernel import Qernel, QernelArgs
from qstack.types import QPUWrapper, Job, distributor_policy


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


    def remove_QPU(self, qpu_id) -> int:
        '''
        Simple method to remove a QPU from the list of QPUs
        '''
        
        self._qpus.remove(qpu_id)
        return qpu_id


class fifo_policy(distributor_policy):
    
    def distribute(self, new_job:Job, kargs: Dict) -> QPUWrapper:
        """
        Distributes a new job to a QPU based on the FIFO
        policy and the QPU with the least elements on the queue.
        Ideally this method should not change the queue just send
        which QPU to send to, however I was having some problems
        with the locks, for it is working but might change in the
        future
        """

        all_queues = []
        chosen_qpu:QPUWrapper
        qpu:QPUWrapper
        backends = kargs["backends"]

        '''Locking all the queues from being changed until the policy
        finishes, this is definitly not the best way of doing this, need
        to think of a better way in the future'''            
        for qpu in backends:
            qpu.scheduler.queue_lock.acquire()

        '''Fetches the queues from all the QPU local schedulers. Another way of doing
        this could be by looking at the semaphores values, but in this case we should
        take into consideration that the scheduler is decrementing the semaphore as
        soon as the job arrives and not at the end of running the job'''
        for qpu in backends:
            all_queues.append(len(qpu.scheduler.queue))
        
        chosen_qpu = backends[all_queues.index(min(all_queues))]
        new_job.assiged_qpu = chosen_qpu
        
        logging.log(42, "Assigned %s for job %d - (current queues: %s)", chosen_qpu.backend_name, new_job.id, str(all_queues))

        for qpu in backends:
            qpu.scheduler.queue_lock.release()
        
        chosen_qpu.scheduler.register_job(new_job, 10)

        return backends[all_queues.index(min(all_queues))]
