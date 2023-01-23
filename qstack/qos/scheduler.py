from typing import Any, Dict, List
from qstack.types import scheduler_policy, Job, Scheduler_base
from threading import Thread, Lock, Semaphore
import logging

class Scheduler(Scheduler_base):
    """Local scheduler"""

    executor: Thread
    queue: List[Job]
    queue_lock: Lock
    queue_counter: Semaphore
    policy: scheduler_policy

    def __init__(self, policy: str):
        self.queue_lock = Lock()
        self.queue = []
        self.queue_counter = Semaphore(0)

        if policy == "fifo":
            self.policy = fifo_policy()
        else:
            raise RuntimeError("Scheduling Policy not implemented")

        self.executor = Thread(target=self._executor, args=[])
        self.executor.start()

    def register_job(self, job: Job, kargs: Dict[str, Any]) -> int:
        
        new_thread = Thread(target=self._register_job, args=[job, kargs])
        new_thread.start()
        new_thread.join() # After registering the task exit the thread
        return 0

    def _register_job(self, job: Job, compile_args: Dict[str, Any]) -> None:
        ''' For now the advise method is actually changing the queue itself instead of
        proposing where the next job should go, this way I left the acquire and
        release here, just because the changing of the queue might should be outside of
        the advise method'''

        logging.log(42, "Registering new job")

        self.policy.schedule(self.queue, job)
        self.queue_counter.release()
        return

    def _executor(self) -> None:
        
        while 1: # When should this thread be stop? Or maybe it should indefinitely
            self.queue_counter.acquire() # If the queue_counter semaphore is higher than 0 there are job to run
            next_job = self.queue[0] # While the job is being executed it stays on the queue.
            
            qpu = next_job.assiged_qpu
            logging.log(42, 'Running task %d on %s', next_job.id, qpu.backend_name)
            qpu._backend.run(next_job._qernel, blocking=True)
            logging.log(42, "Done running task %d on %s, removing from queue", next_job.id, qpu.backend_name)

            self.queue_lock.acquire()
            self.queue.remove(next_job) # I cant simply take the first one because there might be a new job as the top of the queue
            self.queue_lock.release()
            

class fifo_policy(scheduler_policy):
    """First Come First Served Policy or First In First Out"""

    def schedule(self, queue: List[Job], new_job: Job):

        logging.log(42, "Adding new job to queue on %s", new_job.assiged_qpu.backend_name)
        queue.append(new_job)
