from typing import Any, Dict, List
#from multiprocessing import Process, Lock, Semaphore
from threading import Thread, Lock, Semaphore
import time
import logging

from qstack.qernel.qernel import Qernel, QernelArgs
from qstack.types import QOSEngineI, scheduler_policy, Job, Scheduler_base

class Scheduler(Scheduler_base):
    """Local scheduler"""

    #executor: Process
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

        format = "%(asctime)s: %(message)s"
        logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
        self.executor = Thread(target=self._executor, args=[])
        self.executor.start()
        #self.executor.join()

    def register_job(self, job: Job, kargs: Dict[str, Any]) -> int:
        new_thread = Thread(target=self._register_job, args=[job, kargs])
        new_thread.start()
        new_thread.join()
        # This could return this position in which the qernel was inserted on the queue
        return 0

    def _register_job(self, job: Job, compile_args: Dict[str, Any]) -> None:
        logging.info("Opened new thread to register job")
        #self.queue_lock.acquire()
        logging.info("Acquired queue lock to register job")
        # For now the advise method is actually changing the queue itself instead of
        # proposing where the next job should go, this is way I left the acquire and
        # release here, just because the changing of the queue might should be outside of
        # the advise method
        #logging.info("Queue before [%d jobs]", self.queue_counter.value)
        self.policy.schedule(self.queue, job)
        self.queue_counter.release()
        #logging.info("Queue before [%d jobs]", self.queue_counter.value)
        #exit(0)
        #self.queue_lock.release()
        logging.info("Relesead queue lock")
        return

    def _executor(self) -> None:
        logging.info("Executor thread started")  # When should this thread be killed?
        while 1:
            print("Waiting for new jobs on the queue")
            # If the queue_counter semaphore is higher than 0 there are job to run
            self.queue_counter.acquire()
            self.queue_lock.acquire()
            print("Got it, taking the new job on the queue")
            next_job = self.queue.pop(0)
            # TODO Execute next_job on the QPU
            # next_job.assiged_qpu.execute_qernel(next_job._qernel)
            assigned_qpu_name = next_job.assiged_qpu.backend_name
            print(
                "Executing Qernel",
                next_job.id,
                "[ETA:",
                assigned_qpu_name,
                "]",
            )
            next_job.assiged_qpu._backend.run(next_job._qernel)
            #time.sleep(2)
            ## execute_qernel(next_job._qernel,None,10)
            #print("Done executing", next_job.id)
            print("Done executing")

            self.queue_lock.release()


class fifo_policy(scheduler_policy):
    """First Come First Served Policy or First In First Out"""

    def schedule(self, queue: List[Job], new_job: Job):
        print("Giving advice")
        # This is a mess, I should restructure this
        queue.append(new_job)
        print("After adding job to queue:", queue)
