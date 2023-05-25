from typing import Any, Dict, List
from qos.types import Job
from threading import Thread, Lock, Semaphore
import logging
from qos.backends.test_qpu import TestQPU as tqpu
from qos.types import Engine, Job
from qos.database import database as db
import json


class Scheduler(Engine):

    logger = logging.getLogger(__name__)
    # runner: Thread
    # policy: scheduler_policy

    def __init__(self) -> int:
        # new_thread = Thread(target=self._register_job)
        # new_thread.start()
        # new_thread.join()  # After registering the task exit the thread
        pass

    def submit(self, jobId: int) -> None:

        self.logger.log(10, "Got new job")
        job = db.getJob(jobId)
        results = tqpu.run(job)

        db.setJobField(jobId, "status", "DONE")

        # Check what is happening here, what does dumps do
        db.setJobField(jobId, "results", json.dumps(results))

        return 0
