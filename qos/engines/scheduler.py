from typing import Any, Dict, List
from qos.types import Job
from threading import Thread, Lock, Semaphore
import logging
from qos.backends.test_qpu import TestQPU
from qos.backends.ibmq import IBMQPU
from qos.types import Engine, Job
from qiskit.circuit import QuantumCircuit
import qos.database as db
import json
import pdb


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

        if job.provider == "test":
            qpu = TestQPU()
            results = qpu.run()
        elif job.provider == "ibm":
            qpu = IBMQPU()
            circuit = QuantumCircuit.from_qasm_str(job.circuit)
            print(job.backend)
            print(circuit)
            trans_circuit = qpu.transpile(circuit, job.backend)
            results = qpu.run(trans_circuit, job.backend, job.shots).get_counts()

        db.setJobField(jobId, "status", "DONE")
        db.setJobField(jobId, "results", json.dumps(results))

        return 0
