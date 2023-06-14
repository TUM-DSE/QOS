from typing import Any, Dict, List
from qos.types import Job
from threading import Thread, Lock, Semaphore
import logging
from qos.backends.test_qpu import TestQPU
from qos.backends.ibmq import IBMQPU
from qos.types import Engine, Job, QCircuit
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

    def submit(self, job: Job) -> None:

        self.logger.log(10, "Got new circuit to be scheduled")
        results = {"test": 43}

        all_qpus = db.getAllQPU()

        # This would be the time that the scheduler would choose a QPU to run the job
        job.qpu = all_qpus[0]
        job.qpu.shots = 1000

        if job.qpu.provider == "test":
            qpu = TestQPU()
            results = qpu.run()
        elif job.qpu.provider == "ibm":
            qpu = IBMQPU()
            circuit = QuantumCircuit.from_qasm_str(job.circuit)
            print(job.qpu.name)
            print(circuit)
            trans_circuit = qpu.transpile(circuit, job.qpu.name)
            results = qpu.run(trans_circuit, job.qpu.name, job.qpu.shots).get_counts()

        # Here the scheduler would do its job

        db.setJobField(job.id, "status", "DONE")
        db.setJobField(job.id, "results", json.dumps(results))

        # stat = db.getJobField(job.id, "status").decode("utf-8")
        exit()

        return 0
