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

    def submit(self, job: Job, policy) -> None:

        self.logger.log(10, "Got new job to be scheduled")
        # results = {"test": 43}

        policy(job)  # Assign qpus to the subjobs

        for i in job.subjobs:

            tmpjob = db.getJob(i)
            tmpjob.qpu = db.getQPU_fromname(tmpjob.args[b"qpu"].decode())

            if tmpjob.qpu.provider == "test":
                qpu = TestQPU()
                results = qpu.run()
            elif tmpjob.qpu.provider == "ibm":
                qpu = IBMQPU()
                circuit = QuantumCircuit.from_qasm_str(tmpjob.circuit.decode())
                print(tmpjob.qpu.name)
                print(circuit)
                trans_circuit = qpu.transpile(circuit, tmpjob.qpu.name)
                results = qpu.run(
                    trans_circuit, tmpjob.qpu.name, tmpjob.shots
                ).get_counts()

            # Here the scheduler would do its job

            self.logger.log(10, "Got results from qpu, updating")

            db.setJobField(tmpjob.id, "status", "DONE")
            db.setJobField(tmpjob.id, "results", json.dumps(results))

        # This is not supposed to be like this, this should run on a thread and update the database when the job is done in the cloud
        db.setJobField(job.id, "status", "DONE")

        # stat = db.getJobField(job.id, "status").decode("utf-8")

        return 0

    def _bestqpu_policy(self, new_job: Job) -> None:
        self.logger.log(10, "Running best qpu policy")
        # pdb.set_trace()
        for i in new_job.subjobs:
            tmpjob = db.getJob(i)
            tmpjob.args["qpu"] = tmpjob.best_qpu()
            tmpjob.args["shots"] = 1000
            db.updateJob(i, tmpjob)
        return
