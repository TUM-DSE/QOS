from typing import Any, Dict, List
from qos.types import Job
from .engines.transformer import Transformer
import qos.database as database
import redis
import logging
from qos.types import QCircuit
from qiskit import QuantumCircuit
import qos.tools
import threading
import pdb

from qos.tools import debugPrint


class QOS:
    """Main API that will be exposed to the user"""

    transformer: Transformer
    logger = logging.getLogger(__name__)
    workers: List[
        threading.Thread
    ]  # I think we wont be needing this, the API just issues the job and does nothing else with it

    def __init__(self) -> None:
        # self.db = database()
        self.logger = logging.getLogger(__name__)

        # Ignoring certain loggers
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("qiskit").setLevel(logging.WARNING)
        logging.getLogger("stevedore").setLevel(logging.WARNING)

        self.transformer = Transformer()
        self.logger.log(10, "QOS API initialized")
        self.workers = []

        qos.tools.load_qpus("qpus_available")

        print("Available QPUs:")
        for i in range(1, 5):
            print(database.getQPU(i))

    def run(self, circuit: Any) -> int:

        newQC = QCircuit()
        newJob = Job()

        if type(circuit) == QuantumCircuit:
            newQC.type = "qiskit"

        newJob.circuit = circuit.qasm()

        print("Circuit:", newJob.circuit)

        # Adds the job to the database
        QCId = database.addQC(newQC)
        newQC.id = QCId

        # Adds the job to the database
        jobId = database.addJob(newJob)
        newJob.id = jobId

        self.logger.log(10, "New job added to the database")

        self.workers.append(
            threading.Thread(target=self.transformer_submit, args=(newJob,))
        )

        self.logger.log(10, "Opening new thread, sumbitting QC to transformer")
        self.workers[-1].start()

        return jobId

    def results(self, jobId: int) -> None:

        stat = database.getJobField(jobId, "status")

        if stat == b"DONE":
            job = database.getJob(jobId)
            # Probably now we would process the results from the subjobs and return the final job result
            # pdb.set_trace()
            # tmpjob = database.getJob(job.subjobs[0])
            results = database.getJobField(job.subjobs[0], "results").decode()
            return results
        else:
            return 1

    def transformer_submit(self, job: Job) -> None:
        # pdb.set_trace()
        self.transformer.submit(job)
