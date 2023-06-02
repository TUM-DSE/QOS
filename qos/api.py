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

        self.logger.log(10, "New job added to the database")

        self.workers.append(
            threading.Thread(target=self.transformer_submit, args=(newQC,))
        )

        self.logger.log(10, "Opening new thread, sumbitting QC to transformer")
        self.workers[-1].start()

        return QCId

    def results(self, jobId: int) -> None:

        stat = database.getJobField(jobId, "status").decode("utf-8")

        if stat == "DONE":
            return database.getJobField(jobId, "results")
        else:
            return 1

    def transformer_submit(self, qc: QCircuit) -> None:
        self.transformer.submit(qc)
