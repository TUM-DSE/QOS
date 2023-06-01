from typing import Any, Dict, List
from qos.types import Job
from .engines.transformer import Transformer
import qos.database as database
import redis
import logging
from qos.types import QC
from qiskit import QuantumCircuit
import qos.tools


class QOS:
    """Main API that will be exposed to the user"""

    transformer: Transformer
    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        # self.db = database()
        self.logger = logging.getLogger(__name__)

        # Ignoring certain loggers
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("qiskit").setLevel(logging.WARNING)
        logging.getLogger("stevedore").setLevel(logging.WARNING)

        self.transformer = Transformer()
        self.logger.log(10, "QOS API initialized")

        qos.tools.load_qpus("qpus_available")

        print("Available QPUs:")
        for i in range(1, 5):
            print(database.getQPU(i))

    def run(self, circuit: Any) -> int:

        newQC = QC(circuit)

        if type(circuit) == QuantumCircuit:
            newQC.type = "qiskit"

        # Adds the job to the database
        QCId = database.addQC(newQC)
        newQC.id = QCId

        self.logger.log(10, "New job added to the database")

        # Sends the job to the transformer ----------------- This submittion needs to be done in another thread or this will block the API
        self.transformer.submit(newQC)

        return QCId

    def results(self, jobId: int) -> None:

        stat = database.getJobField(jobId, "status").decode("utf-8")

        if stat == "DONE":
            return database.getJobField(jobId, "results")
        else:
            return 1
