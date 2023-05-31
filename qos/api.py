from typing import Any, Dict, List
from qos.types import Job
from .engines.transformer import Transformer
import qos.database as database
import redis
import logging
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

        print("Done")
        exit(0)

    def run(self, job: Job) -> int:

        # Adds the job to the database
        newJobId = database.addJob(job)
        self.logger.log(10, "New job added to the database")

        # Sends the job to the transformer
        self.transformer.submit(newJobId)
        return newJobId

    def results(self, jobId: int) -> None:

        stat = database.getJobField(jobId, "status").decode("utf-8")

        if stat == "DONE":
            return database.getJobField(jobId, "results")
        else:
            return 1
