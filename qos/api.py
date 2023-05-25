from typing import Any, Dict, List
from qos.types import Job
from .engines.transformer import Transformer
from qos.database import database
import redis


class QOS:
    """Main API that will be exposed to the user"""

    db: database
    transformer: Transformer

    def __init__(self) -> None:
        # self.db = database()
        self.transformer = Transformer()
        # db = redis.Redis(host="localhost", port=6379, db=0)

    def run(self, job: Job) -> int:

        # Adds the job to the database
        newJobId = database.addJob(job)

        # Sends the job to the transformer
        self.transformer.submit(newJobId)
        return newJobId

    def results(self, jobId: int) -> None:

        stat = database.getJobField(jobId, "status").decode("utf-8")

        if stat == "DONE":
            return database.getJobField(jobId, "results")
        else:
            return 1
