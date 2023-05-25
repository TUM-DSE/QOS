from typing import Any, Dict, List
from qos.types import Engine, Job
from qos.engines.multiprogrammer import Multiprogrammer
from qos.database import database as db


class Matcher(Engine):
    def submit(self, jobId: int) -> int:

        job = db.getJob(jobId)

        multiprog = Multiprogrammer()

        # Here the matching engine would do its job

        multiprog.submit(jobId)

        return 0
