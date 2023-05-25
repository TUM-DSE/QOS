from typing import Any, Dict, List
from qos.engines.matcher import Matcher
from qos.types import Engine, Job
from qos.database import database as db


class Transformer(Engine):
    def __init__(self) -> None:
        pass

    def submit(self, jobId: int) -> int:

        job = db.getJob(jobId)
        matcher = Matcher()

        # Here the transformer would do its job

        matcher.submit(jobId)

        return 0
